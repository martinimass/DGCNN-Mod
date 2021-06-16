import argparse
import os
import sys
import glob
from datetime import datetime as dt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from sem_seg.model import *
import indoor3d_util
import provider
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score, accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("INIT: ",dt.now())

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--base_path', default='../data/Arch_preprocessed_rgb/', help='Path of the preprocessed dataset')
parser.add_argument('--log_dir', default="data/scene_blocks/train_B_5_S_5_NP_4096/log/", help='log folder path')
parser.add_argument('--model_path', default="", help='checkpoint path - if empty, best_training.ckpt will be loaded')
parser.add_argument('--test_area', type=int, default=0, help='Test Area ID [default value is readed from parameters.txt]')

parser.add_argument('--batch_size', type=int, default=1, help='Batch Size [default: 1]')


def str2bool(s):
  if s =="True":  return True
  else: return False


def best_weight():
  ws=glob.glob(LOG_DIR + "weights/best_training*.ckpt.index")
  print(LOG_DIR)
  print(ws)
  return ws[0][:-6]  # remove final part (.index)


FLAGS = parser.parse_args()

base_path = FLAGS.base_path
LOG_DIR = base_path + FLAGS.log_dir
BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu

if FLAGS.model_path =="":
  MODEL_PATH = best_weight()
else:
  MODEL_PATH = LOG_DIR + "weights/" + FLAGS.model_path
print("Weights --> {}".format(MODEL_PATH))


#Parameters parsing from parameters.txt
with open(LOG_DIR+"parameters.txt","r") as fr:
  dizio={}
  for l in fr:
    if "FEATURES" in l:
      k="FEATURES"
      v=l[9:]
    else:
      k,v=l.split(" ")

    dizio[k]=v.rstrip()
BLOCK_SIZE = int(float(dizio["BLOCK_SIZE"]))
STRIDE = int(float(dizio["STRIDE"]))
NUM_POINT = int(dizio["NUM_POINT"])

if FLAGS.test_area==0:
  test_area = dizio["TEST_AREA"]
else:
  test_area="Area_"+str(FLAGS.test_area)

valid_area = dizio["VALID_AREA"]
FEATURES = dizio["FEATURES"]
INIT_KNN = int(dizio["INIT_KNN"])
END_KNN = int(dizio["END_KNN"])
SCALER = dizio["SCALER"]
FOCAL_LOSS = str2bool(dizio["FOCAL_LOSS"])


print("Parametri letti dal file parameters.txt:")
print("BLOCK_SIZE={}".format(BLOCK_SIZE))
print("STRIDE={}".format(STRIDE))
print("NUM_POINT={}".format(NUM_POINT))
print("valid_area=" + valid_area)
print("test_area=" + test_area)
print("FEATURES={}".format(FEATURES))
print("INIT_KNN={}".format(INIT_KNN))
print("END_KNN={}".format(END_KNN))
print("SCALER={}".format(SCALER))
print("FOCAL_LOSS={}".format(FOCAL_LOSS))


DUMP_DIR = LOG_DIR + "dump_"+str(test_area)+"/"
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


NUM_FEATURES = len(FEATURES.split(" ")) + 3
print("NUM_FEATURES --> {}".format(NUM_FEATURES))

meta_path = base_path + "meta/"
data_path = base_path + "data/"

class_names=[]
with open(meta_path+"class_names.txt", "r") as fr:
  lines=fr.readlines()
  NUM_CLASSES=len(lines)
  for l in lines:
    class_names.append(l.rstrip())


test_area_id=test_area[5:].rstrip()
room_path=meta_path+'area'+test_area_id+'_data_label.txt'
ROOM_PATH_LIST = [line.rstrip() for line in open(room_path)]

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)



def sub_class_names(class_names, true_labels, pred_labels):
  unique_true = set(true_labels)
  unique_pred = set(pred_labels)
  unique_labels = unique_true | unique_pred
  sub = []
  for i in unique_labels:
    sub.append(class_names[i])
  print("sub_class_names:", class_names, unique_labels, sub)
  return sub


def plot_confusion_matrix_seaborn(y_true, y_pred, classes, figsize=(10, 7), fontsize=12, show=False, name="", title="", norm=None):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=norm)

    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    fig = plt.figure(figsize=figsize)
    if norm==None:
      fmt="d"
    else:
      fmt=".4f"
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cmap="plasma")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    if show:  plt.show()
    if name != "": plt.savefig(name)

    return fig


def load_test_data(train_path):

  ALL_FILES = provider.getDataFiles(train_path + 'all_files.txt')
  room_filelist = [line.rstrip() for line in open(train_path + 'room_filelist.txt')]

  # Load ALL data
  print("Load ALL Data...")
  data_batch_list = []
  label_batch_list = []
  for h5_filename in ALL_FILES:
    print(h5_filename)
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
  data_batches = np.concatenate(data_batch_list, 0)
  label_batches = np.concatenate(label_batch_list, 0)
  print("data_batches=", data_batches.shape)
  print("label_batches=", label_batches.shape)


  train_idxs = []
  test_idxs = []
  for i, room_name in enumerate(room_filelist):
    if test_area in room_name:
      test_idxs.append(i)
    else:
      train_idxs.append(i)

  train_data = data_batches[train_idxs, ...]
  train_label = label_batches[train_idxs]
  test_data = data_batches[test_idxs, ...]
  test_label = label_batches[test_idxs]
  print("TRAIN --> ", train_data.shape, train_label.shape)
  print("TEST -- > ", test_data.shape, test_label.shape)

  return test_data, test_label

def evaluate():
  is_training = False
   
  with tf.device('/gpu:'+str(GPU_INDEX)):
    pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FEATURES)
    is_training_pl = tf.placeholder(tf.bool, shape=())

    pred       = get_model(NUM_CLASSES, pointclouds_pl, is_training_pl, init_knn=INIT_KNN, end_knn=END_KNN)

    if FOCAL_LOSS:       loss = focal_loss(pred, labels_pl, num_classes=NUM_CLASSES)
    else:                loss = get_loss(pred, labels_pl)

    pred_softmax = tf.nn.softmax(pred)

    saver = tf.train.Saver()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess = tf.Session(config=config)


  saver.restore(sess, MODEL_PATH)
  log_string("Model restored.")

  ops = {'pointclouds_pl': pointclouds_pl,
       'labels_pl': labels_pl,
       'is_training_pl': is_training_pl,
       'pred': pred,
       #'feat': feat,
       'pred_softmax': pred_softmax,
       'loss': loss}
  
  total_correct = 0
  total_seen = 0


  for room_path in ROOM_PATH_LIST:

    out_path = os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4])

    print(room_path)
    # Evaluate room one by one.
    a,b = eval_one_epoch(sess, ops, room_path, out_path)
    total_correct += a
    total_seen += b

  log_string('all room eval accuracy: %f'% (total_correct / float(total_seen)))



def sub_data_batch(data_batch):

  feat_map={}
  with open(meta_path + "structure.txt", "r") as fr:
    lines = fr.readlines()
    line=lines[0]
    line=line.rstrip()
    feat_vect=line.split(" ")
    feat_vect.remove("label")
    for i,f in enumerate(feat_vect):
      if f in ["x", "y", "z"]:
        pos = i
      else:
        pos = i + 6
      feat_map[f]=pos
  print(feat_map)

  sub = data_batch[:, :, 3:9]


  for f in FEATURES.split(" "):
    if f in ["x","y","z"]:  continue
    if f in feat_map:
      print(" - added ",f)
      pos = feat_map[f]
      feat = data_batch[:, :, pos:(pos+1)]
      sub = np.concatenate((sub, feat), axis=2)
    elif f in ["l","h"]:
      print(" - added ", f)
      pos = feat_map["r"]
      feat = data_batch[:, :, pos:(pos + 1)]
      sub = np.concatenate((sub, feat), axis=2)
    elif f in ["a","s"]:
      print(" - added ", f)
      pos = feat_map["g"]
      feat = data_batch[:, :, pos:(pos + 1)]
      sub = np.concatenate((sub, feat), axis=2)
    elif f in ["bb","v"]:
      print(" - added ", f)
      pos = feat_map["b"]
      feat = data_batch[:, :, pos:(pos + 1)]
      sub = np.concatenate((sub, feat), axis=2)

  return sub

def eval_one_epoch(sess, ops, room_path, out_path):
  global class_names
  is_training = False
  total_correct = 0
  total_seen = 0
  loss_sum = 0
  out_gt_pred_filename = out_path + "_gt_pred.txt"

  fout_gt_pred = open(out_gt_pred_filename, 'w')

  split1=LOG_DIR.split("/")
  new_path="/".join(split1[:-2])+"/"
  current_data, current_label = load_test_data(new_path)

  current_data_features = sub_data_batch(current_data)

  current_label = np.squeeze(current_label)

  
  file_size = current_data_features.shape[0]
  num_batches = file_size // BATCH_SIZE

  true_labels=[]
  pred_labels=[]
  pred_vals=[]

  for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = (batch_idx+1) * BATCH_SIZE
    cur_batch_size = end_idx - start_idx
    
    feed_dict = {ops['pointclouds_pl']: current_data_features[start_idx:end_idx, :, :],
           ops['labels_pl']: current_label[start_idx:end_idx],
           ops['is_training_pl']: is_training}
    loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']], feed_dict=feed_dict)

    pred_label = np.argmax(pred_val, 2) # BxN

    pred_labels += pred_label.flatten().tolist()
    true_labels += current_label[start_idx:end_idx].flatten().tolist()

    pred_val2 = np.argsort(1.0-pred_val)
    pred_vals += pred_val2.tolist()
    
    # Save prediction labels to OBJ file
    for b in range(BATCH_SIZE):
      pts = current_data[start_idx+b, :, :]
      l = current_label[start_idx+b,:]

      pts[:,3:6] *= 255.0
      pred = pred_label[b, :]
      for i in range(NUM_POINT):

        base = '{:.8f} {:.8f} {:.8f} {} {}'.format(pts[i, 0], pts[i, 1], pts[i, 2], l[i], pred[i])
        fout_gt_pred.write(base+"\n")



    
    correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
    total_correct += correct
    total_seen += (cur_batch_size*NUM_POINT)
    loss_sum += (loss_val*BATCH_SIZE)

  log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
  log_string('eval accuracy: %f'% (total_correct / float(total_seen)))

  if len(class_names)!=len(set(true_labels)):
    class_names=sub_class_names(class_names,true_labels, pred_labels)

  reports(true_labels, pred_labels, pred_vals)

  fout_gt_pred.close()


  return total_correct, total_seen


def reports(true_labels, pred_labels, pred_vals):
  acc = accuracy_score(true_labels, pred_labels)
  report = classification_report(true_labels, pred_labels, digits=4, target_names=class_names)
  cm = confusion_matrix(true_labels, pred_labels)
  log_string("\nAccuracy: \n" + str(acc))
  log_string('\nreport:\n' + str(report))
  log_string('\ncm:\n' + str(cm))

  jacc = jaccard_score(true_labels, pred_labels, average=None)
  log_string('\nMean Jaccard (Mean_IoU):\n' + str(np.mean(jacc)))
  log_string('\nJaccard:')
  for i, n in enumerate(class_names):
    log_string(" - {}: {}".format(n, jacc[i]))


  plot_confusion_matrix_seaborn(true_labels, pred_labels, classes=class_names, title='Confusion Matrix', show=False, name=DUMP_DIR + "/eval", norm=None)
  plot_confusion_matrix_seaborn(true_labels, pred_labels, classes=class_names, title='Confusion Matrix, Normalized', show=False, name=DUMP_DIR + "/eval_norm", norm="true", figsize=(10, 7))

  cmc(true_labels, pred_vals, classes=len(class_names), plot_save=DUMP_DIR + "cmc.png")

def cmc(true_labels, pred_vals, classes=10, plot_save=""):
  predictions = np.array(pred_vals)
  predictions = predictions.reshape((predictions.shape[0]*predictions.shape[1],predictions.shape[2]))
  labels = np.array(true_labels)

  ranks = np.zeros(classes)

  for i in range(len(labels)):
    if labels[i] in predictions[i]:
      firstOccurance = np.argmax(predictions[i] == labels[i])
      for j in range(firstOccurance, classes):
        ranks[j] += 1

  cmcScores = [float(i) / float(len(labels)) for i in ranks]
  log_string("\ncmcScores: {}\n".format(cmcScores))

  plt.clf()
  plt.plot(range(1,classes+1), cmcScores)
  plt.xlabel("Rank")
  plt.ylabel("Accuracy")

  plt.savefig(plot_save)

if __name__=='__main__':
  with tf.Graph().as_default():
    evaluate()
  LOG_FOUT.close()
  print("DONE: ", dt.now())


