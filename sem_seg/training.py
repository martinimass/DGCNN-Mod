import argparse
import numpy as np
import tensorflow as tf
import socket
import pandas as pd
import seaborn as sns

import os
import sys
import prepare_train_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
from sem_seg.model import *

from datetime import datetime as dt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

print("INIT: ",dt.now())


parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='the number of GPUs to use [default: 1]')
parser.add_argument('--base_path', default='../data/Arch_preprocessed_rgb/', help='Path of the preprocessed dataset')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training for each GPU [default: 4]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--valid_area', type=int, default=3, help='Which area to use for validation [default: 3]')
parser.add_argument('--test_area', type=int, default=1, help='Which area to use for test [default: 1]')
parser.add_argument('--block_size', type=int, default=5, help='block_size')
parser.add_argument('--stride', type=int, default=5, help='stride')
parser.add_argument('--skip_train', default=False, action='store_true', help='Use it if you want skip the training phase')
parser.add_argument('--features', default='x y z r g b', help='input features')
parser.add_argument('--scaler', default='', help="scaler's type : None|scaler1|scaler2" )
parser.add_argument('--focal_loss', default=False, action='store_true', help='Use it for the Focal Loss function')

parser.add_argument('--test_epoch', type=int, default=5, help='Frequence for the validation evaluation')
parser.add_argument('--save_epoch', type=int, default=50, help='Frequence for saving weights')
parser.add_argument('--init_knn', type=int, default=3, help='init index for the kNN features')
parser.add_argument('--end_knn', type=int, default=6, help='end index for the kNN features')

parser.add_argument('--model_path', default="", help='checkpoint path')

FLAGS = parser.parse_args()

TOWER_NAME = 'tower'

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BLOCK_SIZE = FLAGS.block_size
STRIDE = FLAGS.stride
SKIP_TRAIN = FLAGS.skip_train
FEATURES = FLAGS.features
SCALER = FLAGS.scaler
FOCAL_LOSS = FLAGS.focal_loss

TEST_EPOCH = FLAGS.test_epoch
SAVE_EPOCH = FLAGS.save_epoch
INIT_KNN = FLAGS.init_knn
END_KNN = FLAGS.end_knn


MODEL_PATH = FLAGS.model_path
print("Weights --> {}".format(MODEL_PATH))

base_path = FLAGS.base_path
meta_path = base_path + "meta/"
data_path = base_path + "data/"

list_features=FEATURES.split(" ")
list_features[-1]=list_features[-1].strip()
NUM_FEATURES = len(list_features) + 3 # Added Normalized Coords

with open(meta_path + "structure.txt", "r") as fr:
  lines = fr.readlines()
  line = lines[0]
  line = line.rstrip()
  feat_vect = line.split(" ")

feat_vect.remove("label")


class_names=[]
with open(meta_path+"class_names.txt", "r") as fr:
  lines=fr.readlines()
  NUM_CLASSES=len(lines)
  for l in lines:
    class_names.append(l.rstrip())


# STEP 1: Blocks Creation
train_path=prepare_train_data.prepare_blocks(data_path, feat_vect, NUM_POINT = NUM_POINT,  stride = STRIDE, block_size = BLOCK_SIZE,  scaler_type=SCALER)

if SKIP_TRAIN:
  print("Training Phase skipped...")
  sys.exit(0)

# Log
LOG_DIR = train_path+"log"

if not os.path.exists(LOG_DIR):
  os.mkdir(LOG_DIR)
  print(LOG_DIR, " created")
else:
  i=1
  LOG_DIR_count = LOG_DIR + "-" + str(i)
  while(os.path.exists(LOG_DIR_count)):
    i+=1
    LOG_DIR_count = LOG_DIR + "-" + str(i)
  os.mkdir(LOG_DIR_count)
  print(LOG_DIR_count, " created")
  LOG_DIR = LOG_DIR_count


valid_area = 'Area_'+str(FLAGS.valid_area)
test_area = 'Area_'+str(FLAGS.test_area)


def save_txt_parameters():
  with open(LOG_DIR+"/parameters.txt","w") as fw:
    fw.write("BATCH_SIZE {}\n".format(BATCH_SIZE))
    fw.write("NUM_POINT {}\n".format(NUM_POINT))
    fw.write("MAX_EPOCH {}\n".format(MAX_EPOCH))
    fw.write("BASE_LEARNING_RATE {}\n".format(BASE_LEARNING_RATE))
    fw.write("MOMENTUM {}\n".format(MOMENTUM))
    fw.write("OPTIMIZER {}\n".format(OPTIMIZER))
    fw.write("DECAY_STEP {}\n".format(DECAY_STEP))
    fw.write("DECAY_RATE {}\n".format(DECAY_RATE))
    fw.write("BLOCK_SIZE {}\n".format(BLOCK_SIZE))
    fw.write("STRIDE {}\n".format(STRIDE))
    fw.write("VALID_AREA {}\n".format(valid_area))
    fw.write("TEST_AREA {}\n".format(test_area))
    fw.write("FEATURES {}\n".format(FEATURES))
    fw.write("INIT_KNN {}\n".format(INIT_KNN))
    fw.write("END_KNN {}\n".format(END_KNN))
    fw.write("SCALER {}\n".format(SCALER))
    fw.write("FOCAL_LOSS {}\n".format(FOCAL_LOSS))

save_txt_parameters()

PLOT_DIR = LOG_DIR + "/plot"
if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)
WEIGHTS_DIR = LOG_DIR + "/weights"
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)


LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)

log_string(str(FLAGS))
log_string("NUM_CLASSES: "+str(NUM_CLASSES))
log_string("NUM_FEATURES --> {}".format(NUM_FEATURES))
log_string("train_path: "+train_path)
log_string("Area_VALID="+valid_area)
log_string("Area_TEST="+test_area)


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()


ALL_FILES = provider.getDataFiles(train_path+'all_files.txt')
room_filelist = [line.rstrip() for line in open(train_path+'room_filelist.txt')]



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
        pos = i + 6   #features from 0 to 5 are: x,y,z,xn,yn,zn
      feat_map[f]=pos
  print(feat_map)

  sub = data_batch[:, :, 3:9]

  FEATURES=FLAGS.features.rstrip()
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

# Load ALL data
log_string("Load ALL Data...")
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
  log_string(h5_filename)
  data_batch, label_batch = provider.loadDataFile(h5_filename)
  sub=sub_data_batch(data_batch)
  data_batch_list.append(sub)
  label_batch_list.append(label_batch)
  log_string("- data_batches={}".format(sub.shape))
  log_string("- label_batches={}".format(label_batch.shape))

data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
log_string("TOTAL data_batches={}".format(data_batches.shape))
log_string("TOTAL label_batches={}".format(label_batches.shape))



train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):
  if valid_area in room_name:
    test_idxs.append(i)
  elif test_area in room_name:
    continue
  else:
    train_idxs.append(i)

train_data = data_batches[train_idxs,...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]
log_string("")
log_string("TRAIN --> {} - {}".format(train_data.shape, train_label.shape))
log_string("VALID -- > {} - {}".format(test_data.shape, test_label.shape))



def get_learning_rate(batch):
  learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            DECAY_STEP,          # Decay step.
            DECAY_RATE,          # Decay rate.
            staircase=True)
  learning_rate = tf.maximum(learning_rate, 0.00001) #CLIP THE LEARNING RATE!!
  return learning_rate        

def get_bn_decay(batch):
  bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch*BATCH_SIZE,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
  bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
  return bn_decay

def average_gradients(tower_grads):
  """Calculate average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been 
     averaged across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def plot_confusion_matrix_seaborn(y_true, y_pred, classes, figsize=(10, 7), fontsize=14, show=False, name="",
                                    title=""):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    classes: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(
      cm, index=classes, columns=classes,
    )
    fig = plt.figure(figsize=figsize)
    try:
      heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
      raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    if show:  plt.show()
    if name != "": plt.savefig(name)

    return fig


def train():
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    batch = tf.Variable(0, trainable=False)
    
    bn_decay = get_bn_decay(batch)
    tf.summary.scalar('bn_decay', bn_decay)

    learning_rate = get_learning_rate(batch)
    tf.summary.scalar('learning_rate', learning_rate)
    
    trainer = tf.train.AdamOptimizer(learning_rate)
    
    tower_grads = []
    pointclouds_phs = []
    labels_phs = []
    is_training_phs =[]

    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpu):  # PYTHON 3
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
      
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FEATURES)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            pointclouds_phs.append(pointclouds_pl)
            labels_phs.append(labels_pl)
            is_training_phs.append(is_training_pl)
      
            pred = get_model(NUM_CLASSES, pointclouds_phs[-1], is_training_phs[-1], init_knn=FLAGS.init_knn, end_knn=FLAGS.end_knn, bn_decay=bn_decay)
            if FOCAL_LOSS:
              loss = focal_loss(pred, labels_phs[-1], num_classes=NUM_CLASSES)
            else:
              loss = get_loss(pred, labels_phs[-1])
            tf.summary.scalar('loss', loss)
            pred_softmax = tf.nn.softmax(pred)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_phs[-1]))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            tf.get_variable_scope().reuse_variables()

            grads = trainer.compute_gradients(loss)

            tower_grads.append(grads)
    
    grads = average_gradients(tower_grads)

    train_op = trainer.apply_gradients(grads, global_step=batch)
    
    saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep=10)
    
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    if MODEL_PATH!="":
      saver.restore(sess, data_path+MODEL_PATH)
      log_string("Model restored.")

    # Add summary writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    # Init variables for two GPUs
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)



    ops = {'pointclouds_phs': pointclouds_phs,
         'labels_phs': labels_phs,
         'is_training_phs': is_training_phs,
         'pred': pred,
         'loss': loss,
         'pred_softmax':  pred_softmax,
         'train_op': train_op,
         'merged': merged,
         'step': batch}

    log_string("FOCAL_LOSS = {}".format(FOCAL_LOSS))

    test_results =  [0, 0] #[accuracy, epoch]
    train_results = [0, 0] #[accuracy, epoch]
    for epoch in range(MAX_EPOCH):
      log_string('**** EPOCH %03d ****' % (epoch))
      log_string(str(dt.now()))
      sys.stdout.flush()

      acc = train_one_epoch(sess, ops, train_writer, epoch)

      if train_results[0] < acc:
          train_results[0] = acc
          train_results[1] = epoch
          save_path = saver.save(sess, os.path.join(WEIGHTS_DIR, 'best_training.ckpt'))
          log_string("Model saved in file: %s" % save_path)

      if epoch % TEST_EPOCH == 0:
        current_test_acc = eval_one_epoch(sess, ops, epoch) #test_data, test_label, test_batches,
        if test_results[0] < current_test_acc:
          test_results[0] = current_test_acc
          test_results[1] = epoch
          save_path = saver.save(sess, os.path.join(WEIGHTS_DIR, 'best_valid.ckpt'))
          log_string("Model saved in file: %s" % save_path)

      # Save the variables to disk.
      if epoch % SAVE_EPOCH == 0:
        save_path = saver.save(sess, os.path.join(WEIGHTS_DIR,'epoch_' + str(epoch)+'_acc_' + str(round(acc,4))+'.ckpt'))
        log_string("Model saved in file: %s" % save_path)

    save_path = saver.save(sess, os.path.join(WEIGHTS_DIR, 'final.ckpt'))
    log_string("Model saved in file: %s" % save_path)

    log_string("Best Training: acc={} - epoch={}".format(train_results[0],train_results[1]))
    log_string("Best Validation: acc={} - epoch={}".format(test_results[0], test_results[1]))

    #RENAME WEIGHTS
    end1=".data-00000-of-00001"
    end2=".index"
    end3=".meta"
    os.rename(os.path.join(WEIGHTS_DIR, 'best_training.ckpt' + end1), os.path.join(WEIGHTS_DIR,
                                                                                   'best_training_epoch_' + str(
                                                                                     train_results[1]) + '_acc_' + str(
                                                                                     round(train_results[0],
                                                                                           4)) + '.ckpt' + end1))
    os.rename(os.path.join(WEIGHTS_DIR, 'best_training.ckpt' + end2), os.path.join(WEIGHTS_DIR,
                                                                                   'best_training_epoch_' + str(
                                                                                     train_results[1]) + '_acc_' + str(
                                                                                     round(train_results[0],
                                                                                           4)) + '.ckpt' + end2))
    os.rename(os.path.join(WEIGHTS_DIR, 'best_training.ckpt' + end3), os.path.join(WEIGHTS_DIR,
                                                                                   'best_training_epoch_' + str(
                                                                                     train_results[1]) + '_acc_' + str(
                                                                                     round(train_results[0],
                                                                                           4)) + '.ckpt' + end3))
    os.rename(os.path.join(WEIGHTS_DIR, 'best_valid.ckpt' + end1), os.path.join(WEIGHTS_DIR,
                                                                                   'best_valid_epoch_' + str(
                                                                                     test_results[1]) + '_acc_' + str(
                                                                                     round(test_results[0],
                                                                                           4)) + '.ckpt' + end1))
    os.rename(os.path.join(WEIGHTS_DIR, 'best_valid.ckpt' + end2), os.path.join(WEIGHTS_DIR,
                                                                                   'best_valid_epoch_' + str(
                                                                                     test_results[1]) + '_acc_' + str(
                                                                                     round(test_results[0],
                                                                                           4)) + '.ckpt' + end2))
    os.rename(os.path.join(WEIGHTS_DIR, 'best_valid.ckpt' + end3), os.path.join(WEIGHTS_DIR,
                                                                                   'best_valid_epoch_' + str(
                                                                                     test_results[1]) + '_acc_' + str(
                                                                                     round(test_results[0],
                                                                                           4)) + '.ckpt' + end3))


def sub_class_names(x, y, names):
    u = unique_labels(x, y)
    names2 = [names[i] for i in u]
    return names2

def train_one_epoch(sess, ops, train_writer, epoch):
  """ ops: dict mapping from string to tf ops """
  is_training = True
  
  log_string('----')
  current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
  
  file_size = current_data.shape[0]
  num_batches = file_size // (FLAGS.num_gpu * BATCH_SIZE) 
  
  total_correct = 0
  total_seen = 0
  loss_sum = 0

  pred_labels=[]
  true_labels=[]
  
  for batch_idx in range(num_batches):
    if batch_idx % 50 == 0:
      print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
    start_idx_0 = batch_idx * BATCH_SIZE
    end_idx_0 = (batch_idx+1) * BATCH_SIZE

    feed_dict = {ops['pointclouds_phs'][0]: current_data[start_idx_0:end_idx_0, :, :],
                 ops['labels_phs'][0]: current_label[start_idx_0:end_idx_0],
                 ops['is_training_phs'][0]: is_training}

    
    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                     feed_dict=feed_dict)
    train_writer.add_summary(summary, step)
    pred_val = np.argmax(pred_val, 2)

    pred_labels += pred_val.flatten().tolist()
    true_labels += current_label[start_idx_0:end_idx_0].flatten().tolist()
    correct = np.sum(pred_val == current_label[start_idx_0:end_idx_0])

    total_correct += correct
    total_seen += (BATCH_SIZE*NUM_POINT)
    loss_sum += loss_val

  accuracy = total_correct / float(total_seen)
  print("num_batches: " + str(num_batches))
  log_string('mean loss: %f' % (loss_sum / float(num_batches)))
  log_string('accuracy: %f' % (accuracy))

  sub_names = sub_class_names(true_labels, pred_labels, class_names)
  report=classification_report(true_labels,pred_labels, target_names=sub_names, digits=4)
  cm = confusion_matrix(true_labels,pred_labels)
  log_string('report:\n'+report)
  log_string('cm:\n' + str(cm))

  #plot_confusion_matrix_seaborn(true_labels,pred_labels, classes=sub_names, title='Confusion matrix,', show=False, name=PLOT_DIR+"/train-"+str(epoch))

  return accuracy



def eval_one_epoch(sess, ops, epoch):
  log_string('----VALIDATION---')

  current_data=test_data
  current_label=test_label

  file_size = current_data.shape[0]
  num_batches = file_size // BATCH_SIZE

  is_training = False
  total_correct = 0
  total_seen = 0
  loss_sum = 0

  true_labels=[]
  pred_labels=[]
  for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = (batch_idx + 1) * BATCH_SIZE
    cur_batch_size = end_idx - start_idx

    feed_dict = {ops['pointclouds_phs'][0]: current_data[start_idx:end_idx, :, :],
                 ops['labels_phs'][0]: current_label[start_idx:end_idx],
                 ops['is_training_phs'][0]: is_training}
    loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']], feed_dict=feed_dict)

    pred_label = np.argmax(pred_val, 2)  # BxN

    pred_labels += pred_label.flatten().tolist()
    true_labels += current_label[start_idx:end_idx].flatten().tolist()

    correct = np.sum(pred_label == current_label[start_idx:end_idx, :])
    total_correct += correct
    total_seen += (cur_batch_size * NUM_POINT)
    loss_sum += (loss_val * BATCH_SIZE)

  if total_seen==0:
    accuracy=0
  else:
    accuracy = total_correct / float(total_seen)

  log_string('eval mean loss: %f' % (loss_sum / float(total_seen / NUM_POINT)))
  log_string('eval accuracy: %f' % (accuracy))

  sub_names = sub_class_names(true_labels, pred_labels, class_names)
  report = classification_report(true_labels, pred_labels, target_names=sub_names, digits=4)
  cm = confusion_matrix(true_labels, pred_labels)
  log_string('report:\n' + report)
  log_string('cm:\n' + str(cm))

  plot_confusion_matrix_seaborn(true_labels, pred_labels, classes=sub_names, title='Confusion matrix,', show=False, name=PLOT_DIR + "/test-" + str(epoch))

  return accuracy


def plot_train():
  log_path = LOG_DIR+"/log_train.txt"
  plot_save = log_path[:-13] + "training.png"
  acc = []
  val_acc = []
  with open(log_path, "r") as f:
    for l in f:
      if "eval accuracy:" in l:
        if l.index("eval accuracy:") == 0:
          a = float(l[14:])
          val_acc.append(a)
          continue
      if "accuracy:" in l:
        if l.index("accuracy:") == 0:
          a = float(l[9:])
          acc.append(a)
          continue

  plt.clf()
  plt.plot(range(len(acc)), acc)
  epochs_val = range(0, len(val_acc) * TEST_EPOCH, TEST_EPOCH)
  plt.plot(epochs_val, val_acc)
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")

  plt.savefig(plot_save)
  #plt.show()


if __name__ == "__main__":
  train()
  plot_train()
  LOG_FOUT.close()
  print("training.py completed: ", dt.now())
