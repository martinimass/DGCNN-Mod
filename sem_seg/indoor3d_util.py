import numpy as np
import glob
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)



def log_string(LOG_FOUT,out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)


# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA TO OUR DATA_LABEL FILES
# -----------------------------------------------------------------------------

def collect_point_label(anno_path, out_filename, g_classes, g_class2label, file_format='txt'):
  """ Convert original dataset files to data_label file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.

  Args:
    anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    out_filename: path to save collected points and labels (each line is XYZRGBL)
    file_format: txt or numpy, determines what file format to save.
  Returns:
    None
  Note:
    the points are shifted before save, the most negative point is now at origin.
  """
  points_list = []
 
  for f in glob.glob(os.path.join(anno_path, '*.txt')):
    cls = os.path.basename(f).split('_')[0]
    print("cls: " + str(cls))
    if cls not in g_classes: # note: in some room there is 'stairs' class..
        cls = 'clutter'
    if os.stat(f).st_size > 0: #Otherwise no points for this class!    
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0],1)) * g_class2label[cls]
        if len(points) != 0:
            points_list.append(np.concatenate([points, labels], 1)) # Nx7
  

  data_label = np.concatenate(points_list, 0)
  xyz_min = np.amin(data_label, axis=0)[0:3]
  data_label[:, 0:3] -= xyz_min
  
  if file_format=='txt':
    fout = open(out_filename, 'w')
    for i in range(data_label.shape[0]):
      fout.write('%f %f %f %d %d %d %d\n' % \
              (data_label[i,0], data_label[i,1], data_label[i,2],
               data_label[i,3], data_label[i,4], data_label[i,5],
               data_label[i,6]))
    fout.close()
  elif file_format=='numpy':
    np.save(out_filename, data_label)
  else:
    print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
      (file_format))
    exit()

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------

def sample_data(data, num_sample):
  """ data is in N x ...
    we want to keep num_samplexC of them.
    if N > num_sample, we will randomly keep num_sample of them.
    if N < num_sample, we will randomly duplicate samples.
  """
  N = data.shape[0]
  if (N == num_sample):
    return data, range(N)
  elif (N > num_sample):
    sample = np.random.choice(N, num_sample)
    return data[sample, ...], sample
  else:
    sample = np.random.choice(N, num_sample-N)
    dup_data = data[sample, ...]
    #return np.concatenate([data, dup_data], 0), range(N) + list(sample)
    return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)

def sample_data_label(data, label, num_sample):
  new_data, sample_indices = sample_data(data, num_sample)
  new_label = label[sample_indices]
  return new_data, new_label
  
def room2blocks(data, label, num_point, LOG_FOUT, block_size=1.0, stride=1.0,
        random_sample=False, sample_num=None, sample_aug=1):
  """ Prepare block training data.
  Args:
    data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
      assumes the data is shifted (min point is origin) and aligned
      (aligned with XYZ axis)
    label: N size uint8 numpy array from 0-12
    num_point: int, how many points to sample in each block
    block_size: float, physical size of the block in meters
    stride: float, stride for block sweeping
    random_sample: bool, if True, we will randomly sample blocks in the room
    sample_num: int, if random sample, how many blocks to sample
      [default: room area]
    sample_aug: if random sample, how much aug
  Returns:
    block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
    block_labels: K x num_point x 1 np array of uint8 labels
    
  TODO: for this version, blocking is in fixed, non-overlapping pattern.
  """
  assert(stride<=block_size)
  
  limit = np.amax(data, 0)[0:3]
  log_string(LOG_FOUT," - limits = {}".format(limit))
   
  # Get the corner location for our sampling blocks    
  xbeg_list = []
  ybeg_list = []
  if not random_sample:
    num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
    for i in range(num_block_x):
      for j in range(num_block_y):
        xbeg_list.append(i*stride)
        ybeg_list.append(j*stride)
  else:
    num_block_x = int(np.ceil(limit[0] / block_size))
    num_block_y = int(np.ceil(limit[1] / block_size))
    if sample_num is None:
      sample_num = num_block_x * num_block_y * sample_aug
    for _ in range(sample_num):
      xbeg = np.random.uniform(-block_size, limit[0]) 
      ybeg = np.random.uniform(-block_size, limit[1]) 
      xbeg_list.append(xbeg)
      ybeg_list.append(ybeg)

  log_string(LOG_FOUT," - num_blocks = {} , {} --> {} TOTALS".format(num_block_x, num_block_y, num_block_x * num_block_y ))
        
  # Collect blocks
  block_data_list = []
  block_label_list = []
  idx = 0
  discarded=0
  for idx in range(len(xbeg_list)):
     if idx%100==0:
       print(" - preparing block: " + str(idx+1) + " of "+ str(len(xbeg_list))+ "...")
     xbeg = xbeg_list[idx]
     ybeg = ybeg_list[idx]
     xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
     ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)
     cond = xcond & ycond
     if np.sum(cond) < 10: # discard block if there are less than 100 pts.
       discarded+=1
       continue
     
     block_data = data[cond, :]
     block_label = label[cond]
     
     # randomly subsample data
     block_data_sampled, block_label_sampled = \
       sample_data_label(block_data, block_label, num_point)
     block_data_list.append(np.expand_dims(block_data_sampled, 0))
     block_label_list.append(np.expand_dims(block_label_sampled, 0))

  if discarded>0: log_string(LOG_FOUT," - discarded={}".format(discarded))
  return np.concatenate(block_data_list, 0), \
       np.concatenate(block_label_list, 0)


def room2blocks_plus(data_label, num_point, block_size, stride,
           random_sample, sample_num, sample_aug):
  """ room2block with input filename and RGB preprocessing.
  """
  data = data_label[:,0:6]
  data[:,3:6] /= 255.0
  label = data_label[:,-1].astype(np.uint8)
  
  return room2blocks(data, label, num_point, block_size, stride,
             random_sample, sample_num, sample_aug)
   
def room2blocks_wrapper(data_label_filename, num_point, block_size=1.0, stride=1.0,
            random_sample=False, sample_num=None, sample_aug=1):
  if data_label_filename[-3:] == 'txt':
    data_label = np.loadtxt(data_label_filename)
  elif data_label_filename[-3:] == 'npy':
    data_label = np.load(data_label_filename)
  else:
    print('Unknown file type! exiting.')
    exit()
  return room2blocks_plus(data_label, num_point, block_size, stride,
              random_sample, sample_num, sample_aug)


def room2blocks_plus_normalized(data_label, features, num_point, LOG_FOUT, block_size, stride, random_sample,
                                sample_num, sample_aug, scaler):
  """ room2block, with input filename and RGB preprocessing.
    for each block centralize XYZ, add normalized XYZ
  """

  nfeat=len(features)
  newfeat = 6 #original coords and normalized coords
  total_feat = nfeat + newfeat

  data = data_label[:, 0:nfeat]
  label = data_label[:, -1].astype(np.uint8)
  max_room_x = max(data[:, 0])
  max_room_y = max(data[:, 1])
  max_room_z = max(data[:, 2])

  data_batch, label_batch = room2blocks(data, label, num_point, LOG_FOUT, block_size, stride, random_sample, sample_num, sample_aug)

  new_data_batch = np.zeros((data_batch.shape[0], num_point, total_feat))
  # X,Y,Z
  new_data_batch[:, :, 0:3] = data_batch[:, :, 0:3]

  for b in range(data_batch.shape[0]):
    # X',Y',Z'
    minx = min(data_batch[b, :, 0])
    miny = min(data_batch[b, :, 1])
    new_data_batch[b, :, 3] = data_batch[b, :, 0] - (minx + block_size / 2)
    new_data_batch[b, :, 4] = data_batch[b, :, 1] - (miny + block_size / 2)
    new_data_batch[b, :, 5] = data_batch[b, :, 2]
    # Xn,Yn,Zn
    new_data_batch[b, :, 6] = data_batch[b, :, 0] / max_room_x
    new_data_batch[b, :, 7] = data_batch[b, :, 1] / max_room_y
    new_data_batch[b, :, 8] = data_batch[b, :, 2] / max_room_z
    # other features
    if scaler==None:
      for i,f in enumerate(features):
        if f in ["x","y","z"]: continue
        elif f in ["r","g","b"]:      new_data_batch[b, :, i + newfeat] = data_batch[b, :, i] / 255.0
        elif f in ["h", "s", "v"]:    new_data_batch[b, :, i + newfeat] = data_batch[b, :, i]
        elif f in ["l", "a", "bb"]:   new_data_batch[b, :, i + newfeat] = data_batch[b, :, i]
        elif f in ["nx", "ny", "nz"]: new_data_batch[b, :, i + newfeat] = data_batch[b, :, i]
        else:                         new_data_batch[b, :, i + newfeat] = data_batch[b, :, i]
    else:
      empty_feat=np.zeros((data_batch[b, :].shape[0],1))  #used for the label column
      conc_feats=np.concatenate((data_batch[b, :],empty_feat),axis=1)
      scaled_batch=scaler.transform(conc_feats)
      for i,f in enumerate(features):
        if f in ["x","y","z"]: continue
        else:                         new_data_batch[b, :, i + newfeat] = scaled_batch[:, i]

  return new_data_batch, label_batch

def room2blocks_wrapper_normalized(data_label_filename, features, num_point, LOG_FOUT, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1, scaler=None):
  if data_label_filename[-3:] == 'txt':
    data_label = np.loadtxt(data_label_filename)
  elif data_label_filename[-3:] == 'npy':
    data_label = np.load(data_label_filename)
  else:
    print('Unknown file type! exiting... ', data_label_filename, " - ",data_label_filename[-3:])
    exit()
  return room2blocks_plus_normalized(data_label, features, num_point, LOG_FOUT, block_size, stride, random_sample, sample_num, sample_aug, scaler)

def room2samples(data, label, sample_num_point):
  """ Prepare whole room samples.

  Args:
    data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
      assumes the data is shifted (min point is origin) and
      aligned (aligned with XYZ axis)
    label: N size uint8 numpy array from 0-12
    sample_num_point: int, how many points to sample in each sample
  Returns:
    sample_datas: K x sample_num_point x 9
           numpy array of XYZRGBX'Y'Z', RGB is in [0,1]
    sample_labels: K x sample_num_point x 1 np array of uint8 labels
  """
  N = data.shape[0]
  order = np.arange(N)
  np.random.shuffle(order) 
  data = data[order, :]
  label = label[order]

  batch_num = int(np.ceil(N / float(sample_num_point)))
  sample_datas = np.zeros((batch_num, sample_num_point, 6))
  sample_labels = np.zeros((batch_num, sample_num_point, 1))

  for i in range(batch_num):
    beg_idx = i*sample_num_point
    end_idx = min((i+1)*sample_num_point, N)
    num = end_idx - beg_idx
    sample_datas[i,0:num,:] = data[beg_idx:end_idx, :]
    sample_labels[i,0:num,0] = label[beg_idx:end_idx]
    if num < sample_num_point:
      makeup_indices = np.random.choice(N, sample_num_point - num)
      sample_datas[i,num:,:] = data[makeup_indices, :]
      sample_labels[i,num:,0] = label[makeup_indices]
  return sample_datas, sample_labels

def room2samples_plus_normalized(data_label, num_point):
  """ room2sample, with input filename and RGB preprocessing.
    for each block centralize XYZ, add normalized XYZ as 678 channels
  """
  data = data_label[:,0:6]
  data[:,3:6] /= 255.0
  label = data_label[:,-1].astype(np.uint8)
  max_room_x = max(data[:,0])
  max_room_y = max(data[:,1])
  max_room_z = max(data[:,2])
  
  data_batch, label_batch = room2samples(data, label, num_point)
  new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
  for b in range(data_batch.shape[0]):
    new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x
    new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y
    new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
  new_data_batch[:, :, 0:6] = data_batch
  return new_data_batch, label_batch


def room2samples_wrapper_normalized(data_label_filename, num_point):
  if data_label_filename[-3:] == 'txt':
    data_label = np.loadtxt(data_label_filename)
  elif data_label_filename[-3:] == 'npy':
    data_label = np.load(data_label_filename)
  else:
    print('Unknown file type! exiting.')
    exit()
  return room2samples_plus_normalized(data_label, num_point)
