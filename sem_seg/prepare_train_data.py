import indoor3d_util
import h5py
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def log_string(LOG_FOUT,out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)

def prepare_blocks(data_path, features, NUM_POINT = 4096,  stride = 1.000, block_size = 1.000, scaler_type=""):
    '''
    data_path
    NUM_POINT = 4096
    stride = 1.000      #FLOAT in METERS
    block_size = 1.000  #FLOAT in METERS
    use_norm=False
    '''

    in_path = data_path + "scene_npy/"

    out_path = data_path + "scene_blocks/train_B_" + str(block_size) + "_S_" + str(stride) + "_NP_" + str(NUM_POINT)
    if scaler_type!="": out_path = out_path + "_SC_"+scaler_type
    out_path = out_path + "/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        print(out_path, " already exists!!")
        return out_path

    LOG_FOUT = open(os.path.join(out_path, 'log_blocks.txt'), 'w')
    log_string(LOG_FOUT, "BLOCK_SIZE={} - STRIDE={} - NUM_POINTS={} - SCALER={}".format(block_size,stride,NUM_POINT,scaler_type))

    room_filelist = out_path + "room_filelist.txt"
    all_files = out_path + "all_files.txt"

    scene=sorted(glob.glob(in_path+"*npy"))

    # SCALING
    scaler=None
    if scaler_type != "":
        total = None
        print("Scaling:")
        for i, s in enumerate(scene):
            print("- Load scene", s)
            scene_loaded = np.load(s)
            if i == 0:
                total = scene_loaded
            else:
                total = np.concatenate((total, scene_loaded))
            print(total.shape)
        if scaler_type == "scaler1":
            print("Scaling1...")
            scaler = StandardScaler()
            scaler.fit(total)
        elif scaler_type == "scaler2":
            print("Scaling2...")
            scaler = RobustScaler()
            scaler.fit(total)


    for s in scene:

        name=os.path.basename(s)
        name=name[:-4]
        log_string(LOG_FOUT,name)
        print(" - room2blocks...")
        data, label = indoor3d_util.room2blocks_wrapper_normalized(s, features, NUM_POINT, LOG_FOUT, block_size=block_size, stride= stride, scaler=scaler)
        log_string(LOG_FOUT,"   - data="+str(data.shape))

        h5_out = out_path + name + ".h5"
        print(" - saving .h5 ...")
        save_h5(h5_out,data, label, data_dtype='float_')

        with open(all_files,"a") as f:
            f.write(h5_out + "\n")
        with open(room_filelist,"a") as f:
            for k in range(0,len(data)):
                f.write(name + "\n")

    LOG_FOUT.close()
    return out_path
