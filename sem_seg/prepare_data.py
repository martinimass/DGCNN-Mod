import argparse
import os
from datetime import datetime as dt
import glob
import shutil
from skimage import color
import numpy as np

import indoor3d_util

print("INIT: ",dt.now())


parser = argparse.ArgumentParser()
parser.add_argument('--base_path', default="../data/", help='Path for the output folder')
parser.add_argument('--out_folder', default="ArCH_preprocessed", help='Output folder')
parser.add_argument('--data_path', default="../data/ArCH/", help='Dataset path')
parser.add_argument('--convert', default="rgb", help='convert RGB to HSV or LAB [rgb|hsv|lab|gray]')

FLAGS = parser.parse_args()
base_path = FLAGS.base_path
TEST_NAME = FLAGS.out_folder
CONVERT = FLAGS.convert
TEST_NAME=TEST_NAME+"_"+CONVERT

orig_pts_folder = FLAGS.data_path




#LABELS
labels={}
g_classes=[]
with open(os.path.join(orig_pts_folder,'class_names.txt'), "r") as fr:
    lines=fr.readlines()
    for i,l in enumerate(lines):
        lab=l.rstrip()
        labels[lab]=i
        g_classes.append(lab)



#Output Folders
output_root_folder = os.path.join(base_path,TEST_NAME)
if not os.path.exists(output_root_folder):
    os.mkdir(output_root_folder)

DATA_PATH = os.path.join(output_root_folder , "data")
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
META_PATH = os.path.join(output_root_folder ,"meta")
if not os.path.exists(META_PATH):
    os.makedirs(META_PATH)

shutil.copyfile(os.path.join(orig_pts_folder,'class_names.txt'), os.path.join(META_PATH,"class_names.txt"))
shutil.copyfile(os.path.join(orig_pts_folder,'structure.txt'),os.path.join(META_PATH ,"structure.txt"))


g_class2label = labels

colors=[[0,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,0],[255,0,255],[255,255,0],[255,255,255],
                [0,0,125],[0,125,0],[0,125,125],[125,0,0],[125,0,125],[125,125,0],[125,125,125]]

g_class2color = {cls: colors[i] for i,cls in enumerate(g_classes)}

g_easy_view_labels = range(len(g_classes))  #[0,1,2,3,4,5,6,7,8]
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}


def rgb_to_hsv_skimage(r, g, b):
    rgb_color = [[[r/255.0, g/255.0, b/255.0]]]
    hsv_color = color.rgb2hsv(rgb_color)[0][0]
    return hsv_color[0], hsv_color[1], hsv_color[2]

def rgb_to_lab_skimage(r, g, b):
    rgb_color = [[[r/255.0, g/255.0, b/255.0]]]
    lab_color = color.rgb2lab(rgb_color)[0][0]
    l = lab_color[0] / 100.0
    a = (lab_color[1] + 128)/ 256.0
    b = (lab_color[2] + 128) / 256.0
    return l,a,b

def convert(rgb):
    if CONVERT == "hsv":
        r = int(rgb[0])
        g = int(rgb[1])
        b = int(rgb[2])
        h, s, v = rgb_to_hsv_skimage(r, g, b)
        #h = int(h)
        #s = int(s)
        #v = int(v)
        out = [str(h), str(s), str(v)]
        return out
    elif CONVERT == "lab":
        r = int(rgb[0])
        g = int(rgb[1])
        b = int(rgb[2])
        l, a, b = rgb_to_lab_skimage(r, g, b)
        out = [str(l), str(a), str(b)]
        return out
    else:
        return rgb

def check_nan(vals):
    values=[]
    for v in vals:
        values.append(float(v))
    np_values = np.array(values)
    is_nan = np.isnan(np_values)
    if np.sum(is_nan)>0:
        return True
    else:
        return False


def format_data(orig_pts, out_folder, area, scene_name, labels_to_classes, anno_paths_file):
    #transforms the original point cloud file to the correct format to be fed to DGCNN-Mod data preparation code.

    print("format_data: ",scene_name," - ", dt.now())
    points_in_classes = {}
    for label in labels_to_classes:
        points_in_classes[labels_to_classes[label]] = []

    #Read the structure of the scenes using structure.txt
    with open(os.path.join(os.path.split(orig_pts)[0],"structure.txt"),"r") as fr:
        line=fr.readlines()[0].strip()
        feat_vect=line.split(" ")
    index_label = feat_vect.index("label")

    anno_folder = os.path.join(out_folder, "Annotations")
    if not os.path.exists(anno_folder):
        os.makedirs(anno_folder)

    print(" - creating .txt file of the entire scene....")
    cont=50000
    i=0
    cont_nan = 0
    with open(orig_pts, "r") as f:
        fo=open(os.path.join(out_folder, scene_name + ".txt"), "w")
        fo.close()
        for line in f:
            if i % cont == 0:  print(i)
            i+=1
            vals = line.replace("\n", "").split(" ")
            if check_nan(vals):
                cont_nan+=1
                continue
            label=float(vals[index_label])
            label=int(label)
            if label not in labels_to_classes:
                input("Error: label {} unknown... dict={}".format(label,labels_to_classes))
                continue
            class_name = labels_to_classes[label]
            xyz = vals[feat_vect.index("x"):feat_vect.index("z")+1]
            rgb = vals[feat_vect.index("r"):feat_vect.index("b")+1]
            rgb=convert(rgb)
            points_features = xyz + rgb

            #Check features
            for j,f in enumerate(feat_vect):
                if f not in ["x","y","z","r","g","b","label"]:
                    points_features = points_features + [vals[j]]

            fo = open(os.path.join(out_folder, scene_name + ".txt"), "a")
            fo.write(" ".join(points_features) + "\n")
            fo.close()

            fc=open(os.path.join(anno_folder, class_name + "_1.txt"), "a")
            fc.write(" ".join(points_features) + "\n")
            fc.close()


    print(i)
    print(cont_nan," rows removed with NaN values.")


    with open(anno_paths_file, "a") as f:
        f.write(os.path.join(area, scene_name,"Annotations") + "\n")


def collect_data(data_path,meta_path,anno_paths_file):
    '''
    Creating files: all_data_label.txt and area..._data_label.txt in "meta" folder.
    Creating numpy files for every scene.

    '''
    print("collect_data: ",  dt.now())

    anno_paths = [line.rstrip() for line in open(anno_paths_file)]
    anno_paths = [os.path.join(data_path, p) for p in anno_paths]

    meta_all_data_label = os.path.join(meta_path, "all_data_label.txt")

    output_folder = os.path.join(data_path, "scene_npy")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    f_all = open(meta_all_data_label, "w")

    for anno_path in anno_paths:
        print(anno_path)
        anno_path2=anno_path.replace('\\','/')
        elements = anno_path2.split('/')
        out_filename = elements[-3] + '_' + elements[-2] + '.npy'
        print(" - creating ", out_filename)
        indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), g_classes, g_class2label, 'numpy')

        f_all.write(out_filename + "\n")
        area_data_label = os.path.join(meta_path, elements[-3].replace("_", "").lower() + '_data_label.txt')
        with open(area_data_label, "a") as f:
            f.write(os.path.join(output_folder, out_filename) + "\n")


    f_all.close()

def main(orig_pts_folder, data_path, meta_path ,labels):
    print("INIT prepare_data: ", dt.now())
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)


    scene=[]
    files=sorted(glob.glob(orig_pts_folder+"*.txt"))
    i=1
    for f in files:
        if "class_names" in f:
            continue
        if "structure" in f:
            continue
        scene.append((f,"Area_"+str(i)))
        i+=1

    anno_paths_file = os.path.join(meta_path, "anno_paths.txt")

    for scene_path,area in scene:
        base_name= os.path.basename(scene_path)
        sub_scene_name = base_name[:-4] + "_1"  #ROOM of the Area

        data_path = os.path.join(output_root_folder , "data")
        out_folder = os.path.join(data_path , area, sub_scene_name)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        labels_to_classes={}
        for k in labels:
            labels_to_classes[labels[k]]=k

        format_data(scene_path, out_folder, area, sub_scene_name, labels_to_classes, anno_paths_file)

    collect_data(data_path, meta_path, anno_paths_file)

    print("prepare_data completed: ",dt.now())




main(orig_pts_folder, DATA_PATH, META_PATH ,labels)

