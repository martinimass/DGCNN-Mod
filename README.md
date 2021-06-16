# DGCNN-Mod (TensorFlow)
Implementation of the DGCNN-Mod architecture, by using Tensorflow framework.

## Semantic Segmentation on ArCH dataset
### Data Preparation  
Download Cultural Heritage dataset (**ArCH**) [here](http://archdataset.polito.it/)  and save it in `data/ArCH/`.  
The dataset should be composed by:
- point clouds saved in .txt files, in which the values are separated by a space (77.74060059 30.44643021 314.95358276 136 164 159 0 -0.027905 0.243868 -0.969407)
- a .txt file, called `class_names.txt`, in which there are names of the classes (one for every line)
- a .txt fila, called `structure.txt`, in which there is a single line containing the structure of the point clouds (x y z r g b label nx ny nz)
### Data Preprocessing  
```  
cd sem_seg  
python prepare_data.py  
```  
Processed data will be saved in `data/ArCH_preprocessed_rgb/`.  You can use different parameters:
- '--base_path', default="../data/", help='Path for the output folder'
- '--out_folder', default="ArCH-preprocessed", help='Output folder'
- '--data_path', default="../data/ArCH/", help='Dataset path'
- '--convert', default="rgb", help='convert RGB values to HSV or LAB [rgb|hsv|lab|gray]'

### Training phase
```  
python training.py 
```  
The training phase consists of two steps: the first is another preprocessing, in which the scenes are divided into blocks; the second is the actual training. 
The first phase is executed only the first time; in the following trainings the previously preprocessed files will be used.
Some important parameters you can use:
- '--base_path', default='..data/ArCH_preprocessed_rgb/', help='Path of the preprocessed dataset'
- '--num_point', type=int, default=4096, help='Point number [default: 4096]'
- '--max_epoch', type=int, default=100, help='Epoch to run [default: 100]'
- '--batch_size', type=int, default=4, help='Batch Size during training for each GPU [default: 4]'
- '--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]'
- '--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]'
- '--optimizer', default='adam', help='adam or momentum [default: adam]'
- '--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]'
- '--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]'
  
- '--valid_area', type=int, default=3, help='Which area to use for validation [default: 3]'
- '--test_area', type=int, default=1, help='Which area to use for test [default: 1]'
- '--block_size', type=int, default=5, help='block_size'
- '--stride', type=int, default=5, help='stride'
- '--skip_train', default=False, action='store_true', help='Use it if you want skip the actual training phase'
- '--features', default='x y z r g b', help='input features'
- '--scaler', default='', help="scaler's type : None|scaler1|scaler2"
- '--focal_loss', default=False, action='store_true', help='Use it for the Focal Loss function'

- '--test_epoch', type=int, default=5, help='Frequence for the validation evaluation'
- '--save_epoch', type=int, default=50, help='Frequence for saving weights'
- '--init_knn', type=int, default=3, help='init index for the kNN features'
- '--end_knn', type=int, default=6, help='end index for the kNN features'

- '--model_path', default="", help='checkpoint's path to load'

The partitioning of scenes into blocks can be controlled by the --stride, --block_size parameters. In addition, the values can be normalized by --scaler, which uses the scikit-learn library.
In the training phase it is possible to choose which features to give in input to the network, using the --features parameter. It is necessary to use the same names contained in the `structure.txt` file.
Finally, you can define which of these parameters will be used in the kNN phase, through the parameters --init_knn and --end_knn. The first one must always be equal to 3 (because in the first 3 positions of the features vector will be saved the not normalized coordinates), while the second one indicates the end of the parameters. Example:
- --features 'x y z r g b'
- --init_knn 3
- --end_knn 6
It means that only the features x y z (normalized) will be used in the kNN phase.
  
Scene blocks will be saved in the `data/scene_blocks/` position, inside a folder with a name based on the parameters (for example, train_B_5_S_5_NP_4096). 
Within this folder, each time you run the training with the same settings for the blocks, the results will be saved in a new folder called `log`.

### Test phase
```  
python testing.py 
```  
Some important parameters you can use:
- '--base_path', default='..data/ArCH_preprocessed_rgb/', help='Path of the preprocessed dataset'
- '--log_dir', default="data/scene_blocks/train_B_5_S_5_NP_4096/log/", help='log folder path'
- '--model_path', default="", help='checkpoint path - if empty, best_training.ckpt will be loaded'
- '--test_area', type=int, default=0, help='Test Area ID (default value is readed from parameters.txt)'

The test results will be saved in the `dump` folder, inside the `log` folder. This folder will contain the results of various metrics and the predicted point cloud.


##  How to use a Custom Dataset
To use a custom dataset simply transform it into the same structure of the ArCH dataset
  
## Reference By 
[DGCNN](https://github.com/WangYueFt/dgcnn)
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) <br>  
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>  
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)  
  
## Environments  
- Ubuntu 16.04  
- Python 3.6.7  
- Tensorflow 1.13.2
- scikit-learn 0.22.2
- matplotlib 3.2.1
- h5py 2.10.0
- scikit-image 0.17.2
- seaborn 0.10.1
- pandas 1.0.3

## Citation
Please cite these papers if you want to use it in your work,

	@article{matrone2020comparing,
  		title={Comparing machine and deep learning methods for large 3D heritage semantic segmentation},
  		author={Matrone, Francesca and Grilli, Eleonora and Martini, Massimo and Paolanti, Marina and Pierdicca, Roberto and Remondino, Fabio},
  		journal={ISPRS International Journal of Geo-Information},
		volume={9},
	  	number={9},
	  	pages={535},
	  	year={2020},
	  	publisher={Multidisciplinary Digital Publishing Institute}
	}

	@article{pierdicca2020point,
	  	title={Point cloud semantic segmentation using a deep learning framework for cultural heritage},
	  	author={Pierdicca, Roberto and Paolanti, Marina and Matrone, Francesca and Martini, Massimo and Morbidoni, Christian and Malinverni, Eva Savina and Frontoni, Emanuele and Lingua, Andrea Maria},
	  	journal={Remote Sensing},
	  	volume={12},
	  	number={6},
	  	pages={1005},
	  	year={2020},
	  	publisher={Multidisciplinary Digital Publishing Institute}
	}


## Acknowledgement
This code is based on the structure of [DGCNN](https://github.com/WangYueFt/dgcnn).