This README file contains instructions for processing data and implementing the InterDiff framework.

### 1. Data Preparation

#### Preparing data from the BEHAVE dataset

The config file is located at [./data/cfg/behave.yml](data/cfg/BEHAVE.yml).
You may directly follow the structure in the config file, *i.e.,* putting the data into a folder named [./data/behave/](data/behave/), or you can modify the config file.

Please follow the steps below:
* Download the motion data from [this link](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-30fps-params-v1.tar), and put them into [./data/behave/sequence](data/behave/sequence/). The final file structure should be:
    ```
    ./data/behave/sequence
    |--data_name
    |----object_fit_all.npz # object's pose sequences
    |----smpl_fit_all.npz # human's pose sequences
    ```
* Download object data from [this link](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip), and put them into [./data/behave/objects](data/behave/objects/). The final file structure would be similar to:
    ```
    ./data/behave/objects
    |--object_name
    |----object_name.jpg  # one photo of the object
    |----object_name.obj  # reconstructed 3D scan of the object
    |----object_name.obj.mtl  # mesh material property
    |----object_name_tex.jpg  # mesh texture
    |----object_name_fxxx.ply  # simplified object mesh 
    ```
* Download train/test split from [this link](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/split.json), and put it as [./data/behave/split.json](data/behave/split.json)
* We use the SMPL-H body model, please download the latest model(v1.2) from the [website](https://mano.is.tue.mpg.de/), and indicate its path in the config file ([./data/cfg/behave.yml](data/cfg/BEHAVE.yml)). 
* Generate pseudo contact labels
  ```
  python data/prepare_behave.py
  ```

For further information about the BEHAVE dataset and for download links, please click [here](https://virtualhumans.mpi-inf.mpg.de/behave/) 

> [!IMPORTANT]
> Users may have different sampled point clouds when generating contact labels, which may affect the trained model. We are evaluating this and may release the our model weights and point cloud sample data later.


#### Preparing data from the skeleton-based dataset

The config file is located at [./data/cfg/HOI.yml](data/cfg/HOI.yml).
You may directly follow the structure in the config file, *i.e.,* putting the data into a folder named [./data/hoi/](data/hoi/), or you can modify the config file.

Download data from [this link](https://github.com/HiWilliamWWL/Learn-to-Predict-How-Humans-Manipulate-Large-Sized-Objects-From-Interactive-Motions-objects) and put it in the folder [./data/hoi/](data/hoi/).


### 2. Training

#### Training interaction diffusion on the BEHAVE dataset

```
python train_diffusion_smpl.py
```

#### Training interaction diffusion on the skeleton-based dataset

```
python train_diffusion_skeleton.py
```