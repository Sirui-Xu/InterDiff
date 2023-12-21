This README file contains instructions for processing data and implementing the InterDiff framework.

### 1. Data Preparation

#### Preparing data from the BEHAVE dataset

The config file is located at [./data/cfg/behave.yml](data/cfg/BEHAVE.yml).
You may follow the structure in the config file, *i.e.,* putting the data into a folder named [./data/behave/](data/behave/), or you can modify the config file.

You can download our processed data from [this link](), unzip the file, and put it under [./data/](data/).

Or follow the steps below:
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
> Users may have different sampled point clouds when generating contact labels, which may affect our provided pretrained model. 
> We provide our processed BEHAVE data on [this link](https://drive.google.com/file/d/1txDAK9YmLT13hGMlNAwHAP--wiN-kMQr/view?usp=sharing). You may consider trying our pretrained model with these data.

#### Preparing data from the skeleton-based dataset

The config file is located at [./data/cfg/HOI.yml](data/cfg/HOI.yml).
You may follow the structure in the config file, *i.e.,* putting the data into a folder named [./data/hoi/](data/hoi/), or you can modify the config file.

Download data from [this link](https://github.com/HiWilliamWWL/Learn-to-Predict-How-Humans-Manipulate-Large-Sized-Objects-From-Interactive-Motions-objects) and put it in the folder [./data/hoi/](data/hoi/).


### 2. Training

#### Training interaction diffusion on the BEHAVE dataset

```
python train_diffusion_smpl.py
```

You may evaluate a plain diffusion for short-term generation with checkpoint indicated:

```
python train_diffusion_smpl.py --mode test --resume_checkpoint PATH/TO/YOUR/CHECKPOINT.ckpt 
```

#### Training interaction diffusion on the skeleton-based dataset

```
python train_diffusion_skeleton.py
```

#### Training interaction correction on the BEHAVE dataset

```
python train_correction_smpl.py
```

#### Training interaction correction on the skeleton-based dataset

```
python train_correction_skeleton.py
```

#### Evaluate short-term generation on the BEHAVE dataset

You need to specify the checkpoint of interaction diffusion (PATH/TO/YOUR/DIFFUSION.ckpt) and interaction correction (PATH/TO/YOUR/CORRECTION.ckpt)

To evaluate the performance of the full InterDiff pipeline with interaction correction: 
```
python eval_smpl_short.py --resume_checkpoint PATH/TO/YOUR/DIFFUSION.ckpt --resume_checkpoint_obj PATH/TO/YOUR/CORRECTION.ckpt
```

To use our provided model:
```
python eval_smpl_short.py --resume_checkpoint checkpoints/diffusion.ckpt --resume_checkpoint_obj checkpoints/correction.ckpt
```

To evaluate the performance of the plain interaction diffusion without interaction correction: 
```
python eval_smpl_short.py --resume_checkpoint PATH/TO/YOUR/DIFFUSION.ckpt --resume_checkpoint_obj PATH/TO/YOUR/CORRECTION.ckpt --mode no_correction
```

To use our provided model:
```
python eval_smpl_short.py --resume_checkpoint checkpoints/diffusion.ckpt --resume_checkpoint_obj checkpoints/correction.ckpt --mode no_correction
```
#### Evaluate short-term generation on the skeleton-based dataset

You need to specify the checkpoint of interaction diffusion (PATH/TO/YOUR/DIFFUSION.ckpt) and interaction correction (PATH/TO/YOUR/CORRECTION.ckpt)

To evaluate the performance of the full InterDiff pipeline with interaction correction: 
```
python eval_skeleton.py --resume_checkpoint PATH/TO/YOUR/DIFFUSION.ckpt --resume_checkpoint_obj PATH/TO/YOUR/CORRECTION.ckpt
```

To evaluate the performance of the plain interaction diffusion without interaction correction: 

```
python eval_skeleton_no_correction.py --resume_checkpoint PATH/TO/YOUR/DIFFUSION.ckpt
```