# FLEX: Parameter-free Multi-view 3D Human Motion Reconstruction

This repository is anonymous. That is, it does not reveal the identity of the authors, neither in its contents nor in links to other sites. 
Notice that the latest date on which the content of this repository has been updated is before the sup. mat. deadline.

Clips attached to sup. mat. `zip` file with better quality (stored in an anonymous account):
- A clip describing our work: [clip.mp4](https://drive.google.com/file/d/1k2R-ff0-wn609qQVWFheLBrbTlqFhZVa/view?usp=sharing)
- Video files showing our results on the Human3.6M dataset: [Human36M_S9_Posing_1.mp4](https://drive.google.com/file/d/19eHUYwEnd52ML4xyMACQhN6tcKhq2B4J/view?usp=sharing) & [Human36M_S9_Sitting.mp4](https://drive.google.com/file/d/1F0nZA257StpxzKVNNI84Y89SwCEAmUVK/view?usp=sharing)
- Video file showing our results on the KTH multi-view Football II dataset: [KTH_football.mp4](https://drive.google.com/file/d/12o6MYtX53kZ7_pxy-ga26Bn0JwZyGJ4I/view?usp=sharing)
- Video files comparing MotioNet (single-view) \& Iskakov et.al. results versus ours: [MotioNet_comparison.mp4](https://drive.google.com/file/d/1BNmIJ_eb5LyP2WuIsG5C1wo0CDZWzoCk/view?usp=sharing) & [Iskakov_comparison.mp4](https://drive.google.com/file/d/1oyrvhq5245__lxcgKfPC8eiKdwZvORo4/view?usp=sharing)
- Syntethic videos from our Blender studio are available to download [here](https://drive.google.com/drive/folders/1yvBYLr8GgRSlsCK25bUsQVbAEqZQYMG6?usp=sharing)

## Requirements

- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

Run the following command to install other packages:
```setup
conda env create -f environment.yml -n <YOUR-ENV-NAME>
```

## Data
- Download the data .zip file from [here](https://drive.google.com/file/d/1hJoyuptbXe4-WcO7sWNUHkNO4iaZJzDh/view?usp=sharing) and unzip it inside the `FLEX/data` folder.
- Download the pre-trained model checkpoint from [here](https://drive.google.com/file/d/1rJMh6SzzsjU4pAMq9bg4ssnUgyx1bF_Q/view?usp=sharing) and add it under the `FLEX/checkpoint` folder.

## Evaluation
After you have downloaded the data & pre-trained checkpoint you can evaluate our model by running:
```
python evaluate_multiview.py --resume=./4_views_mha64_gt.pth --device=<GPU-DEVICE-ID>
```
Notes: 
- In case you are not on a GPU supported machine, just delete the `--device` flag, and the evaluation will run on CPU.
- In order to save bvh files under `FLEX/output` folder, add `--save_bvh_files` argument.


## Training

To train the model(s) in the paper, run this command:

Using GT data:
```train
python train.py --batch_size=32 --channel=1024 --n_views=4 --kernel_width=5 --padding=2 --kernel_size_stage_1=5,3,1 --kernel_size_stage_2=5,3,1 --data=gt --n_joints=20 --dilation=1,1,1 --stride=1,1,1 --kernel_size=5,3,1 --transformer_mode=mha --transformer_n_heads=64 --device=<GPU-DEVICE-ID>
```

Using Iskakov et al. 2D detected pose:
```train
python train.py --batch_size=32 --channel=1024 --n_views=4 --kernel_width=5 --padding=2 --kernel_size_stage_1=5,3,1 --kernel_size_stage_2=5,3,1 --data=learnable --n_joints=20 --dilation=1,1,1 --stride=1,1,1 --kernel_size=5,3,1 --transformer_mode=mha --transformer_n_heads=64 --device=<GPU-DEVICE-ID>
```

## Results
The evaluation script will output some results to the terminal.
Here is an example of our pre-trained model output, using ground-truth 2D input:
```
+--------------+------------+---------------------+
|    Action    | MPJPE (mm) | Acc. Error (mm/s^2) |
+--------------+------------+---------------------+
|  Directions  |   18.04    |         0.54        |
|  Discussion  |   22.03    |         0.73        |
|    Eating    |   20.52    |         0.55        |
|   Greeting   |   20.60    |         1.38        |
|   Phoning    |   22.82    |         0.94        |
|    Photo     |   31.77    |         0.68        |
|    Posing    |   19.68    |         0.70        |
|  Purchases   |   21.88    |         1.02        |
|   Sitting    |   26.98    |         0.49        |
| SittingDown  |   28.65    |         0.81        |
|   Smoking    |   24.05    |         0.93        |
|   Waiting    |   21.06    |         0.58        |
|   WalkDog    |   25.93    |         1.72        |
| WalkTogether |   19.23    |         0.87        |
|   Walking    |   18.92    |         1.09        |
|              |            |                     |
|   Average    |   22.89    |         0.87        |
+--------------+------------+---------------------+
```
