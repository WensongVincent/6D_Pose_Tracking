# Self-Supervised Category-Level 6D Object Pose Tracking using Learned Mesh

## Progress

- [ ] Training code
- [ ] Evaluation code
- [ ] Pretrained models
- [ ] More datasets

## Environment Setup

1. Check Nvidia Driver
   ```sh
   nvidia-smi
   ```
   If it is in nouveau status not nvidia, then download

   Update root path:
   ```sh
   sudo apt-get update
   ```
   
2. Download Nvidia Driver
   Check available drivers:
   ```sh
   apt-cache search nvidia-driver-
   ```
   
   Choose the most suitable driver, in this case, 470 (nvidia-driver-xxx, for desktop user; nvidia-driver-xxx-server, for server user; nvidia-driver-xxx-open, for open source user):
   ```sh
   sudo apt install nvidia-driver-470
   ```

   Check GPU in use by checking the string in square bracket:
   ```sh
   lspci -Dnn | grep ‘NVIDIA’
   ```

   Check kernel in use:
   ```sh
   lspci -nnk -d <string_above>
   ```

   Restart Ubuntu
   ```sh
   sudo reboot
   ```
   
3. Download anaconda (use $ conda -V check whether have one):
   ```sh
   cd <file_want_to_download>
   ```

   Get .sh file from website:
   ```sh
   wget <website_of_anaconda_file>
   ```

   Install:
   ```sh
   bash <file_name_of_anaconda_download>
   ```

   Init:
   ```sh
   conda init
   ```
   
4. Download CUDA:
   Find correct version in website: https://developer.nvidia.com/cuda-toolkit-archive, this case: CUDA 11.1.0

   Use runfile(local) method, Download .run file first:
   ```sh
   wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
   ```

   Install CUDA, don’t choose [ ]Driver, since already got one:
   ```sh
   sudo sh cuda_11.1.0_455.23.05_linux.run
   ```
   
5. Set environment variable:
   Open ~/.bashrc file:
   ```sh
   sudo nano ~/.bashrc
   ```
   Copy these to the end of file:
   ```sh
   export CUDA_HOME=/usr/local/cuda
   export PATH=$PATH:$CUDA_HOME/bin
   export LD_LIBRARY_PATH=/usr/local/cuda11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

   Save:
   ```sh
   source ~/.bashrc
   ```
   These steps not sure: https://blog.csdn.net/chen20170325/article/details/130294270

6. Setup Pytorch and others:
PyTorch with CUDA support are required. Our code is tested on python 3.8, torch 1.10.0, CUDA 11.1, and RTX 3090.

We recommend installing the required packages in the following order to avoid potential version conflicts:
```sh
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install absl-py tensorboard opencv-python setuptools==59.5.0 trimesh kornia fvcore iopath matplotlib wandb scikit-learn scipy
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu111_pyt1100/download.html
```
If the third step fails to find a valid version of pytorch3d, an alternative approach is to download the .tar.bz2 package from [the Anaconda website](https://anaconda.org/pytorch3d/pytorch3d/files?sort=length&sort_order=desc&page=1) (version: pytorch3d-0.6.1-py38_cu111_pyt1100.tar.bz2), and install the package with:
```sh
conda install /path/to/xxx.tar.bz2
```
Then, git clone our repository, and build the required [SoftRas](https://github.com/ShichenLiu/SoftRas) package located in ```PROJECT_ROOT/third-party/softras```:
```sh
cd third-party/softras
python setup.py install
```

## Data Preparation

To reproduce our experiments, please download the raw [Wild6D](https://github.com/OasisYang/Wild6D) dataset, unzip, and add dataset paths to training and testing configurations, for example, for the laptop training set, the correct path assignment is 
```sh
--dataset_path /path/to/wild6d/laptop/ --test_dataset_path /path/to/wild6d/test_set/laptop/
```

## Training
Our model uses a pretrained [DINO](https://github.com/facebookresearch/dino) for correspondence learning. Download the pretrained models from [here](https://drive.google.com/drive/folders/1MOeWKoHoBK9GH6jJ-BZPvD9rj9xQdWux?usp=share_link) and put them in the `PROJECT_ROOT/pretrain/` directory. Also create the ```PROJECT_ROOT/log/``` directory for logging. You can also assign custom paths via configuration.

Run training with ```train.py```. We have provided an example training script for the laptop category, run by the following command:
```sh
bash scripts/train.sh
```

## Testing
Run testing with ```predict.py```. We have provided an example testing script, run by the following, run by the following command:
```sh
bash scripts/predict.sh
```
The testing script also offers visualization options. For example, use ```--vis_pred --vis_bbox``` for bounding box visualization.

## Pretrained Models

We provide the pretrained models on Wild6D dataset containing all 5 categories. 

To use, download the checkpoints in this [link](https://drive.google.com/drive/folders/1m9JwibSun0GTHRcfHoVLBLmPc3DWqy0Q?usp=share_link). Indicate the checkpoint path with the ```--model_path``` argument in ```scripts/predict.sh```. 



