# Real-time Super-Resolution for Video Quality Improvement

## Introduction
In this project, we find techniques to imrpove video quality in real-time for applications such as video-chat.

The program is written in Python, and uses the Pytorch library for generating and loading the mode CNN models.

## Setup Instructions
All the programs were tested on the following setup:
### Training Machine details:
* CPU: 4x Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz with 8 GB
* GPU: GeForce GTX 1050 Ti with 4GB VRAM
* Operating System: Ubuntu 16.04.3 LTS 
* Kernel (_this should not matter_): Linux 4.10.0-40-generic (x86_64)
* CUDA version: release 8.0, V8.0.61
* CuDNN: 6.0.21
* OpenCV: 3.3.0

**Note**: While a good GPU will help immensely with training the networks, it is not absolutely required to evaluate these programs.

### Software Installation:

#### Install the recommended packages
```
$ sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python2.7-dev python3.5-dev
```

#### Download OpenCV source:
```
$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/3.3.0.zip
$ unzip opencv.zip
```
```
$ cd ~
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.3.0.zip
$ unzip opencv_contrib.zip
```
#### Setup the Python virtualenv:
1. Install the latest version of pip:
```
$ sudo apt-get install python-pip && pip install --upgrade pip
```
2. Install virtualenv and virtualenvwrapper:
```
$ sudo pip install virtualenv virtualenvwrapper
$ sudo rm -rf ~/.cache/pip
```
3. Add the following lines to your ~/.bashrc file:

```
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```
4. Now source your ~/.bashrc:
```
source ~/.bashrc
```
5. Next create a virtual environment:

```
mkvirtualenv supres -p python3
```
6. To get back into the virtualenv, just type:
```
workon supres
```
7. Now, install numpy:
```
pip install numpy
```
#### Building OpenCV:
1. Perform the cmake:

```
$ cd ~/opencv-3.3.0/
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
      -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
      -D BUILD_EXAMPLES=ON ..
```
2. Build OpenCV!
```
$ make -j4
```
3. Now install OpenCV:
```
$ sudo make install
$ sudo ldconfig
```
4. Some OpenCV bookkeeping:
```
$ cd /usr/local/lib/python3.5/site-packages/
$ ls -l
```
Here, you should see a file like ```cv2.cpython-35m-x86_64-linux-gnu.so```
Go ahead and bravely rename it to cv2.so:
```
sudo mv cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
```
5. Create a symlink in your virtual env:
```
$ cd ~/.virtualenvs/supres/lib/python3.5/site-packages/
$ ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so
```
#### Install the other dependencies:
1. Install pytorch
```
pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
pip3 install torchvision
```
2. Install the other requirements
```
pip install -r requirements.txt
```

## Usage
### Running the live webcam demo
The code supports two scales of super-resolution: 4x and 8x.
Packaged with the code are two pretrained models:
1. `saved_models/coco4x_epoch_20.model` for 4x Super-Resolution
2. `saved_models/coco8x_epoch_20.model` for 8x Super-Resolution

In order to run the live demo for 4x Super-Resolution, run the following command:
```
python super-resolution.py eval --model saved_models/coco4x_epoch_20.model --downsample-scale 4
```

In order to run the live demo for 8x Super-Resolution, run the following command:
```
python super-resolution.py eval --model saved_models/coco8x_epoch_20.model --downsample-scale 8
```
These are all the options available in `eval` mode:
```
$ python super-resolution.py eval --help
usage: super-resolution.py eval [-h] --model MODEL --downsample-scale
                                DOWNSAMPLE_SCALE

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         saved model to be used for super resolution
  --downsample-scale DOWNSAMPLE_SCALE
                        amount that you wish to downsample by. Default = 8
```
### Training your own models
It is possible to train your own models with the super-resolution.py's `train` mode.

The packaged models were trained with [MS COCO dataset](http://cocodataset.org/#download). In order to download the dataset and set aside 10k images for training, please run the `download_dataset.sh` script.

Once downloaded, you can run the following example command to train a model for 4x Super-Resolution:
```
python super-resolution.py train --epochs 20 --dataset data/ --save-model-dir saved_models/ --checkpoint-model-dir checkpoints/ --downsample-scale 4
```

The full set of help options for `train` mode are as follows:
```
$ python super-resolution.py train --help
usage: super-resolution.py train [-h] [--epochs EPOCHS]
                                 [--batch-size BATCH_SIZE] [--dataset DATASET]
                                 --save-model-dir SAVE_MODEL_DIR
                                 [--checkpoint-model-dir CHECKPOINT_MODEL_DIR]
                                 [--lr LR] [--log-interval LOG_INTERVAL]
                                 [--checkpoint-interval CHECKPOINT_INTERVAL]
                                 [--downsample-scale DOWNSAMPLE_SCALE]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs, default is 2
  --batch-size BATCH_SIZE
                        batch size for training, default is 4
  --dataset DATASET     path to training dataset, the path should point to a
                        folder containing another folder with all the training
                        images
  --save-model-dir SAVE_MODEL_DIR
                        path to folder where trained model will be saved.
  --checkpoint-model-dir CHECKPOINT_MODEL_DIR
                        path to folder where checkpoints of trained models
                        will be saved
  --lr LR               learning rate, default is 1e-3
  --log-interval LOG_INTERVAL
                        number of images after which the training loss is
                        logged, default is 500
  --checkpoint-interval CHECKPOINT_INTERVAL
                        number of batches after which a checkpoint of the
                        trained model will be created
  --downsample-scale DOWNSAMPLE_SCALE
                        amount that you wish to downsample by. Default = 8
```
