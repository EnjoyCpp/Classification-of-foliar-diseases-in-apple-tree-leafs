# Classification of foliar diseases in apple tree leafs

## Hardware requirements

With any CPU training of a model can be done but using GPU 
is highly recommended since it can cut time needed to train model
signifantly.


## Environment requirements

Training of model to classify foliar diseases in trees is done using 
TensorFlow 

## Training of model is done on Linux operating system

Firstly install Miniconda to create seperate environment to train model using GPU


```
conda create --name training_model python=3.9
conda activate training_model
```

Then after activating environment if you want to GPU instead of CPU.
IMPORTANT!!!: This if you want to use CPU instead skip to Install TensorFlow

```
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
```

```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
```

```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### Install TensorFlow


```
pip install --upgrade pip
```

```
pip install tensorflow==2.11.*
```

### Install efficientNet and other modules required for training of a model

```
pip install numpy
pip install pandas
pip install -q efficientnet
pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.10.*
pip install matplotlib
pip install albumentations
```
