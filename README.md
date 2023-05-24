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

Then after activating environment 

```
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
```

```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
```

