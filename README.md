pytorch-chat-bot
==============================
## Demo

Depending on input, the output is displayed as the result like the image below.
![result](https://github.com/SnowMasaya/pytorch-chat-bot/blob/master/img/pytorch-chat.mov.gif)

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# Environiment

export PYTHONPATH=$PYTHONPATH:`pwd`

# Install

## Initial Setting

```
touch /etc/apt/sources.list.d/nvidia-ml.list
sudo echo 'deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/ /' > /etc/apt/sources.list.d/nvidia-ml.list
sudo apt-get update && sudo apt-get install -y --no-install-recommends build-essential python3-dev cmake git curl vim ca-certificates libnccl2=2.0.5-2+cuda8.0 libnccl-dev=2.0.5-2+cuda8.0 libjpeg-dev libpng12-dev && rm -rf /var/lib/apt/lists/*
sudo apt-get update && sudo apt-get install -y libxtst6 g++
sudo -E add-apt-repository -y ppa:george-edison55/cmake-3.x
sudo -E apt-get update
sudo apt-get install cmake
sudo apt-get update
sudo apt-get upgrade
```

## Prepare Mecab

```
sudo apt-get install libmecab-dev make
sudo apt-get install mecab mecab-ipadic-utf8
```

## Prepare Mecab-neologd dict

```
$ git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git

$ cd mecab-ipadic-neologd

$ ./bin/install-mecab-ipadic-neologd -n

$ echo `mecab-config --dicdir`"/mecab-ipadic-neologd"

$ ./bin/install-mecab-ipadic-neologd -h
```


## Conda install

```
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
sh ~/miniconda.sh -b -p conda/ && rm ~/miniconda.sh
conda/bin/conda install numpy pyyaml mkl setuptools cmake cffi matplotlib
```

## GPU environiment

```
conda/bin/conda install -c soumith magma-cuda80
```

## PyTorch Install

```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export PYTHON_VERSION=3.6
conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl pyymal
conda/bin/conda clean -ya
source ../conda/bin/activate pytorch-py3.6
python setup.py insatll
export PATH=/opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH
conda install --name pytorch-py$PYTHON_VERSION -c soumith magma-cuda80
export PATH=`pwd`/pytorch-reinforcement-learning/conda/envs/pytorch-py3.6/bin/:$PATH
conda install --name pytorch-py$PYTHON_VERSION -c soumith magma-cuda80
TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
pip install -v .
git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .
conda install pytorch torchvision cuda80 -c soumith
pip install git+https://github.com/pytorch/pytorch
pip install torchvision
```

## Active Conda Environiment

```
source conda/bin/activate pytorch-py3.6
```
