# AutoMesh

## Setup:

To setup the conda environment
```
conda env create -n automesh python=3.8
conda activate automesh
conda install pytorch torchvision torchaudio -c pytorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

pip install class_resolver
pip install pytorch
pip install torchmetrics
pip install open3d
pip install optuna
```