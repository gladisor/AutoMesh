# AutoMesh

## Setup:

WARNING: These setup instructions have not been tested for gpu

```
conda create -n automesh python=3.8
conda activate automesh
pip install -r requirements.txt
```

### Install pytorch geometric

replace cpu with cuda version if needed

```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
```

## Usage

Our dataset class expects the mesh data to be in the form of .ply files. Data must be contained in a subfolder titled "raw".

```
data = LeftAtriumData('path/to/data')
```