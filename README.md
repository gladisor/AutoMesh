# AutoMesh

## Setup:

WARNING: These setup instructions have not been tested for gpu

To install
```
conda env create -f environment.yml
```

To update
```
conda env update -f environment.yml --prune
```

## Usage

Our dataset class expects the mesh data to be in the form of .ply files. Data must be contained in a subfolder titled "raw".

```
data = LeftAtriumData('path/to/data')
```