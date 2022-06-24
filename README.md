# AutoMesh

Setup:

```
conda create -n automesh python=3.7
conda activate automesh
```

```
conda install -c pytorch pytorch=1.6.0 torchvision
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
```

```
pip install pytorch3d==0.2.0
```