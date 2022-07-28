#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name am_edge
#SBATCH --output=log.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

export CUDA_LAUNCH_BLOCKING=1

source /scratch/htc/tshah/miniconda3/bin/activate automesh2
srun python /scratch/htc/tshah/AutoMesh/scripts/main.py


### salloc -p gpu --time 12:00:00 --nodes 1 --gpus-per-node 4 --gres gpu:4 --cpus-per-task 4 --job-name automesh2
### comment