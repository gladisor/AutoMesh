#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --job-name automesh
#SBATCH --output=log1.out

source /scratch/htc/pzimmer/miniconda3/bin/activate automesh2

srun python /scratch/htc/pzimmer/AutoMesh/scripts/main.py

### salloc -p gpu --time 06:00:00 --nodes 1 --gpus-per-node 1 --cpus-per-task 4 --job-name automesh