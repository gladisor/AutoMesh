#!/bin/bash -l
#SBATCH -p gpu

#SBATCH --gres=gpu:4
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0gb

#SBATCH --job-name automesh_multinode
#SBATCH --output=log_multinode.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

source /scratch/htc/tshah/miniconda3/bin/activate automesh2
srun python /scratch/htc/tshah/AutoMesh/scripts/test.py
