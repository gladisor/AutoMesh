#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name am_edge
#SBATCH --output=log.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

source /scratch/htc/tshah/miniconda3/bin/activate automesh2
srun python /scratch/htc/tshah/AutoMesh/scripts/main.py