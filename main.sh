#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name am_eval
#SBATCH --output=log.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

source /scratch/htc/tshah/miniconda3/bin/activate automesh2
srun python /scratch/htc/tshah/AutoMesh/scripts/run.py
