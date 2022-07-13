#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --job-name automesh
#SBATCH --output=log.out

source /scratch/htc/tshah/miniconda3/bin/activate automesh
srun python /scratch/htc/tshah/AutoMesh/scripts/main.py