#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name automesh
#SBATCH --output=log2.out

source /scratch/htc/tshah/miniconda3/bin/activate automesh
module load cuda-11.0.2
srun python /scratch/htc/tshah/AutoMesh/scripts/main.py
