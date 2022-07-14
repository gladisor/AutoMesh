#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name automesh
#SBATCH --output=log2.out

export WANDB_API_KEY = "9a6992594ce0851dbccb151860b2751420a558a3"
export WANDB_ENTITY= "tshah"
source /scratch/htc/tshah/miniconda3/bin/activate automesh2
srun python /scratch/htc/tshah/AutoMesh/scripts/main.py
