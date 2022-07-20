#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name automesh
#SBATCH --output=log.out

source /scratch/htc/pzimmer/miniconda3/bin/activate automesh2

srun python /scratch/htc/pzimmer/AutoMesh/scripts/main.py

