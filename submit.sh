#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --job-name automesh
#SBATCH --output=log.out

source /scratch/htc/pzimmer/miniconda3/bin/activate automesh2

srun /scratch/htc/pzimmer/AutoMesh/scripts/main.py


salloc -p gpu --time=01:00:00 --nodes=3 --gpus-per-node=4 --cpus-per-task=4 --job-name automesh --pty bash -i