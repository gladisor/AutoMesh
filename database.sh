#!/bin/bash -l
#SBATCH -p gpu

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

#SBATCH --time=12:00:00
#SBATCH --job-name automesh_db
#SBATCH --output=optunadb.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

export NCCL_DEBUG=INFO

source /scratch/htc/tshah/miniconda3/bin/activate automesh2
srun python /scratch/htc/tshah/AutoMesh/scripts/main2.py
