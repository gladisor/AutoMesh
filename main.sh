#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --job-name automesh_db
#SBATCH --output=optunadb_overnight.out
#SBATCH --mail-user=paul.zimmer@zib.de

export NCCL_DEBUG=INFO

source /scratch/htc/pzimmer/miniconda3/bin/activate automesh2
srun python /scratch/htc/pzimmer/AutoMesh/scripts/main.py


### salloc -p gpu --time 12:00:00 --nodes 1 --gpus-per-node 4 --gres gpu:4 --cpus-per-task 4 --job-name automesh2
### comment