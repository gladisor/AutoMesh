#!/bin/bash -l
#SBATCH -p gpu

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

#SBATCH --time=05:00:00
#SBATCH --job-name automesh
#SBATCH --output=log.out
#SBATCH --mail-user=tristan.shah@sjsu.edu

export NCCL_DEBUG=INFO
# export MASTER_PORT=46356
# export WORLD_SIZE=4

# echo "NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

source /scratch/htc/tshah/miniconda3/bin/activate automesh2
srun python /scratch/htc/tshah/AutoMesh/scripts/main.py
