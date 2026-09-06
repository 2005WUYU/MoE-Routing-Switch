#!/bin/bash
#SBATCH --job-name=moe-engineering
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=engineering-%j.out

# Set image, shared storage and partition in the local submission environment.
export MOE_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MOE_MASTER_PORT=29500
export PYTHONPATH="$PWD/src"
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
exec srun --ntasks-per-node=1 --gres=gpu:8 \
  apptainer exec --nv --bind "$PWD:$PWD" --bind "$MOE_BIND_PATHS" \
  --pwd "$PWD" "$MOE_RUNTIME_IMAGE" bash launch/engineering_node.sh "$@"
