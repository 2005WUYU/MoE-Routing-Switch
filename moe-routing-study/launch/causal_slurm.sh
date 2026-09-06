#!/bin/bash
#SBATCH --job-name=moe-causal-step
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=causal-%j.out

# Run from moe-routing-study/. Keep image and shared-storage paths in local env.
export MOE_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MOE_MASTER_PORT=29500
export PYTHONPATH="$PWD/src"
export TRITON_LIBCUDA_PATH=/.singularity.d/libs

exec srun --ntasks-per-node=1 --gres=gpu:8 \
  apptainer exec --nv \
  --bind "$PWD:$PWD" \
  --bind "$MOE_BIND_PATHS" \
  --pwd "$PWD" "$MOE_RUNTIME_IMAGE" \
  bash launch/causal_node.sh "$@"
