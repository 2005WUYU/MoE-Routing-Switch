#!/bin/bash
#SBATCH --job-name=moe-real-update
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out

# H20 allocation from machines/h20_64.yaml. Run from the project code directory.
experiment=$1
machine=$2
schedule=$3
segment=$4
project_root=/path/to/moe-routing-switch
model_source=/path/to/Qwen3-30B-A3B
runtime_image=$project_root/environment/nemo-25.11.01.sif
export MOE_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MOE_MASTER_PORT=29500
export PYTHONPATH="$PWD/src"

exec srun --ntasks-per-node=1 --gres=gpu:8 \
  apptainer exec --nv --containall \
  --bind "$project_root:$project_root" \
  --bind "$model_source:$model_source:ro" \
  --pwd "$PWD" "$runtime_image" \
  bash launch/node.sh "$experiment" "$machine" "$schedule" "$segment"
