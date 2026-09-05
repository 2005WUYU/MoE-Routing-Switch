#!/bin/bash
# One torchrun per Slurm task; contiguous local ranks form the node-local EP group.
exec torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node=8 \
  --node-rank="$SLURM_PROCID" \
  --master-addr="$MOE_MASTER_ADDR" \
  --master-port="$MOE_MASTER_PORT" \
  --module moe_study.train "$@"
