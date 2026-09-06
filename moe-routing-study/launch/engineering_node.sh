#!/bin/bash
exec python3 -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node=8 \
  --node-rank="$SLURM_PROCID" \
  --master-addr="$MOE_MASTER_ADDR" \
  --master-port="$MOE_MASTER_PORT" \
  --module moe_study.engineering "$@"
