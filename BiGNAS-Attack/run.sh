#!/usr/bin/env bash
set -e

SGL_DIR="/mnt/sda1/yuxin/project/SGL-Torch/dataset/amazon-cd/pretrain-embeddings/SGL/n_layers=3/"

python search.py \
  --T-max=10 \
  --conv-lr=0.01 \
  --descent-step=30 \
  --dropout=0.8 \
  --hpo-lr=0.001 \
  --lr=0.001 \
  --meta-hidden-dim=16 \
  --meta-interval=20 \
  --meta-op=sage \
  --num-layers=2 \
  --weight-decay=0.001 \
  --top_k=60 \
  --device cuda:0 \
  --categories CD Kitchen \
  --target Kitchen \
  --use-source \
  --use-meta \
  --use-sgl-init True \
  --sgl-dir "$SGL_DIR" \
  --freeze-src-steps 1000 \
  --use-align True \
  --align-weight 1e-2 \
  --detach-source True \
  --source-item-top-ratio 0.1
