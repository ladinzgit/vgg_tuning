#!/usr/bin/bash

#SBATCH -J vgg-struct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out
#SBATCH -e logs/slurm-%A.err

set -euo pipefail
mkdir -p logs/slurm logs/struct

echo "[INFO] $(date)"
pwd; hostname; which python || true; nvidia-smi || true

# 구조적 튜닝만 실행
SUITE="struct6"

echo "[RUN] suite=$SUITE"

python train_vgg.py \
  --epochs 30 --lr 1e-3 --batch_size 128 \
  --suite $SUITE \
  --data_root ../../data/ \
  --logdir logs

