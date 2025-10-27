#!/usr/bin/bash

#SBATCH -J vgg-cutmix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail
mkdir -p logs/slurm logs/cutmix5

echo "[INFO] $(date)"
pwd; hostname; which python || true; nvidia-smi || true

# CutMix 학습기법 튜닝
SUITE="cutmix5"

echo "[RUN] suite=$SUITE"

python train_vgg.py \
  --epochs 30 --lr 1e-3 --batch_size 128 \
  --suite $SUITE \
  --data_root ../../data/ \
  --logdir logs

