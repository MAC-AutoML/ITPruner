#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 train.py --train \
 --model "vgg" \
 --cfg "[44, 35, 71, 71, 145, 104, 145, 290, 205, 325, 432, 429, 469]" \
 --path "Exp_train/train_vgg_100m_${RANDOM}" \
 --dataset "cifar10" \
 --save_path 'your_data_path' \
 --base_path "Exp_base/vgg_base" \
 --warm_epoch 1 \
 --n_epochs 300 \
 --sync_bn

