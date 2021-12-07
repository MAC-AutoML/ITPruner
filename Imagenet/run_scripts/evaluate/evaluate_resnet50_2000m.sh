#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "resnet50" \
 --path "pretrain_models/train_resnet50_2000m_2229" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --cfg "[50, 53, 54, 125, 54, 54, 54, 54, 101, 108, 179, 107, 107, 107, 107, 107, 107, 203, 216, 148, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 407, 429, 1341, 429, 429, 429, 428]"