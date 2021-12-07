#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "pretrain_models/train_mobilenet_150m_31752" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --cfg "[18, 31, 30, 38, 71, 76, 173, 52, 78, 78, 51, 178, 250, 522]"