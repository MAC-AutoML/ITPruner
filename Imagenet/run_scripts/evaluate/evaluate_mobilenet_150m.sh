#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "pretrain_models/train_mobilenet_150m_31752" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --cfg "[23, 42, 63, 67, 132, 134, 275, 194, 202, 202, 194, 277, 522, 687]"