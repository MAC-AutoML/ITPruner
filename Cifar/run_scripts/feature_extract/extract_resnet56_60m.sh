#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet56" \
 --path "Exp_base/resnet56_base" \
 --dataset "cifar10" \
 --save_path 'your_data_path' \
 --target_flops 60000000 \
 --beta 500