#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet20" \
 --path "Exp_base/resnet20_base" \
 --dataset "cifar10" \
 --save_path 'your_data_path' \
 --target_flops 20000000 \
 --beta 246