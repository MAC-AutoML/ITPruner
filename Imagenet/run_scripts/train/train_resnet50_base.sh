#!/usr/bin/env bash
code_path='/ITPruner/Imagenet/'
chmod +x ${code_path}/prep_imagenet.sh
cd ${code_path}
echo "preparing data"
bash ${code_path}/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet50" \
 --path "Exp_base/resnet50_base_${RANDOM}" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --sync_bn \
 --warm_epoch 1 \
 --n_epochs 120
