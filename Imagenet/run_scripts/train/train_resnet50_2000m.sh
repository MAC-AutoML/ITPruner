#!/usr/bin/env bash
code_path='/ITPruner/Imagenet/'
chmod +x ${code_path}/prep_imagenet.sh
cd ${code_path}
echo "preparing data"
bash ${code_path}/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet50" \
 --cfg "[50, 53, 54, 125, 54, 54, 54, 54, 101, 108, 179, 107, 107, 107, 107, 107, 107, 203, 216, 148, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 407, 429, 1341, 429, 429, 429, 428]" \
 --path "Exp_train/train_resnet50_2000m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --base_path "Exp_base/resnet50_base" \
 --sync_bn \
 --warm_epoch 1 \
 --n_epochs 120
