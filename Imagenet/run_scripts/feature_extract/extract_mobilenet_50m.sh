#!/usr/bin/env bash
code_path='/ITPruner/Imagenet/'
chmod +x ${code_path}/prep_imagenet.sh
cd ${code_path}
echo "preparing data"
bash ${code_path}/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "mobilenet" \
 --path "Exp_base/mobilenet_base" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --target_flops 50000000 \
 --beta 243