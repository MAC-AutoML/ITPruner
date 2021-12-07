# ITPruner: An Information Theory-inspired Strategy for Automatic Network Pruning

This repository contains all the experiments of our paper "An Information Theory-inspired Strategy for Automatic Network Pruning". It also includes some [pretrain_models](https://drive.google.com/drive/folders/12B741haMD8qGV6x2UMe0yjnIYOTuIokU?usp=sharing) which we list in the paper.

# Requirements

* [DALI](https://github.com/NVIDIA/DALI)
* [Apex](https://github.com/NVIDIA/apex)
* [torchprofile](https://github.com/zhijian-liu/torchprofile)
* other requirements, running requirements.txt

```python
pip install -r requirements.txt
```



# Running

<font size=4>**feature extract**</font>

You need to download [base models](https://drive.google.com/drive/folders/1jAXw4pEFmaw6fSgqNGRhhvtsM-ZUhZsr?usp=sharing) and copy the path of them to "--path".

```python
# [optional]cache imagenet dataset in RAM for accelerting I/O
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
 --target_flops 150000000 \
 --beta 243
```

or 

```python
bash ./run_scripts/feature_extract/extract_mobilenet_150m.sh
```



<font size=4>**Train**</font>

Because of random seed, cfg obtained through feature extraction may have a little difference from ours. Our cfg are given in .sh files.

```python
# [optional]cache imagenet dataset in RAM for accelerting I/O
code_path='/ITPruner/Imagenet/'
chmod +x ${code_path}/prep_imagenet.sh
cd ${code_path}
echo "preparing data"
bash ${code_path}/prep_imagenet.sh >> /dev/null
echo "preparing data finished"

python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "mobilenet" \
 --cfg "[23, 42, 63, 67, 132, 134, 275, 194, 202, 202, 194, 277, 522, 687]" \
 --path "Exp_train/train_mobilenet_150m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --base_path "Exp_base/mobilenet_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 250 \
 --label_smoothing 0.1
```

or

```python
bash ./run_scripts/train/train_mobilenet_150m.sh
```



<font size=4>**Evaluate**</font>

We provide some [pretrain_models](https://drive.google.com/drive/folders/12B741haMD8qGV6x2UMe0yjnIYOTuIokU?usp=sharing) which we list in the paper.

```python
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "pretrain_models/train_mobilenet_150m_31752" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --cfg "[23, 42, 63, 67, 132, 134, 275, 194, 202, 202, 194, 277, 522, 687]"
```

or

```python
bash ./run_scripts/evaluate/evaluate_mobilenet_150m.sh
```

