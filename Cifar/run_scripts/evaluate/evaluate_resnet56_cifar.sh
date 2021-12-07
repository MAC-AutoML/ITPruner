## ============================= resnet cifar10 decode  ============================
python main.py \
  --gpu_id 0 \
  --exec_mode eval \
  --learner vanilla \
  --dataset cifar10 \
  --data_path ~/datasets \
  --model_type resnet_decode \
  --load_path /ITPruner/Cifar/models/best_models/resnet56_60m/model.pt \
  --cfg 16,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,22,22,21,21,21,21,21,21,21,21,21,21,21,21,21,21,22,22,44,44,42,42,42,42,42,42,42,42,42,42,42,42,64,64,64,64
