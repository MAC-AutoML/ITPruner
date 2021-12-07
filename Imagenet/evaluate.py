import os
import json
import argparse
import numpy as np

import torch
from run_manager import RunManager
from models import ResNet_ImageNet, MobileNet, MobileNetV2, TrainRunConfig

parser = argparse.ArgumentParser()

""" model config """
parser.add_argument('--path', type=str)
parser.add_argument('--model', type=str, default="vgg",
                    choices=['vgg', 'resnet56', 'resnet110', 'resnet18', 'resnet34', 'resnet50', 'mobilenetv2', 'mobilenet'])
parser.add_argument('--cfg', type=str, default="None")

""" dataset config """
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'imagenet'])
parser.add_argument('--save_path', type=str, default='/userhome/data/cifar10')

""" runtime config """
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--n_worker', type=int, default=24)
parser.add_argument("--local_rank", default=0, type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    # distributed setting
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # prepare run config
    run_config_path = '%s/run.config' % args.path

    run_config = TrainRunConfig(
        **args.__dict__
    )
    if args.local_rank == 0:
        print('Run config:')
        for k, v in args.__dict__.items():
            print('\t%s: %s' % (k, v))

    if args.model == "resnet50":
        assert args.dataset == 'imagenet', 'resnet50 only supports imagenet dataset'
        net = ResNet_ImageNet(
            depth=50, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == "mobilenetv2":
        assert args.dataset == 'imagenet', 'mobilenetv2 only supports imagenet dataset'
        net = MobileNetV2(
            num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == "mobilenet":
        assert args.dataset == 'imagenet', 'mobilenet only supports imagenet dataset'
        net = MobileNet(
            num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))

    # build run manager
    run_manager = RunManager(args.path, net, run_config)

    # load checkpoints
    best_model_path = '%s/checkpoint/model_best.pth.tar' % args.path
    assert os.path.isfile(best_model_path), 'wrong path'
    if torch.cuda.is_available():
        checkpoint = torch.load(best_model_path)
    else:
        checkpoint = torch.load(best_model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    run_manager.net.load_state_dict(checkpoint)

    output_dict = {}

    # test
    print('Test on test set')
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5
    }
    if args.local_rank == 0:
        print(output_dict)
