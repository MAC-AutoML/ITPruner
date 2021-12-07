import os
import argparse
import numpy as np
import math
from scipy import optimize
import random, sys
sys.path.append("../../ITPruner")
from CKA import cka
import torch
import torch.nn as nn
from run_manager import RunManager
from models import ResNet_ImageNet, MobileNet, MobileNetV2, TrainRunConfig

parser = argparse.ArgumentParser()

""" model config """
parser.add_argument('--path', type=str)
parser.add_argument('--model', type=str, default="vgg",
                    choices=['resnet50', 'mobilenetv2', 'mobilenet'])
parser.add_argument('--cfg', type=str, default="None")
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument("--target_flops", default=0, type=int)
parser.add_argument("--beta", default=1, type=int)

""" dataset config """
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'imagenet'])
parser.add_argument('--save_path', type=str, default='/userhome/data/cifar10')

""" runtime config """
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--n_worker', type=int, default=24)
parser.add_argument("--local_rank", default=0, type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
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

    # feature extract
    data_loader = run_manager.run_config.test_loader
    data = next(iter(data_loader))
    data = data[0]
    n = data.size()[0]

    with torch.no_grad():
        feature = net.feature_extract(data)

    for i in range(len(feature)):
        feature[i] = feature[i].view(n, -1)
        feature[i] = feature[i].data.cpu().numpy()

    similar_matrix = np.zeros((len(feature), len(feature)))

    for i in range(len(feature)):
        for j in range(len(feature)):
            with torch.no_grad():
                similar_matrix[i][j] = cka.cka(cka.gram_linear(feature[i]), cka.gram_linear(feature[j]))

    def sum_list(a, j):
        b = 0
        for i in range(len(a)):
            if i != j:
                b += a[i]
        return b

    important = []
    temp = []
    flops = []

    for i in range(len(feature)):
        temp.append( sum_list(similar_matrix[i], i) )

    b = sum_list(temp, -1)
    temp = [x/b for x in temp]

    for i in range(len(feature)):
        important.append( math.exp(-1* args.beta *temp[i] ) )

    length = len(net.cfg)
    flops_singlecfg, flops_doublecfg, flops_squarecfg = net.cfg2flops_perlayer(net.cfg, length)
    important = np.array(important)
    important = np.negative(important)

    # Objective function
    def func(x, sign=1.0):
        """ Objective function """
        global important,length
        sum_fuc =[]
        for i in range(length):
            sum_fuc.append(x[i]*important[i])
        return sum(sum_fuc)

    # Derivative function of objective function
    def func_deriv(x, sign=1.0):
        """ Derivative of objective function """
        global important
        diff = []
        for i in range(len(important)):
            diff.append(sign * (important[i]))
        return np.array(diff)

    # Constraint function
    def constrain_func(x):
        """ constrain function """
        global flops_singlecfg, flops_doublecfg, flops_squarecfg, length
        a = []
        for i in range(length):
            a.append(x[i] * flops_singlecfg[i])
            a.append(flops_squarecfg[i] * x[i] * x[i])
        for i in range(1,length):
            for j in range(i):
                a.append(x[i] * x[j] * flops_doublecfg[i][j])
        return np.array([args.target_flops - sum(a)])


    bnds = []
    for i in range(length):
        bnds.append((0,1))

    bnds = tuple(bnds)
    cons = ({'type': 'ineq',
             'fun': constrain_func})

    result = optimize.minimize(func,x0=[1 for i in range(length)], jac=func_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    prun_cfg = np.around(np.array(net.cfg)*result.x)

    optimize_cfg = []
    for i in range(len(prun_cfg)):
        b = list(prun_cfg)[i].tolist()
        optimize_cfg.append(int(b))
    print(optimize_cfg)
    print(net.cfg2flops(prun_cfg))
    print(net.cfg2flops(net.cfg))


