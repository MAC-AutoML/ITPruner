# https://github.com/pytorch/vision/blob/master/torchvision/models
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

from utils import cutout_batch
from models.base_models import MyNetwork
import numpy as np

cifar_cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG_CIFAR(MyNetwork):
    def __init__(self, cfg=None, cutout=True, num_classes=10):
        super(VGG_CIFAR, self).__init__()
        if cfg is None:
            cfg = [64, 64, 128, 128, 256, 256,
                   256, 512, 512, 512, 512, 512, 512]
        self.cutout = cutout
        self.cfg = cfg
        _cfg = list(cfg)
        _cfg.insert(2, 'M')
        _cfg.insert(5, 'M')
        _cfg.insert(9, 'M')
        _cfg.insert(13, 'M')
        self._cfg = _cfg
        self.feature = self.make_layers(_cfg, True)
        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Sequential(
            nn.Linear(self.cfg[-1], 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self.num_classes = num_classes
        self.classifier_param = (
            self.cfg[-1] + 1) * 512 + (512 + 1) * num_classes

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        pool_index = 0
        conv_index = 0
        for v in cfg:
            if v == 'M':
                layers += [('maxpool_%d' % pool_index,
                            nn.MaxPool2d(kernel_size=2, stride=2))]
                pool_index += 1
            else:
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=1, bias=False)
                conv_index += 1
                if batch_norm:
                    bn = nn.BatchNorm2d(v)
                    layers += [('conv_%d' % conv_index, conv2d), ('bn_%d' % conv_index, bn),
                               ('relu_%d' % conv_index, nn.ReLU(inplace=True))]
                else:
                    layers += [('conv_%d' % conv_index, conv2d),
                               ('relu_%d' % conv_index, nn.ReLU(inplace=True))]
                in_channels = v
        self.conv_num = conv_index
        return nn.Sequential(OrderedDict(layers))

    def cfg2param(self, cfg):
        total_param = self.classifier_param
        c_in = 3
        _cfg = list(cfg)
        _cfg.insert(2, 'M')
        _cfg.insert(5, 'M')
        _cfg.insert(9, 'M')
        _cfg.insert(13, 'M')
        for c_out in _cfg:
            if isinstance(c_out, int):
                total_param += 3 * 3 * c_in * c_out + 2 * c_out
                c_in = c_out
        total_param += 512 * (_cfg[-1] + 1)
        return total_param

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        _cfg = list(cfg)
        _cfg.insert(2, 'M')
        _cfg.insert(5, 'M')
        _cfg.insert(9, 'M')
        _cfg.insert(13, 'M')
        input_size = 32
        num_classes = 10
        total_flops = 0
        c_in = 3
        for c_out in _cfg:
            if c_out !='M':
                total_flops += 3 * 3 * c_in * c_out * input_size * input_size  # conv
                total_flops += 4 * c_out * input_size * input_size  # bn
                total_flops += input_size * input_size * c_out  # relu
                c_in = c_out
            else:
                input_size /= 2
                total_flops += 2 * input_size * input_size * c_in  # max pool
        input_size /= 2
        total_flops += 3 * input_size * input_size * _cfg[-1]  # avg pool
        total_flops += (2 * input_size * input_size *
                        _cfg[-1] - 1) * input_size * input_size * 512  # fc
        total_flops += 4 * _cfg[-1] * input_size * input_size  # bn_1d
        total_flops += input_size * input_size * 512  # relu
        total_flops += (2 * input_size * input_size * 512 - 1) * \
            input_size * input_size * num_classes  # fc
        return total_flops

    def cfg2flops_perlayer(self, cfg, length):  # to simplify, only count convolution flops
        _cfg = list(cfg)
        _cfg.insert(2, 'M')
        _cfg.insert(5, 'M')
        _cfg.insert(9, 'M')
        _cfg.insert(13, 'M')
        input_size = 32
        num_classes = 10
        flops_singlecfg = [0 for j in range(length)]
        flops_doublecfg = np.zeros((length, length))
        flops_squarecfg = [0 for j in range(length)]
        c_in = 3
        count = 0
        for c_out in _cfg:
            if c_out !='M' and count ==0:
                flops_singlecfg[count] += 3 * 3 * c_in * c_out * input_size * input_size  # conv
                flops_singlecfg[count] += 4 * c_out * input_size * input_size  # bn
                flops_singlecfg[count] += input_size * input_size * c_out  # relu
                c_in = c_out
                count += 1
            elif c_out !='M' and count !=0:
                flops_doublecfg[count-1][count] += 3 * 3 * c_in * c_out * input_size * input_size  # conv
                flops_doublecfg[count][count-1] += 3 * 3 * c_in * c_out * input_size * input_size
                flops_singlecfg[count] += 4 * c_out * input_size * input_size  # bn
                flops_singlecfg[count] += input_size * input_size * c_out  # relu
                c_in = c_out
                count += 1
            else:
                input_size /= 2
                flops_singlecfg[count-1] += 2 * input_size * input_size * c_in  # max pool
        input_size /= 2
        flops_singlecfg[count-1] += 3 * input_size * input_size * _cfg[-1]  # avg pool
        flops_singlecfg[count-1] += (2 * input_size * input_size * _cfg[-1] ) * input_size * input_size * 512  # fc
        flops_singlecfg[count-1] += 4 * _cfg[-1] * input_size * input_size  # bn_1d

        return flops_singlecfg, flops_doublecfg, flops_squarecfg

    def forward(self, x):
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def feature_extract(self, x):
        tensor = []
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        for _layer in self.feature:
            x = _layer(x)
            if type(_layer) is nn.ReLU:
                tensor.append(x)
        return tensor

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
            'dataset': 'cifar10',
        }


