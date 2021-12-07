import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.base_models import *
from collections import OrderedDict
import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes_1, planes_2=0, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv1 = conv3x3(inplanes, planes_1, stride)
        bn1 = norm_layer(planes_1)
        relu = nn.ReLU(inplace=True)
        if planes_2 == 0:
            conv2 = conv3x3(planes_1, inplanes)
            bn2 = norm_layer(inplanes)
        else:
            conv2 = conv3x3(planes_1, planes_2)
            bn2 = norm_layer(planes_2)
        self.relu = relu
        self.conv1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1), ('relu', relu)]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes_1, planes_2, planes_3=0, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv1 = conv1x1(inplanes, planes_1)
        bn1 = norm_layer(planes_1)
        conv2 = conv3x3(planes_1, planes_2, stride)
        bn2 = norm_layer(planes_2)
        if planes_3 == 0:
            conv3 = conv1x1(planes_2, inplanes)
            bn3 = norm_layer(inplanes)
        else:
            conv3 = conv1x1(planes_2, planes_3)
            bn3 = norm_layer(planes_3)
        relu = nn.ReLU(inplace=True)
        self.relu = relu
        self.conv1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1), ('relu', relu)]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2), ('relu', relu)]))
        self.conv3 = nn.Sequential(OrderedDict([('conv', conv3), ('bn', bn3)]))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet_ImageNet(MyNetwork):
    def __init__(self, cfg=None, depth=18, block=BasicBlock, num_classes=1000):
        super(ResNet_ImageNet, self).__init__()
        self.cfgs_base = {18: [64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512],
                          34: [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512],
                          50: [64, 64, 64, 256, 64, 64, 64, 64, 128, 128, 512, 128, 128, 128, 128, 128, 128, 256, 256, 1024, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 2048, 512, 512, 512, 512]}
        if depth==18:
            block = BasicBlock
            blocks = [2, 2, 2, 2]
            _cfg = self.cfgs_base[18]
        elif depth==34:
            block = BasicBlock
            blocks = [3, 4, 6, 3]
            _cfg = self.cfgs_base[34]
        elif depth==50:
            block = Bottleneck
            blocks = [3, 4, 6, 3]
            _cfg = self.cfgs_base[50]
        if cfg == None:
            cfg = _cfg
        norm_layer = nn.BatchNorm2d
        self.num_classes = num_classes
        self._norm_layer = norm_layer
        self.depth = depth
        self.cfg = cfg
        self.inplanes = cfg[0]
        self.blocks = blocks
        self.conv1 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                                                ('bn', norm_layer(self.inplanes)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if depth!=50:
            self.layer1 = self._make_layer(block, cfg[1 : blocks[0]+2], blocks[0])
            self.layer2 = self._make_layer(block, cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], blocks[1], stride=2,)
            self.layer3 = self._make_layer(block, cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], blocks[2], stride=2,)
            self.layer4 = self._make_layer(block, cfg[blocks[0]+blocks[1]+blocks[2]+4: ], blocks[3], stride=2,)
            self.fc = nn.Linear(cfg[blocks[0]+blocks[1]+blocks[2]+5], num_classes)
        else:
            self.layer1 = self._make_layer(block, cfg[1 : 2*blocks[0]+2], blocks[0])
            self.layer2 = self._make_layer(block, cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1], blocks[1], stride=2,)
            self.layer3 = self._make_layer(block, cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4], blocks[2], stride=2,)
            self.layer4 = self._make_layer(block, cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ], blocks[3], stride=2,)
            self.fc = nn.Linear(cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+6], num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if self.depth == 50:
            first_planes = planes[0:3]
            # downsample at each 1'st layer, for pruning
            downsample = nn.Sequential(OrderedDict([('conv', conv1x1(self.inplanes, first_planes[-1], stride)),
                                                    ('bn', norm_layer(first_planes[-1]))]))
            layers = []
            layers.append(block(self.inplanes, first_planes[0], first_planes[1], first_planes[2], stride, downsample, norm_layer))
            self.inplanes = first_planes[-1]
            later_planes = planes[3:3+2*(blocks-1)]
            for i in range(1, blocks):
                layers.append(block(self.inplanes, later_planes[2*(i-1)], later_planes[2*(i-1)+1], norm_layer=norm_layer))
            return nn.Sequential(*layers)
        else:
            first_planes = planes[0:2]
            # downsample at each 1'st layer, for pruning
            downsample = nn.Sequential(OrderedDict([('conv', conv1x1(self.inplanes, first_planes[-1], stride)),
                                                    ('bn', norm_layer(first_planes[-1]))]))
            layers = []
            layers.append(block(self.inplanes, first_planes[0], first_planes[1], stride, downsample, norm_layer))
            self.inplanes = first_planes[-1]
            later_planes = planes[2:2+blocks-1]
            for i in range(1, blocks):
                layers.append(block(self.inplanes, later_planes[i-1], norm_layer=norm_layer))
            return nn.Sequential(*layers)

    def cfg2params(self, cfg):
        blocks = self.blocks
        params = 0.
        params += (3 * 7 * 7 * cfg[0] + 2 * cfg[0]) # first layer
        inplanes = cfg[0]
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2],
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], 
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4],
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]]
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4):
            planes = sub_cfgs[i]
            if self.depth != 50:
                first_planes = planes[0:2]
                later_planes = planes[2:2+blocks[i]-1]
            else:
                first_planes = planes[0:3]
                later_planes = planes[3:3+2*(blocks[i]-1)]
            params += (inplanes * 1 * 1 * first_planes[-1] + 2 * first_planes[-1]) # downsample layer
            if self.depth != 50:
                params += (inplanes * 3 * 3 * first_planes[0] + 2 * first_planes[0])
                params += (first_planes[0] * 3 * 3 * first_planes[1] + 2 * first_planes[1])
            else:
                params += (inplanes * 1 * 1 * first_planes[0] + 2 * first_planes[0])
                params += (first_planes[0] * 3 * 3 * first_planes[1] + 2 * first_planes[1])
                params += (first_planes[1] * 1 * 1 * first_planes[2] + 2 * first_planes[2])
            for j in range(1, self.blocks[i]):
                inplanes = first_planes[-1]
                if self.depth != 50:
                    params += (inplanes * 3 * 3 * later_planes[j-1] + 2 * later_planes[j-1])
                    params += (later_planes[j-1] * 3 * 3 * inplanes + 2 * inplanes)
                else:
                    params += (inplanes * 1 * 1 * later_planes[2*(j-1)] + 2 * later_planes[2*(j-1)])
                    params += (later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] + 2 * later_planes[2*(j-1)+1])
                    params += (later_planes[2*(j-1)+1] * 1 * 1 * inplanes + 2 * inplanes)
        if self.depth==50:
            params += (cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+6] + 1) * self.num_classes
        else:
            params += (cfg[blocks[0]+blocks[1]+blocks[2]+5] + 1) * self.num_classes
        return params

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        blocks = self.blocks
        flops = 0.
        size = 224
        size /= 2 # first conv layer s=2
        flops += (3 * 7 * 7 * cfg[0] * size * size + 5 * cfg[0] * size * size) # first layer, conv+bn+relu
        inplanes = cfg[0]
        size /= 2 # pooling s=2
        flops += (3 * 3 * cfg[0] * size * size) # maxpooling
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2],
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], 
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4],
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]]
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4): # each layer
            planes = sub_cfgs[i]
            if self.depth != 50:
                first_planes = planes[0:2]
                later_planes = planes[2:2+blocks[i]-1]
            else:
                first_planes = planes[0:3]
                later_planes = planes[3:3+2*(blocks[i]-1)]
            if i in [1, 2, 3]:
                size /= 2
            flops += (inplanes * 1 * 1 * first_planes[-1] + 5 * first_planes[-1]) * size * size # downsample layer
            if self.depth != 50:
                flops += (inplanes * 3 * 3 * first_planes[0] + 5 * first_planes[0]) * size * size
                flops += (first_planes[0] * 3 * 3 * first_planes[1] + 5 * first_planes[1]) * size * size
            else:
                size *= 2
                flops += (inplanes * 1 * 1 * first_planes[0] + 5 * first_planes[0]) * size * size
                size /= 2
                flops += (first_planes[0] * 3 * 3 * first_planes[1] + 5 * first_planes[1]) * size * size
                flops += (first_planes[1] * 1 * 1 * first_planes[2] + 5 * first_planes[2]) * size * size
            for j in range(1, self.blocks[i]):
                inplanes = first_planes[-1]
                if self.depth != 50:
                    flops += (inplanes * 3 * 3 * later_planes[j-1] + 5 * later_planes[j-1]) * size * size
                    flops += (later_planes[j-1] * 3 * 3 * inplanes + 5 * inplanes) * size * size
                else:
                    flops += (inplanes * 1 * 1 * later_planes[2*(j-1)] + 5 * later_planes[2*(j-1)]) * size * size
                    flops += (later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] + 5 * later_planes[2*(j-1)+1]) * size * size
                    flops += (later_planes[2*(j-1)+1] * 1 * 1 * inplanes + 5 * inplanes) * size * size
        flops += (2 * cfg[-1] + 1) * self.num_classes
        return flops

    def cfg2flops_perlayer(self, cfg, length):  # to simplify, only count convolution flops
        blocks = self.blocks
        flops_singlecfg = [0 for j in range(length)]
        flops_doublecfg = np.zeros((length, length))
        flops_squarecfg = [0 for j in range(length)]
        size = 224
        size /= 2 # first conv layer s=2
        flops_singlecfg[0] += (3 * 7 * 7 * cfg[0] * size * size + 5 * cfg[0] * size * size) # first layer, conv+bn+relu
        inplanes = cfg[0]
        size /= 2 # pooling s=2
        flops_singlecfg[0] += (3 * 3 * cfg[0] * size * size) # maxpooling
        if self.depth != 50:
            count = 2
        else:
            count = 3
        count1 = 0
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2],
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1],
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4],
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]]
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4): # each layer
            planes = sub_cfgs[i]
            if self.depth != 50:
                first_planes = planes[0:2]
                later_planes = planes[2:2+blocks[i]-1]
            else:
                first_planes = planes[0:3]
                later_planes = planes[3:3+2*(blocks[i]-1)]
            if i in [1, 2, 3]:
                size /= 2
                if self.depth != 50:
                    count += blocks[i-1]+1
                else:
                    count += 2 * blocks[i-1] + 1
            flops_doublecfg[count][count1] += inplanes * 1 * 1 * first_planes[-1] * size * size # downsample layer
            flops_doublecfg[count1][count] += inplanes * 1 * 1 * first_planes[-1] * size * size
            flops_singlecfg[count] += 5 * first_planes[-1] * size * size
            if self.depth != 50:
                flops_doublecfg[count-1][count1] += inplanes * 3 * 3 * first_planes[0] * size * size
                flops_doublecfg[count1][count-1] += inplanes * 3 * 3 * first_planes[0] * size * size
                flops_singlecfg[count-1] += 5 * first_planes[0] * size * size

                flops_doublecfg[count-1][count] += first_planes[0] * 3 * 3 * first_planes[1] * size * size
                flops_doublecfg[count][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size
                flops_singlecfg[count] += 5 * first_planes[1] * size * size
            else:
                size *= 2
                flops_doublecfg[count-2][count1] += inplanes * 1 * 1 * first_planes[0] * size * size
                flops_doublecfg[count1][count-2] += inplanes * 1 * 1 * first_planes[0] * size * size
                flops_singlecfg[count-2] += 5 * first_planes[0] * size * size

                size /= 2
                flops_doublecfg[count-2][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size
                flops_doublecfg[count-1][count-2] += first_planes[0] * 3 * 3 * first_planes[1] * size * size
                flops_singlecfg[count-1] += 5 * first_planes[1] * size * size

                flops_doublecfg[count][count-1] += first_planes[1] * 1 * 1 * first_planes[2] * size * size
                flops_doublecfg[count-1][count] += first_planes[1] * 1 * 1 * first_planes[2] * size * size
                flops_singlecfg[count] += 5 * first_planes[2] * size * size
            for j in range(1, self.blocks[i]):
                inplanes = first_planes[-1]
                count1 = count
                if self.depth != 50:
                    flops_doublecfg[count1][count+j] += inplanes * 3 * 3 * later_planes[j-1] * size * size
                    flops_doublecfg[count+j][count1] += inplanes * 3 * 3 * later_planes[j-1] * size * size
                    flops_singlecfg[count+j] += 5 * later_planes[j-1] * size * size

                    flops_doublecfg[count1][count+j] += later_planes[j-1] * 3 * 3 * inplanes * size * size
                    flops_doublecfg[count+j][count1] += later_planes[j-1] * 3 * 3 * inplanes * size * size
                    flops_singlecfg[count1] += 5 * inplanes * size * size
                else:
                    flops_doublecfg[count1][count+1+2*(j-1)] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size
                    flops_doublecfg[count+1+2*(j-1)][count1] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size
                    flops_singlecfg[count+1+2*(j-1)] += 5 * later_planes[2*(j-1)] * size * size

                    flops_doublecfg[count+2+2*(j-1)][count+1+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size
                    flops_doublecfg[count+1+2*(j-1)][count+2+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size
                    flops_singlecfg[count+2+2*(j-1)] += 5 * later_planes[2*(j-1)+1] * size * size

                    flops_doublecfg[count+2+2*(j-1)][count1] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size
                    flops_doublecfg[count1][count+2+2*(j-1)] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size
                    flops_singlecfg[count1] += 5 * inplanes * size * size
        flops_singlecfg[-1] += (2 * cfg[-1] + 1) * self.num_classes
        return flops_singlecfg, flops_doublecfg, flops_squarecfg

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def feature_extract(self, x):
        tensor = []
        count = 0
        x = self.conv1(x)
        tensor.append(x)
        x = self.maxpool(x)
        if self.depth !=50:
            for _layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in _layer:
                    if count == 0:
                        tensor.append(block.conv1(x))
                    x = block(x)
                    tensor.append(x)
                    count += 1
                count = 0
        else:
            for _layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                count = 0
                for block in _layer:
                    if count == 0:
                        tensor.append(block.conv1(x))
                        tensor.append(block.conv2(block.conv1(x)))
                        x = block(x)
                        tensor.append(x)
                    else:
                        tensor.append(block.conv1(x))
                        tensor.append(block.conv2(block.conv1(x)))
                        x = block(x)
                    count += 1
        return tensor

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'depth': self.depth,
            'cfg': self.cfg,
            'cfg_base': self.cfgs_base[self.depth],
            'dataset': 'ImageNet',
        }
