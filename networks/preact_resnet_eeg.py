# -*-coding:utf-8-*-
# from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import math



def conv1x1 (in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x15 (in_planes, out_planes, stride=(1,2), padding = (0,7)):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,15), stride=stride, padding=padding, bias=False)

def conv1x7 (in_planes, out_planes, stride=(1,2), padding = (0,3)):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 7), stride=stride, padding=padding, bias=False)

def conv3x3 (in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv7x7 (in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2./n))
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1        = nn.BatchNorm2d(in_planes)
        self.conv1      = conv3x3(in_planes, planes, stride)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv2      = conv3x3(planes, planes)
        self.downsample = downsample
        # self.stride     = stride


    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        out += residual

        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1        = nn.BatchNorm2d(in_planes)
        self.conv1      = conv1x1(in_planes, planes)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv2      = conv3x3(planes, planes, stride=stride)
        self.bn3        = nn.BatchNorm2d(planes)
        self.conv3      = conv1x1(planes, planes*Bottleneck.expansion)
        self.downsample = downsample
        # self.strdie = stride


    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        out = self.bn3(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)

        out += residual

        return out


class PreActResNet(nn.Module):
    def __init__(self, depth, num_classes_list, dataset, bottleneck=False):
        super(PreActResNet, self).__init__()

        print('| Apply bottleneck: {TF}'.format(TF=bottleneck))

        num_classes = num_classes_list[0]
        self.dataset = dataset
        if self.dataset.startswith('NIH_EEG'):
            self.in_planes = 128

            if bottleneck == False:
                assert (depth - 2) % 6 == 0, \
                    'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
                n            = (depth - 2) // 6
                block_type   = BasicBlock

            else:
                assert (depth - 2) % 9 == 0, \
                    'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
                n            = (depth - 2) // 9
                block_type   = Bottleneck

            self.conv1 = conv1x15(128, self.in_planes, stride=(1, 3))
            self.bn1 = nn.BatchNorm2d(self.in_planes)
            self.conv2 = conv1x15(self.in_planes, self.in_planes, stride=(1, 2))
            self.bn2 = nn.BatchNorm2d(self.in_planes)
            # self.conv1  = conv3x3(128, self.in_planes)
            self.layer1 = self._make_layer(block_type, 256, n)
            self.layer2 = self._make_layer(block_type, 256, n, stride=2)
            self.layer3 = self._make_layer(block_type, 512, n, stride=2)
            self.layer4 = self._make_layer(block_type, 512, n, stride=2)
            self.bn3    = nn.BatchNorm2d(512 * block_type.expansion)
            self.fc     = nn.Linear(512 *2, num_classes)



    def _make_layer(self, block_type, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block_type.expansion, stride=stride),
                # nn.BatchNorm2d(planes * block_type.expansion),
            )

        layers = []
        layers.append(block_type(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block_type.expansion
        for i in range(1, blocks):
            layers.append(block_type(self.in_planes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        if self.dataset == 'NIH_EEG_V2':
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out, inplace=True)
            out = self.conv2(out)
            out = self.bn2(out)
            out = F.relu(out, inplace=True)
            # out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)


            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = self.bn3(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 7)
            # out = F.avg_pool2d(out, 7, stride=1)
            # =====================================================================================================================#
            # stride 1 이 필요한지 확인해보기

            out = out.view(out.size(0), -1)
            out = self.fc(out)

        return out


        # elif self.dataset == 'imagenet':
        #     out = self.conv1(x)
        #     out = self.bn1(out)
        #     out = F.relu(out, inplace=True)
        #     out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        #
        #     out = self.layer1(out)
        #     out = self.layer2(out)
        #     out = self.layer3(out)
        #     out = self.layer4(out)
        #
        #     out = self.bn2(out)
        #     out = F.relu(out)
        #     out = F.avg_pool2d(out, 7)
        #     # out = F.avg_pool2d(out, 7, stride=1)
        #     # =====================================================================================================================#
        #     # stride 1 이 필요한지 확인해보기
        #
        #     out = out.view(out.size(0), -1)
        #     out = self.fc(out)
        #
        #
        # return out










