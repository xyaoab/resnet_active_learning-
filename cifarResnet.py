import torch.nn as nn
import torch
import random
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, index=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        #the No. of block in each layer
        self.index = index
        #print("this is block",index)

    def forward(self, tup):
        x = tup[0]
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        # if no downsampling involved and the index matches 
        if tup[1] is not None and self.index in tup[1]:
           # only keep the shortcut path
            print("skip connections", self.index)
            out = residual 
        else:
            out += residual
            
        out = self.relu(out)

        return [out,tup[1]]

class ResNet(nn.Module):

    def __init__(self, n, d, num_classes=10):
        self.inplanes = d
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, d, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(d, n)
        self.layer2 = self._make_layer(d*2, n, stride=2)
        self.layer3 = self._make_layer(d*4, n, stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(d*4 * BasicBlock.expansion, num_classes)
        
        # record the number of blocks 
        self.num_block = n
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, index=i))

        return nn.Sequential(*layers)

    def forward(self, x, committee=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #generate random index of blocks in each layer to skip the connection
        #print("committee is ", committee)
        if committee == True:
            #skip the downsample block
            #drop three blocks at each resolution 
            drop1 =  random.sample(range(1,self.num_block),3)
            drop2 =  random.sample(range(1,self.num_block),3)
            drop3 =  random.sample(range(1,self.num_block),3)
            print("blocks to drop", drop1,drop2,drop3)
        else:
            drop1 = None
            drop2 = None
            drop3 = None
            
        x, _ = self.layer1([x,drop1])
        x, _ = self.layer2([x,drop2])
        x, _ = self.layer3([x,drop3])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
