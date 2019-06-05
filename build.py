#!/usr/bin/env python
# coding: utf-8


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
import os
import glob
from PIL import Image
from collections import defaultdict
import math
from copy import deepcopy
import pandas as pd
import struct, os
import re, numpy as np
import itertools, operator
import pickle
from torch.optim.lr_scheduler import _LRScheduler
from nltk.corpus import wordnet as wn

import os
import sys
import xml.etree.ElementTree as ET
import glob

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import yolo2loss
import nntools as nt



class Yolov2(nt.NeuralNetwork):
    
    def __init__(self):
        super(Yolov2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(256)
        
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm13 = nn.BatchNorm2d(512)
        
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm14 = nn.BatchNorm2d(1024)
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm16 = nn.BatchNorm2d(1024)
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm18 = nn.BatchNorm2d(1024)

        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm19 = nn.BatchNorm2d(1024)
        self.conv20 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm20 = nn.BatchNorm2d(1024)
        
        self.conv21 = nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm21 = nn.BatchNorm2d(1024)
        
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=125, kernel_size=1, stride=1, padding=0)
        
    def reorg_layer(self, x):
        stride = 2
        batch_size, channels, height, width = x.size()
        new_ht = int(height/stride)
        new_wd = int(width/stride)
        new_channels = channels * stride * stride
        
        passthrough = x.permute(0, 2, 3, 1)
        passthrough = passthrough.contiguous().view(-1, new_ht, stride, new_wd, stride, channels)
        passthrough = passthrough.permute(0, 1, 3, 2, 4, 5)
        passthrough = passthrough.contiguous().view(-1, new_ht, new_wd, new_channels)
        passthrough = passthrough.permute(0, 3, 1, 2)
        return passthrough

    def forward(self, x):
        out = F.max_pool2d(F.leaky_relu(self.batchnorm1(self.conv1(x)), negative_slope=0.1), 2, stride=2)
        out = F.max_pool2d(F.leaky_relu(self.batchnorm2(self.conv2(out)), negative_slope=0.1), 2, stride=2)
        
        out = F.leaky_relu(self.batchnorm3(self.conv3(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm4(self.conv4(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm5(self.conv5(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)
        
        out = F.leaky_relu(self.batchnorm6(self.conv6(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm7(self.conv7(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm8(self.conv8(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm9(self.conv9(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm10(self.conv10(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm11(self.conv11(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm12(self.conv12(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm13(self.conv13(out)), negative_slope=0.1)
        passthrough = self.reorg_layer(out)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm14(self.conv14(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm15(self.conv15(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm16(self.conv16(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm17(self.conv17(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm18(self.conv18(out)), negative_slope=0.1)
        
        out = F.leaky_relu(self.batchnorm19(self.conv19(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm20(self.conv20(out)), negative_slope=0.1)
        
        out = torch.cat([passthrough, out], 1)
        out = F.leaky_relu(self.batchnorm21(self.conv21(out)), negative_slope=0.1)
        out = self.conv22(out)

        return out
    
    def criterion(self, y, d, n_truths):
        return yolo2loss.loss(y, d, n_truths)
        


# In[ ]:
def load_pretrained_weights(model):
    group_mapping = defaultdict(lambda: defaultdict())
    cnt = 0
    print(model.children())
    for child in model.children():
        if type(child) == nn.Conv2d:
            cnt += 1
            if cnt > 18:
                break
            group_mapping[cnt]['conv'] = child
            group_mapping[cnt]['bias'] = child
        else:
            group_mapping[cnt]['bias'] = child


    f = open('darknet53.conv.74', 'rb')
#     f = open('../yolo.weights', 'rb')
    major, minor, revision, seen = struct.unpack('4i', f.read(16))
    for i in range(1, 19):

        bias_var = group_mapping[i]['bias']
        cnt = int(bias_var.bias.size()[0])
        bias_var.bias.data = torch.from_numpy(np.array(struct.unpack('%df' % cnt, f.read(4*cnt)))).float()
        bias_var.weight.data = torch.from_numpy(np.array(struct.unpack('%df' % cnt, f.read(4*cnt)))).float()
        bias_var.running_mean = torch.from_numpy(np.array(struct.unpack('%df' % cnt, f.read(4*cnt)))).float()
        bias_var.running_var = torch.from_numpy(np.array(struct.unpack('%df' % cnt, f.read(4*cnt)))).float()

        for param in bias_var.parameters():
            param.requires_grad = False

        conv_var = group_mapping[i]['conv']
        c_out, c_in, f1, f2 = conv_var.weight.size()
        cnt = int(c_out * c_in * f1 * f2)
        p = struct.unpack('%df' % cnt, f.read(4*cnt))
        conv_var.weight.data = torch.from_numpy(np.reshape(p, [c_out, c_in, f1, f2])).float()
        for param in conv_var.parameters():
            param.requires_grad = False
    
    return model

def get_model():
    model = Yolov2()
    model = load_pretrained_weights(model)
    if torch.cuda.is_available():
        model.cuda()
    return model

    
def model_freeze_upto(model, layer_x):
    childs = list(model.children())
    for i in range(len(childs)):
        child = childs[i]
        for param in child.parameters():
            if i < layer_x:
                param.requires_grad = False
            else:
                param.requires_grad = True



