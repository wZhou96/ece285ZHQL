#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[4]:


class Yolov2(nn.Module):
    
    def __int__(self):
        super(Yolov2,self).__init__()
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(32)
        
        self.conv.append(conv1)
        self.bn.append(bn1)
        
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(64)
        
        self.conv.append(conv2)
        self.bn.append(bn2)
        
        conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(128)
        
        self.conv.append(conv3)
        self.bn.append(bn3)
        
        conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        bn4 = nn.BatchNorm2d(64)
        
        self.conv.append(conv4)
        self.bn.append(bn4)
        
        conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        bn5 = nn.BatchNorm2d(128)
        
        self.conv.append(conv5)
        self.bn.append(bn5)
        
        conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        bn6 = nn.BatchNorm2d(256)
        
        self.conv.append(conv6)
        self.bn.append(bn6)
        
        conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        bn7 = nn.BatchNorm2d(128)
        
        self.conv.append(conv7)
        self.bn.append(bn7)
        
        conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        bn8 = nn.BatchNorm2d(256)
        
        self.conv.append(conv8)
        self.bn.append(bn8)
        
        conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        bn9 = nn.BatchNorm2d(512)
        
        self.conv.append(conv9)
        self.bn.append(bn9)
        
        conv10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        bn10 = nn.BatchNorm2d(256)
        
        self.conv.append(conv10)
        self.bn.append(bn10)
        
        conv11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        bn11 = nn.BatchNorm2d(512)
        
        self.conv.append(conv11)
        self.bn.append(bn11)
        
        conv12 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        bn12 = nn.BatchNorm2d(256)
        
        self.conv.append(conv12)
        self.bn.append(bn12)
        
        conv13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        bn13 = nn.BatchNorm2d(512)
        
        self.conv.append(conv13)
        self.bn.append(bn13)
        
        conv14 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn14 = nn.BatchNorm2d(1024)
        
        self.conv.append(conv14)
        self.bn.append(bn14)
        
        conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        bn15 = nn.BatchNorm2d(512)
        
        self.conv.append(conv15)
        self.bn.append(bn15)
        
        conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn16 = nn.BatchNorm2d(1024)
        
        self.conv.append(conv16)
        self.bn.append(bn16)
        
        conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        bn17 = nn.BatchNorm2d(512)
        
        self.conv.append(conv17)
        self.bn.append(bn17)
        
        conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn18 = nn.BatchNorm2d(1024)
        
        self.conv.append(conv18)
        self.bn.append(bn18)

        conv19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn19 = nn.BatchNorm2d(1024)
        
        self.conv.append(conv19)
        self.bn.append(bn19)
        
        conv20 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn20 = nn.BatchNorm2d(1024)
        
        self.conv.append(conv20)
        self.bn.append(bn20)
        
        conv21 = nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        bn21 = nn.BatchNorm2d(1024)
        
        self.conv.append(conv21)
        self.bn.append(bn21)
        
        conv22 = nn.Conv2d(in_channels=1024, out_channels=125, kernel_size=1, stride=1, padding=0)
        self.conv.append(conv22)
        
    def reorg_layer(self, x):
        stride = 2
        batch_size, channels, height, width = x.size()
        new_ht = height/stride
        new_wd = width/stride
        new_channels = channels * stride * stride
        
        reorg = x.permute(0, 2, 3, 1)
        reorg = reorg.contiguous().view(-1, new_ht, stride, new_wd, stride, channels)
        reorg = reorg.permute(0, 1, 3, 2, 4, 5)
        reorg = reorg.contiguous().view(-1, new_ht, new_wd, new_channels)
        reorg = reorg.permute(0, 3, 1, 2)
        return reorg
    
    def forward(self, x):
        h = F.max_pool2d(F.leaky_relu(self.bn[0](self.conv[0](x)), negative_slope=0.1), 2, stride=2)
        h = F.max_pool2d(F.leaky_relu(self.bn[1](self.conv[1](h)), negative_slope=0.1), 2, stride=2)
        
        h = F.leaky_relu(self.bn[2](self.conv[2](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[3](self.conv[3](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[4](self.conv[4](h)), negative_slope=0.1)
        h = F.max_pool2d(h, 2, stride=2)
        
        h = F.leaky_relu(self.bn[5](self.conv[5](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[6](self.conv[6](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[7](self.conv[7](h)), negative_slope=0.1)
        h = F.max_pool2d(h, 2, stride=2)

        h = F.leaky_relu(self.bn[8](self.conv[8](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[9](self.conv[9](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[10](self.conv[10](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[11](self.conv[11](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[12](self.conv[12](h)), negative_slope=0.1)
        reorg = self.reorg_layer(h)
        h = F.max_pool2d(h, 2, stride=2)

        h = F.leaky_relu(self.bn[13](self.conv[13](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[14](self.conv[14](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[15](self.conv[15](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[16](self.conv[16](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[17](self.conv[17](h)), negative_slope=0.1)

        h = F.leaky_relu(self.bn[18](self.conv[18](h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn[19](self.conv[19](h)), negative_slope=0.1)
        
        h = torch.cat([reorg, h], 1)
        h = F.leaky_relu(self.bn[20](self.conv[20](h)), negative_slope=0.1)
        y = self.conv[21](h)

        return y 


# In[ ]:




