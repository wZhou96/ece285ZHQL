#!/usr/bin/env python
# coding: utf-8

# In[2]:


from collections import defaultdict
import struct
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision as tv
from torch.autograd import Variable

import nntools as nt

# In[4]:


class Yolov2(nt.NeuralNetwork):
    
    def __init__(self):
        super(Yolov2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)     
        self.conv4 = nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)     
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)  
        self.conv7 = nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(128)        
        self.conv8 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        
        self.conv9 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        
        self.conv14 = nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(1024)
        self.conv15 = nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(1024)
        self.conv17 = nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False)
        self.bn17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(1024)
        
        self.conv19 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv19.weight.data)
        self.bn19 = nn.BatchNorm2d(1024)
        nn.init.ones_(self.bn19.weight.data)
        
        self.conv20 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv20.weight.data)
        self.bn20 = nn.BatchNorm2d(1024)
        nn.init.ones_(self.bn20.weight.data)
        
        self.conv21 = nn.Conv2d(3072, 1024, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv21.weight.data)
        self.bn21 = nn.BatchNorm2d(1024)
        nn.init.ones_(self.bn21.weight.data)
        
        self.conv22 = nn.Conv2d(1024, 125, 1, stride=1, padding=0)
        nn.init.xavier_normal_(self.conv22.weight.data)
        
    def reorg_layer(self, x):
        stride = 2
        batch_size, channels, height, width = x.shape
        h = height // stride
        w = width // stride
        c = channels * stride * stride
        
        y = x.permute(0, 2, 3, 1)
        y = y.contiguous().view(-1, h, stride, w, stride, channels)
        y = y.permute(0, 1, 3, 2, 4, 5)
        y = y.contiguous().view(-1, h, w, c)
        y = y.permute(0, 3, 1, 2)
        return y

    
    def forward(self, x):
        
        h = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1), 2, stride=2)
        h = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(h)), negative_slope=0.1), 2, stride=2)
        
        h = F.leaky_relu(self.bn3(self.conv3(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn4(self.conv4(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn5(self.conv5(h)), negative_slope=0.1)
        h = F.max_pool2d(h, 2, stride=2)
        
        h = F.leaky_relu(self.bn6(self.conv6(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn7(self.conv7(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn8(self.conv8(h)), negative_slope=0.1)
        h = F.max_pool2d(h, 2, stride=2)

        h = F.leaky_relu(self.bn9(self.conv9(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn10(self.conv10(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn11(self.conv11(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn12(self.conv12(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn13(self.conv13(h)), negative_slope=0.1)
        
        y = self.reorg_layer(h)
        h = F.max_pool2d(h, 2, stride=2)

        h = F.leaky_relu(self.bn14(self.conv14(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn15(self.conv15(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn16(self.conv16(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn17(self.conv17(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn18(self.conv18(h)), negative_slope=0.1)

        h = F.leaky_relu(self.bn19(self.conv19(h)), negative_slope=0.1)
        h = F.leaky_relu(self.bn20(self.conv20(h)), negative_slope=0.1)
        
        h = torch.cat([y, h], 1)
        h = F.leaky_relu(self.bn21(self.conv21(h)), negative_slope=0.1)
        h = self.conv22(h)

        return h


def load_pretrained_weights(model):
    group_mapping = defaultdict(lambda: defaultdict())
    cnt = 0
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
