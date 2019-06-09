import os
import torch
from torch import nn
from torch.nn import functional as F 
import torch.utils.data as td
import torchvision as tv
import glob
import re
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
from copy import deepcopy

# Data Augmentation
# Transforms classes
# Random scale
# Flip
# x, y reposition

classes =  np.array(['sheep', 'horse', 'bicycle', 'bottle', 'cow', 'sofa', 'car', 'dog', 'cat', 'person', 'train', 'diningtable', 'aeroplane', 'bus', 'pottedplant', 'tvmonitor', 'chair', 'bird', 'boat', 'motorbike'])

class RandomCrop(object):
    
    def imcv2_affine_trans(self, im):
        # Scale and translate
        h, w, c = im.shape
        scale = np.random.uniform() / 10. + 1.
        max_offx = (scale-1.) * w
        max_offy = (scale-1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)

        im = cv2.resize(im, (0,0), fx = scale, fy = scale)
        im = im[offy : (offy + h), offx : (offx + w)]

        return im, [w, h], [scale, [offx, offy]]

    
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
#         image, bboxes = sample[0], sample[1]
        result = self.imcv2_affine_trans(image)
        image, dims, trans_param = result
        scale, offs = trans_param
        
        offs = np.array(offs*2)
        dims = np.array(dims*2)
        bboxes = deepcopy(bboxes)
        bboxes[:, 1:] = np.array(bboxes[:, 1:]*scale - offs, np.int64)
        bboxes[:, 1:] = np.maximum(np.minimum(bboxes[:, 1:], dims), 0)
        
        check_errors = (((bboxes[:, 1] >= bboxes[:, 3]) | (bboxes[:, 2] >= bboxes[:, 4])) & (bboxes[:, 0]!=-1))
        if sum(check_errors) > 0:
            bool_mask = ~ check_errors
            bboxes = bboxes[bool_mask]
#         return image, bboxes
        return {"image": image, "bboxes": bboxes}

class RandomFlip(object):
    
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
#         image, bboxes = sample[0], sample[1]

        bboxes = deepcopy(bboxes)
        flip = np.random.binomial(1, .5)
        if flip: 
            w = image.shape[1]
            image = cv2.flip(image, 1)
            backup_min = deepcopy(bboxes[:, 1])
            bboxes[:, 1] = w - bboxes[:, 3]
            bboxes[:, 3] = w - backup_min
        
        if sum(((bboxes[:, 1] >= bboxes[:, 3]) | (bboxes[:, 2] >= bboxes[:, 4])) & (bboxes[:, 0]!=-1)) > 0:
            print ("random flip")
        
#         return image, bboxes
        return {"image": image, "bboxes": bboxes}

class Rescale(object):
    
    def __init__(self, output):
        self.new_h, self.new_w = output
        
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
#         image, bboxes = sample

        h, w, c = image.shape
        new_h = int(self.new_h)
        new_w = int(self.new_w)
        image = cv2.resize(image, (new_w, new_h))
        
        bboxes = deepcopy(bboxes)
        bboxes = np.array(bboxes, np.float64)
        bboxes[:, 1] *= new_w*1.0/w
        bboxes[:, 2] *= new_h*1.0/h
        bboxes[:, 3] *= new_w*1.0/w
        bboxes[:, 4] *= new_h*1.0/h
        if sum(((bboxes[:, 1] >= bboxes[:, 3]) | (bboxes[:, 2] >= bboxes[:, 4])) & (bboxes[:, 0]!=-1)) > 0:
            print ("random scale", bboxes, sample['bboxes'], new_w, new_h, w, h)

        return {"image": image, "bboxes": bboxes}
#         return image, bboxes

class TransformBoxCoords(object):
    
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
#         image, bboxes = sample[0], sample[1]
        height, width, _ = image.shape
        
        bboxes = deepcopy(bboxes)
        bboxes = np.array(bboxes, np.float64)
        x = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
        y = 0.5 * (bboxes[:, 2] + bboxes[:, 4])
        w = 1. * (bboxes[:, 3] - bboxes[:, 1])
        h = 1. * (bboxes[:, 4] - bboxes[:, 2])
        if sum(((w <= 0) | (h <= 0) | (x <= 0) | (y <= 0)) & (bboxes[:, 0]!=-1))>0:
            print ("up", bboxes, sample["bboxes"])
        bboxes[:, 1] = x/width
        bboxes[:, 2] = y/height
        bboxes[:, 3] = w/width
        bboxes[:, 4] = h/height
        if sum(((bboxes[:, 1] <0) | (bboxes[:, 2]<0) | (bboxes[:, 3]<=0) | (bboxes[:, 4]<=0)) & (bboxes[:, 0]!=-1)) > 0:
            print ("random transform box coords")

#         return image, bboxes
        return {"image": image, "bboxes": bboxes}

class Normalize(object):
    
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
#         image, bboxes = sample[0], sample[1]
        image = np.array(image, np.float64)
        image = image * 2 / 255.0 - 1
        return {"image": image, "bboxes": bboxes}
#         return image, bboxes

class EliminateSmallBoxes(object):
    
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
#         image, bboxes = sample[0], sample[1]
        bool_mask = ((bboxes[: , 3] > self.thresh) & (bboxes[: , 4] > self.thresh))
        bboxes = bboxes[bool_mask]
        return {"image": image, "bboxes": bboxes}
#         return image, bboxes


class ToTensor(object):

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
#         image, bboxes = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        if len(bboxes) == 0:
            return {'image': torch.from_numpy(image), 'bboxes': torch.DoubleTensor()}
#             return torch.from_numpy(image), torch.DoubleTensor()
        return {'image': torch.from_numpy(image), 'bboxes': torch.from_numpy(bboxes)}
#         return torch.from_numpy(image), torch.from_numpy(bboxes)
    

class VOCDataset(td.Dataset):
    def __init__(self, root_dir, infiles=[], sample=-1, transform=None, max_truth=30):
        imgnames = glob.glob(root_dir+"/JPEGImages/*jpg")
        imgnames = [re.split("\\.", (re.split("\\/", imgnames[i]))[-1])[0] for i in range(len(imgnames))]
        self.imgnames = list(set(imgnames) & set(infiles))  ## get a specific type of data(eg train) by &infiles
        np.random.shuffle(self.imgnames)
        if sample != -1:
            self.imgnames = self.imgnames[:sample]
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.root_dir = root_dir
        self.transform = transform
        self.max_truth = max_truth
    
    def __len__(self):
        return len(self.imgnames)
    
    def parseXML(self, XML):
        inXML = open(XML)
        tree = ET.parse(inXML)
        tree_root = tree.getroot()
        ## find all object boxes
        boxes = []
        for obj in tree_root.iter("object"):
            objname = obj.find("name").text
            class_id = np.argwhere(classes==objname)[0][0]
            pos = obj.find("bndbox")
            xmin = int(pos.find("xmin").text)
            ymin = int(float(pos.find("ymin").text))
            xmax = int(pos.find("xmax").text)
            ymax = int(pos.find("ymax").text)
            box = [class_id, xmin, ymin, xmax, ymax]
            boxes.append(box)
        inXML.close()
        return np.array(boxes)
    
    def __getitem__(self, idx):
        img_name = self.imgnames[idx]
        XML_file = os.path.join(self.root_dir, "Annotations", img_name+".xml")
        img_file = os.path.join(self.root_dir, "JPEGImages", img_name+".jpg")
        img_array = np.asarray(Image.open(img_file))
        bboxes = self.parseXML(XML_file)
        data = {"image": img_array, "bboxes": bboxes}
        ## transform if needed
        if self.transform:
            data = self.transform(data)
        bboxes = data['bboxes'].numpy()
        n_true = len(bboxes)
        if n_true > self.max_truth:
            n_true = self.max_truth
            bboxes = bboxes[0:n_true]
        else:
            zero_fill = self.max_truth-n_true
            nullbox = -1*(np.ones(5*zero_fill).reshape(zero_fill, 5))
            if n_true == 0:
                bboxes = nullbox
            else:
                bboxes = np.concatenate([bboxes, nullbox])
        data["bboxes"] = torch.from_numpy(bboxes)
        data["n_true"] = torch.tensor(n_true)
        return data
    
def get_traintest(root_dir):
    imgtt_dir = os.path.join(root_dir, "ImageSets", "Main")
    train = pd.read_csv(os.path.join(imgtt_dir, "train.txt"), sep='\s+', header=-1)
    val = pd.read_csv(os.path.join(imgtt_dir, "val.txt"), sep='\s+', header=-1)
    VOCtrain = pd.concat([train, val]).drop_duplicates()
    VOCtrain.columns = ["filename"]
    VOCtrain["train"] = 1
    imagesname = glob.glob(root_dir+"/JPEGImages/*jpg")
    imagesname = [re.split("\\.", (re.split("\\/", imagesname[i]))[-1])[0] for i in range(len(imagesname))]
    VOCtotal = pd.DataFrame({"filename": imagesname})
    VOCtotal = pd.merge(VOCtotal, VOCtrain, on="filename", how="left")
    VOCtotal.fillna(0, inplace=True)
    return VOCtotal

def getdata(root_dir, image_size=416, sample=-1, batch_size=64):
    transform = tv.transforms.Compose([
        ### incomplete, bunch of transform funtions needed to add
        RandomCrop(),
        RandomFlip(),
        Rescale((image_size, image_size)),
        TransformBoxCoords(),
        Normalize(),
        EliminateSmallBoxes(0.025),
        ToTensor(),
    ])
    VOCtotal = get_traintest(root_dir)
    VOCtrain = VOCDataset(root_dir, infiles=(VOCtotal[VOCtotal["train"]==1])["filename"].values, \
                          sample=sample, transform=transform)
    VOCtest = VOCDataset(root_dir, infiles=(VOCtotal[VOCtotal["train"]==0])["filename"].values, \
                          sample=sample, transform=transform)
#     train_loader = td.DataLoader(dataset=VOCtrain, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_loader = td.DataLoader(dataset=VOCtest, batch_size=batch_size, shuffle=True, num_workers=4)
    return VOCtrain, VOCtest

