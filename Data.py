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

class VOCDataset(td.Dataset):
    def __init__(self, root_dir, infiles=[], sample=-1, transform=None, max_truth=30):
        imgnames = glob.glob(root_dir+"/JPEGImages/*jpg")
        imgnames = [re.split("\\.", (re.split("\\/", imgnames[i]))[-1])[0] for i in range(len(imgnames))]
        self.imgnames = list(set(imgnames) & set(infiles))  ## get a specific type of data(eg train) by &infiles
        np.random.shuffle(self.imgnames)
        if sample != -1:
            self.imgnames = self.imgnames[:sample]
        self.root_dir = root_dir
        self.transform = transform
        self.max_truth = max_truth
    
    def __len__(self):
        return len(self.imgnames)
    
    def __parseXML__(self, XML):
        inXML = open(XML)
        tree = ET.parse(inXML)
        tree_root = tree.getroot()
        ## find all object boxes
        boxes = []
        for obj in tree_root.find("object"):
            objname = obj.find("name").text
            pos = obj.find("bndbox")
            xmin = int(pos.find("xmin").text)
            ymin = int(pos.find("ymin").text)
            xmax = int(pos.find("xmax").text)
            ymax = int(pos.find("ymax").text)
            box = [objname, xmin, ymin, xmax, ymax]
            boxes.append(box)
        inXML.close()
        return boxes
    
    def __getitem__(self, idx):
        img_name = self.imgnames[idx]
        XML_file = os.path.join(self.root_dir, "Annotations", img_name+".xml")
        img_file = os.path.join(self.root_dir, "JPEGImages", img_name+".jpg")
        img_array = np.asarray(Image.open(img_file))
        bboxes = parseXML(XML_file)
        data = {"image": img_array, "bboxes": bboxes}
        ## transform if needed
        if self.transform:
            data = self.transform(data)
        bboxes = bboxes.numpy()
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
        data["n_true"] = n_true
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
        tv.transforms.Resize(image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    VOCtotal = get_traintest(root_dir)
    VOCtrain = VOCDataset(root_dir, infiles=(VOCtotal[VOCtotal["train"]==1])["filename"].values, \
                          sample=sample, transform=transform)
    VOCtest = VOCDataset(root_dir, infiles=(VOCtotal[VOCtotal["train"]==0])["filename"].values, \
                          sample=sample, transform=transform)
    train_loader = td.DataLoader(dataset=VOCtrain, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = td.DataLoader(dataset=VOCtest, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader

