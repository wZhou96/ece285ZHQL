import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import numpy as np
from random import uniform
import cv2
import glob
import re
import pandas as pd



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for function_ in self.transforms:
            data = function_(data)
        return data


class Crop(object):
    def __init__(self, max_crop=0.1):
        super().__init__()
        self.max_crop = max_crop

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        xmin = width
        ymin = height
        xmax = 0
        ymax = 0
        for lb in label:
            xmin = min(xmin, lb[0])
            ymin = min(ymin, lb[1])
            xmax = max(xmax, lb[2])
            ymax = max(ymax, lb[2])
        cropped_left = uniform(0, self.max_crop)
        cropped_right = uniform(0, self.max_crop)
        cropped_top = uniform(0, self.max_crop)
        cropped_bottom = uniform(0, self.max_crop)
        new_xmin = int(min(cropped_left * width, xmin))
        new_ymin = int(min(cropped_top * height, ymin))
        new_xmax = int(max(width - 1 - cropped_right * width, xmax))
        new_ymax = int(max(height - 1 - cropped_bottom * height, ymax))

        image = image[new_ymin:new_ymax, new_xmin:new_xmax, :]
        label = [[
            lb[0] - new_xmin, lb[1] - new_ymin, lb[2] - new_xmin,
            lb[3] - new_ymin, lb[4]
        ] for lb in label]

        return image, label


class VerticalFlip(object):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        image, label = data
        if uniform(0, 1) >= self.prob:
            image = cv2.flip(image, 1)
            width = image.shape[1]
            label = [[width - lb[2], lb[1], width - lb[0], lb[3], lb[4]]
                     for lb in label]
        return image, label


class HSVAdjust(object):
    def __init__(self, hue=30, saturation=1.5, value=1.5, prob=0.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.prob = prob

    def __call__(self, data):
        def clip_hue(hue_channel):
            hue_channel[hue_channel >= 360] -= 360
            hue_channel[hue_channel < 0] += 360
            return hue_channel

        image, label = data
        adjust_hue = uniform(-self.hue, self.hue)
        adjust_saturation = uniform(1, self.saturation)
        if uniform(0, 1) >= self.prob:
            adjust_saturation = 1 / adjust_saturation
        adjust_value = uniform(1, self.value)
        if uniform(0, 1) >= self.prob:
            adjust_value = 1 / adjust_value
        image = image.astype(np.float32) / 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 0] += adjust_hue
        image[:, :, 0] = clip_hue(image[:, :, 0])
        image[:, :, 1] = np.clip(adjust_saturation * image[:, :, 1], 0.0, 1.0)
        image[:, :, 2] = np.clip(adjust_value * image[:, :, 2], 0.0, 1.0)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        image = (image * 255).astype(np.float32)

        return image, label


class Resize(object):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        width_ratio = float(self.image_size) / width
        height_ratio = float(self.image_size) / height
        new_label = []
        for lb in label:
            resized_xmin = lb[0] * width_ratio
            resized_ymin = lb[1] * height_ratio
            resized_xmax = lb[2] * width_ratio
            resized_ymax = lb[3] * height_ratio
            resize_width = resized_xmax - resized_xmin
            resize_height = resized_ymax - resized_ymin
            new_label.append([
                resized_xmin, resized_ymin, resize_width, resize_height, lb[4]
            ])

        return image, new_label


class VOCDataset(Dataset):
    def __init__(self,
                 root_dir,
                 infiles = [],
                 year="2007",
                 mode="train",
                 image_size=448,
                 is_training=True):
        self.root_dir = root_dir
        imgnames = glob.glob(root_dir+"/JPEGImages/*jpg")
        imgnames = [re.split("\\.", (re.split("\\/", imgnames[i]))[-1])[0] for i in range(len(imgnames))]
        self.imgnames = list(set(imgnames) & set(infiles))  ## get a specific type of data(eg train) by &infiles
        
#         if (mode in ["train", "val", "trainval", "test"]
#                 and year == "2007") or (mode in ["train", "val", "trainval"]
#                                         and year == "2012"):
#             self.data_path = os.path.join(root_path, "VOC{}".format(year))
#         id_list_path = os.path.join(self.data_path,
#                                     "ImageSets/Main/{}.txt".format(mode))
#         self.ids = [id.strip() for id in open(id_list_path)]

        self.classes = ['sheep', 'horse', 'bicycle', 'bottle', 'cow', 'sofa', 'car', 'dog', 'cat', 'person', 'train', 'diningtable', 'aeroplane', 'bus', 'pottedplant', 'tvmonitor', 'chair', 'bird', 'boat', 'motorbike']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.imgnames)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.imgnames[item]
        image_path = os.path.join(self.root_dir, "JPEGImages",
                                  "{}.jpg".format(id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_xml_path = os.path.join(self.root_dir, "Annotations",
                                      "{}.xml".format(id))
        annot = ET.parse(image_xml_path)

        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [
                int(obj.find('bndbox').find(tag).text) - 1
                for tag in ["xmin", "xmax", "ymin", "ymax"]
            ]
            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])
        if self.is_training:
            transformations = Compose(
                [HSVAdjust(),
                 VerticalFlip(),
                 Crop(),
                 Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))

        return np.transpose(np.array(image, dtype=np.float32),
                            (2, 0, 1)), np.array(objects, dtype=np.float32)
    
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

def getdata(root_dir, image_size=416, sample=-1, batch_size=16):
    VOCtotal = get_traintest(root_dir)
    VOCtrain = VOCDataset(root_dir, infiles=(VOCtotal[VOCtotal["train"]==1])["filename"].values)
    VOCtest = VOCDataset(root_dir, infiles=(VOCtotal[VOCtotal["train"]==0])["filename"].values)
#     train_loader = td.DataLoader(dataset=VOCtrain, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_loader = td.DataLoader(dataset=VOCtest, batch_size=batch_size, shuffle=True, num_workers=4)
    return VOCtrain, VOCtest
