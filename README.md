# ece285ZHQL

Description
===========
This is project Multi-Object detection using Deep Learning developed by team ZHQL composed of Zifan Qian, Wenbo Zhou, Qinyan Li, Yunzhe Hu

Requirements
============
cuda
pytorch
Please download pretrained and trained weights from google drive:

Code organization
=================
demo.ipynb        --  Run a demo of our code
YOLOv2_train.ipynb -- Run the training of our model (3 experiments)
Data1.py -- Load data and data augmentation
bbox.py -- bounding box related functions
build.py -- network architecture
loss.py -- loss function
nntool.py -- training tools modified from nntools
utils.py -- post processing and draw bounding boxes
