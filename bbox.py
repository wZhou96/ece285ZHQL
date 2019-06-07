#!/usr/bin/env python
# coding: utf-8

# In[29]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import os
from collections import defaultdict
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as td 
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches as patches #
from torch.autograd import Variable
import itertools, operator


get_ipython().system('rm nntools.py')

get_ipython().system('ln -s /datasets/ee285f-public/nntools.py')
import nntools as nt


# In[19]:

meta = defaultdict()
meta['anchors'] = 5
meta['classes'] = 20
meta['threshold'] = 0.6
meta['anchor_bias'] = np.array([1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52])
meta['scale_no_obj'] = 1
meta['scale_coords'] = 1
meta['scale_class'] = 1
meta['scale_obj']  = 5
meta['iteration'] = 0 

def bbox_iou(bbox1, bbox2):
    
    x1, y1, w1, h1 = torch.chunk(bbox1, 4, dim = 1)
    x2, y2, w2, h2 = torch.chunk(bbox2, 4, dim = 1)
    
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    
    left = torch.max(x1-0.5*w1, (x2-0.5*w2).transpose(0, 1))
    top = torch.max(y1-0.5*h1, (y2-0.5*h2).transpose(0, 1))
    right = torch.min(x1+0.5*w1, (x2+0.5*w2).transpose(0, 1))
    bot = torch.min(y1+0.5*h1, (y2+0.5*h2).transpose(0, 1))
    
    inter_w = torch.clamp(right - left, 0)
    inter_h = torch.clamp(bot - top, 0)
    inter_area = inter_w * inter_h
    
    union = bbox1_area + bbox2_area.transpose(0, 1) - inter_area

    return inter_area / union


# In[47]:


def draw_bbox(img, bbox):
    
    x1, y1, w1, h1 = torch.chunk(bbox, 4, dim = 1)
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    for x, y, w, h in zip(x1, y1, w1, h1):
        rect = patches.Rectangle((x-0.5*w, y-0.5*h), w, h, linewidth=2, fill = False, color=np.random.rand(3,))
        ax.add_patch(rect)
    
    plt.show()
    
# im = plt.imread('./img-Copy1.png')
# a = torch.Tensor([[80, 35, 40, 30], [150, 80, 20, 60]])

# draw_bbox(im, a)


# In[ ]:


def get_nms_boxes(output, obj_thresh, iou_thresh):
#     import pdb; pdb.set_trace()
    N, C, H, W = output.size()
    N, C, H, W = int(N), int(C), int(H), int(W)
    B = meta['anchors']
    anchor_bias = meta['anchor_bias']
    n_classes = meta['classes']
    
    # -1 => unprocesse, 0 => suppressed, 1 => retained
    box_tags = Variable(-1 * torch.ones(H*W*B)).float()
    
    wh = Variable(torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2]))).float()
    anchor_bias_var = Variable(torch.from_numpy(np.reshape(anchor_bias, [1, 1, 1, B, 2]))).float()
    
    w_list = np.array(list(range(W)), np.float32)
    wh_ids = Variable(torch.from_numpy(np.array(list(map(lambda x: np.array(list(itertools.product(w_list, [x]))), range(H)))).reshape(1, H, W, 1, 2))).float() 
    
    if torch.cuda.is_available():
        wh = wh.cuda()
        wh_ids = wh_ids.cuda()
        box_tags = box_tags.cuda()
        anchor_bias_var = anchor_bias_var.cuda()                           

    anchor_bias_var = anchor_bias_var / wh

    predicted = output.permute(0, 2, 3, 1)
    predicted = predicted.contiguous().view(N, H, W, B, -1)
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=4)
    
    adjusted_xy = sigmoid(predicted[:, :, :, :, :2])
    adjusted_obj = sigmoid(predicted[:, :, :, :, 4:5])
    adjusted_classes = softmax(predicted[:, :, :, :, 5:])
    
    adjusted_coords = (adjusted_xy + wh_ids) / wh
    adjusted_wh = torch.exp(predicted[:, :, :, :, 2:4]) * anchor_bias_var

    batch_boxes = defaultdict()

    for n in range(N):
        
        scores = (adjusted_obj[n] * adjusted_classes[n]).contiguous().view(H*W*B, -1)
    
        class_probs = adjusted_classes[n].contiguous().view(H*W*B, -1)
        class_ids = torch.max(class_probs, 1)[1]
            
        pred_outputs = torch.cat([adjusted_coords[n], adjusted_wh[n]], 3)
        pred_bboxes = pred_outputs.contiguous().view(H*W*B, 4)
                
        ious = bbox_iou(pred_bboxes, pred_bboxes)
        
        confidences = adjusted_obj[n].contiguous().view(H*W*B)
        # get all boxes with tag -1
        final_boxes = Variable(torch.FloatTensor())
        if torch.cuda.is_available():
            final_boxes = final_boxes.cuda()
   
        for class_id in range(n_classes):
            bboxes_state = ((class_ids==class_id).float() * (scores[:, class_id] > obj_thresh).float() * box_tags).long().float()
                        
            while (torch.sum(bboxes_state==-1) > 0).data[0]:
                max_conf, index = torch.max(scores[:, class_id] * (bboxes_state==-1).float(), 0)
                bboxes_state = ((ious[index] < iou_thresh)[0].float() * bboxes_state).long().float()
                bboxes_state[index] = 1

                index_vals = torch.cat([pred_bboxes[index], confidences[index].view(1, 1), class_probs[index]], 1)
                if len(final_boxes.size()) == 0:
                    final_boxes = index_vals
                else:
                    final_boxes = torch.cat([final_boxes, index_vals], 0)
        
        batch_boxes[n] = final_boxes
        
    return batch_boxes


# In[ ]:


# Non Max Suppression
def get_nms_detections(output, obj_thresh, iou_thresh):
#     import pdb; pdb.set_trace()
    N, C, H, W = output.size()
    N, C, H, W = int(N), int(C), int(H), int(W)
    B = meta['anchors']
    anchor_bias = meta['anchor_bias']
    
    # -1 => unprocesse, 0 => suppressed, 1 => retained
    box_tags = Variable(-1 * torch.ones(H*W*B)).float()
    
    wh = Variable(torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2]))).float()
    anchor_bias_var = Variable(torch.from_numpy(np.reshape(anchor_bias, [1, 1, 1, B, 2]))).float()
    
    w_list = np.array(list(range(W)), np.float32)
    wh_ids = Variable(torch.from_numpy(np.array(list(map(lambda x: np.array(list(itertools.product(w_list, [x]))), range(H)))).reshape(1, H, W, 1, 2))).float() 
    
    if torch.cuda.is_available():
        wh = wh.cuda()
        wh_ids = wh_ids.cuda()
        box_tags = box_tags.cuda()
        anchor_bias_var = anchor_bias_var.cuda()                           

    anchor_bias_var = anchor_bias_var / wh

    predicted = output.permute(0, 2, 3, 1)
    predicted = predicted.contiguous().view(N, H, W, B, -1)
    

    
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=4)
    
    adjusted_xy = sigmoid(predicted[:, :, :, :, :2])
    adjusted_obj = sigmoid(predicted[:, :, :, :, 4:5])
    adjusted_classes = softmax(predicted[:, :, :, :, 5:])
    
    adjusted_coords = (adjusted_xy + wh_ids) / wh
    adjusted_wh = torch.exp(predicted[:, :, :, :, 2:4]) * anchor_bias_var

    batch_boxes = defaultdict()

    for n in range(N):
        
        class_probs = adjusted_classes[n].contiguous().view(H*W*B, -1)
        pred_outputs = torch.cat([adjusted_coords[n], adjusted_wh[n]], 3)
        pred_bboxes = pred_outputs.contiguous().view(H*W*B, 4)
        ious = bbox_iou(pred_bboxes, pred_bboxes)
        
        confidences = adjusted_obj[n].contiguous().view(H*W*B)
        bboxes_state = ((confidences>obj_thresh).float() * box_tags).long().float()
        
        # get all boxes with tag -1
        final_boxes = Variable(torch.FloatTensor())
        if torch.cuda.is_available():
            final_boxes = final_boxes.cuda()
        while (torch.sum(bboxes_state==-1) > 0).data[0]:
            max_conf, index = torch.max(confidences * (bboxes_state==-1).float(), 0)
            bboxes_state = ((ious[index] < iou_thresh)[0].float() * bboxes_state).long().float()
            bboxes_state[index] = 1
            
            index_vals = torch.cat([pred_bboxes[index], confidences[index].view(1, 1), class_probs[index]], 1)
            if len(final_boxes.size()) == 0:
                final_boxes = index_vals
            else:
                final_boxes = torch.cat([final_boxes, index_vals], 0)
        
        batch_boxes[n] = final_boxes
        
    return batch_boxes


# In[ ]:


def calc_map(boxes_dict, iou_threshold=0.5):
#     import pdb; pdb.set_trace()
    v = Variable(torch.zeros(1))
    if torch.cuda.is_available():
        v = v.cuda()
    
    if (len(boxes_dict['ground_truth'].size())==0) | (len(boxes_dict['prediction'].size())==0):
        return v

    gt = boxes_dict['ground_truth']
    pr = boxes_dict['prediction']

    gt_matched = Variable(-torch.ones(gt.size(0)))
    pr_matched = Variable(-torch.ones(pr.size(0)))

    if torch.cuda.is_available():
        gt_matched = gt_matched.cuda()
        pr_matched = pr_matched.cuda()
            
    for i in range(len(pr)):
        b = pr[i]
        ious = bbox_overlap_iou(b[:4].view(1, 4), gt, True)
        matched_scores = (gt_matched == -1).float() * (ious[0]>iou_threshold).float() * ious[0]
        if torch.sum(matched_scores).data[0] > 0:
            gt_idx = torch.max(matched_scores, 0)[1]
            gt_matched[gt_idx] = i
            pr_matched[i] = gt_idx
        
    tp = (pr_matched != -1).float()
    fp = (pr_matched == -1).float()
    tp_cumsum = torch.cumsum(tp, 0)
    fp_cumsum = torch.cumsum(fp, 0)
    n_corrects = tp_cumsum * tp
    total = tp_cumsum + fp_cumsum
    precision = n_corrects / total
    for i in range(precision.size(0)):
        precision[i] = torch.max(precision[i:])

    average_precision = torch.sum(precision) / len(gt)
    return average_precision
    
    
def evaluation(ground_truths, nms_output, n_truths, iou_thresh):
#     import pdb; pdb.set_trace()
    N = ground_truths.size(0)
    
    mean_avg_precision = Variable(torch.FloatTensor([0]))
    if torch.cuda.is_available():
        mean_avg_precision = mean_avg_precision.cuda()

    for batch in range(int(N)):
        category_map = defaultdict(lambda: defaultdict(lambda: torch.FloatTensor()))
        
        if n_truths[batch] == 0:
            continue

        ground_truth = ground_truths[batch, :n_truths[batch]]
        for gt in ground_truth:
            gt_class = gt[0].int().data[0]
            t1 = category_map[gt_class]['ground_truth']
            if len(t1.size()) == 0:
                t1 = gt[1:].unsqueeze(0)
            else:
                t1 = torch.cat([t1, gt[1:].unsqueeze(0)], 0)
            category_map[gt_class]['ground_truth'] = t1
            
        nms_boxes = nms_output[batch]
        if len(nms_boxes.size()) == 0:
            continue

        for box in nms_boxes:
            class_id = (torch.max(box[5:], 0)[1]).int().data[0]
            t2 = category_map[class_id]['prediction']
            if len(t2.size()) == 0:
                t2 = box[:5].unsqueeze(0)
            else:
                t2 = torch.cat([t2, box[:5].unsqueeze(0)], 0)
            category_map[class_id]['prediction'] = t2
        cat_ids = category_map.keys()
#         return category_map
        mean_avg_precision += torch.mean(torch.cat([calc_map(category_map[cat_id], iou_thresh) for cat_id in cat_ids], 0 ))
    return mean_avg_precision/N

