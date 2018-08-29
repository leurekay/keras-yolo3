#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:19:41 2018

@author: ly
"""


import numpy as np

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from train import get_anchors,get_classes



annotation_path = 'train.txt'
log_dir = 'logs/000/'
classes_path = 'model_data/coco_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

input_shape = (416,416) # multiple of 32, hw

is_tiny_version = len(anchors)==6 # default setting
#if is_tiny_version:
#    model = create_tiny_model(input_shape, anchors, num_classes,
#        freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
#else:
#    model = create_model(input_shape, anchors, num_classes,
#        freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze
#
#logging = TensorBoard(log_dir=log_dir)
#checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
#    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

val_split = 0.1
with open(annotation_path) as f:
    annotation_lines = f.readlines()
    
    
    
image, box = get_random_data(annotation_lines[0], input_shape, random=False)   
box_data=[]
image_data=[]
for i in range(7):
    image, box = get_random_data(annotation_lines[i], input_shape, random=False) 
    box_data.append(box)
    image_data.append(image)

box_data=np.array(box_data) 
image_data=np.array(image_data)
y_true=preprocess_true_boxes(box_data,input_shape,anchors,num_classes)

y0=y_true[0]
ind0=np.argwhere(y0>0)

y1=y_true[1]
ind1=np.argwhere(y1>0)

y2=y_true[2]
ind2=np.argwhere(y2>0)


row,col=ind0.shape
for i in range(row):
    val=y0[ind0[i,0],ind0[i,1],ind0[i,2],ind0[i,3],ind0[i,4]]
    print (val)
    
    
    
    
num_layers = len(anchors)//3 # default setting
anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
true_boxes=box_data
true_boxes = np.array(true_boxes, dtype='float32')
input_shape = np.array(input_shape, dtype='int32')
boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

m = true_boxes.shape[0]
grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
    dtype='float32') for l in range(num_layers)]
    
    
# Expand dim to apply broadcasting.
anchors = np.expand_dims(anchors, 0)
anchor_maxes = anchors / 2.
anchor_mins = -anchor_maxes
valid_mask = boxes_wh[..., 0]>0

for b in range(m):
    # Discard zero rows.
    wh = boxes_wh[b, valid_mask[b]]
    if len(wh)==0: continue
    # Expand dim to apply broadcasting.
    wh = np.expand_dims(wh, -2)
    box_maxes = wh / 2.
    box_mins = -box_maxes
    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    
    # Find best anchor for each true box
    best_anchor = np.argmax(iou, axis=-1)
