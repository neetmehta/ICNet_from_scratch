import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple

import torch

Label = namedtuple('Label', [
                   'name', 
                   'id', 
                   'trainId', 
                   'category', 
                   'categoryId', 
                   'hasInstances', 
                   'ignoreInEval', 
                   'color'])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    
    Label(  'caravan'              , -1 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , -1 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'polegroup'            , -1 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'guard rail'           , -1 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , -1 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , -1 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'parking'              , -1 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , -1 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'dynamic'              , -1 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  1 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  2 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  3 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             ,  4 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 ,  5 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                ,  6 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 ,  7 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        ,  8 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         ,  9 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 10 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 11 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 12 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 13 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 14 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 15 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 16 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 17 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 18 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 19 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 20 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'unlabeled'            , -1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    
]

def convert_image(pred):
    output = np.zeros((3,pred.shape[0], pred.shape[1]))
    for i in labels:
        output[0][pred==i.id] = i.color[0]
        output[1][pred==i.id] = i.color[1]
        output[2][pred==i.id] = i.color[2]
    output = np.transpose(output, (1,2,0))
    return output.astype(np.uint8)

def plot_image(pred, target, image):
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    image = image[0]
    imgplot = plt.imshow(image.permute(1,2,0))
    ax.set_title('Image')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    ax = fig.add_subplot(1, 3, 2)
    pred = torch.argmax(pred[0], dim=0)
    pred_label = convert_image(pred)
    imgplot = plt.imshow(pred_label)
    ax.set_title('Prediction')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    ax = fig.add_subplot(1, 3, 3)
    target = torch.argmax(target[0], dim=0)
    target_label = convert_image(target)
    imgplot = plt.imshow(target_label)
    ax.set_title('Target')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    return fig