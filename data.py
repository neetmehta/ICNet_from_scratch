from torch.utils.data import Dataset
import os
from os.path import join as osp
from PIL import Image
import cv2
import numpy as np
from collections import namedtuple

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
    Label(  'unlabeled'            , -1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          , -1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' , -1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           , -1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               , -1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              , -1 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  1 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  2 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  3 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              , -1 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , -1 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             ,  4 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 ,  5 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                ,  6 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , -1 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , -1 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , -1 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 ,  8 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , -1 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        ,  9 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 10 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 11 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 12 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 13 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 14 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 15 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 16 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 17 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 18 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , -1 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , -1 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 19 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 20 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 21 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


class Cityscapes(Dataset):

    def __init__(self, root, set_type='train', transforms=None) -> None:
        super(Cityscapes, self).__init__()
        self.image_list = []
        self.label_list = []
        image_root = osp(root,'leftImg8bit', set_type)
        label_root = osp(root,'gtFine', set_type)
        for city in os.listdir(image_root):
            self.image_list += [osp(image_root,city,i) for i in os.listdir(osp(image_root, city))]

        for city in os.listdir(label_root):
            self.label_list += [osp(label_root,city,i) for i in os.listdir(osp(label_root, city))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        label = cv2.imread(self.label_list[index])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        target = np.zeros((22,label.shape[0],label.shape[1]))
        for obj in labels:
            if obj.id == -1:
                target[0][np.where(np.logical_and( np.logical_and(label[:,:,0]==obj.color[0], label[:,:,1]==obj.color[1]) , label[:,:,2]==obj.color[2]))] = 1
            else:    
                target[obj.id][np.logical_and(np.logical_and(label[:,:,0]==obj.color[0], label[:,:,1]==obj.color[1]), label[:,:,2]==obj.color[2])] = 1
                target[0][np.logical_and(np.logical_and(label[:,:,0]==obj.color[0], label[:,:,1]==obj.color[1]), label[:,:,2]==obj.color[2])] = 0

        return image, target, label

                
cs = Cityscapes(r"E:\Deep Learning Projects\datasets\Cityscapes")
print(cs[0])