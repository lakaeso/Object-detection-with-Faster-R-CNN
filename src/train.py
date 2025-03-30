import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import VOCDetection
from torchvision.transforms import v2

import numpy as np

import cv2

from PIL import Image, ImageDraw

from utils import *
from rpn_loss import RPNLoss
from model import *

# hiperparams
NUM_EPOCH = 2

DEVICE = torch.device('cpu') # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define transforms
def transform(x):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
    ])

    x_t = transforms(x)

    C, H, W = x_t.shape

    return transforms(x), H, W

def target_transforms(y):
        
        # class mapping
        class_mapping = {
            "aeroplane": 0,
            "bicycle": 1,
            "bird": 2,
            "boat": 3,
            "bottle": 4,
            "bus": 5,
            "car": 6,
            "cat": 7,
            "chair": 8,
            "cow": 9,
            "diningtable": 10,
            "dog": 11,
            "horse": 12,
            "motorbike": 13,
            "person": 14,
            "pottedplant": 15,
            "sheep": 16,
            "sofa": 17,
            "tvmonitor": 18,
            "train": 19,
            "NOTHING": 20,
        }

        boxes = []
        labels = []
        
        for i in range(len(y['annotation']['object'])):
            xmin = int(y['annotation']['object'][i]['bndbox']['xmin'])
            xmax = int(y['annotation']['object'][i]['bndbox']['xmax'])
            ymin = int(y['annotation']['object'][i]['bndbox']['ymin'])
            ymax = int(y['annotation']['object'][i]['bndbox']['ymax'])

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            w = abs(xmax - xmin)
            h = abs(ymax - ymin)

            # append gtb
            boxes.append(torch.Tensor([[x_center, y_center, w, h]]))

            # class
            label = y["annotation"]["object"][i]["name"]
            labels.append(class_mapping[label])

        # vstack boxes
        boxes = torch.vstack(boxes)

        # return
        return (boxes, labels)

# define dataset
dataset_train = VOCDetection('./data', year='2007', image_set='train', download=False, transform=transform, target_transform=target_transforms)
# dataset_test = VOCDetection('./data', year='2007', image_set='test', download=True)
# dataset_val = VOCDetection('./data', year='2007', image_set='val', download=True)

# define dataloader
dl_train = DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=collate_fn)

# init models
faster_rcnn = FasterRCNN().to(DEVICE)

# define criterion
criterion_rpn = RPNLoss()

# define optim
optim = torch.optim.Adam(list(faster_rcnn.feature_extractor.parameters()) + list(faster_rcnn.region_proposal_network.parameters()) , lr=0.001)

# begin single train
faster_rcnn.train_rpn(NUM_EPOCH, dl_train, optim, criterion_rpn)

# define criterion for clsn
#criterion_cls = nn.CrossEntropyLoss()

# define optim for clsn
#optim_2 = torch.optim.Adam(list(faster_rcnn.feature_extractor.parameters()) + list(faster_rcnn.roi_pooling_layer.parameters()) + list(faster_rcnn.classifier.parameters()), lr=0.001)

# begin train
#faster_rcnn.train_clsn(NUM_EPOCH, dl_train, optim_2, criterion_cls)