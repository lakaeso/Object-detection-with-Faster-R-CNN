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

from utils import get_anchors, non_max_supression, calculate_IoU, draw_region


class FeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bnorm1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.bnorm2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor):
        
        x = x.reshape((1, *x.shape))

        x = self.bnorm1(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        x = self.bnorm2(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = x.squeeze(0)

        return x



class RegionProposalNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # smaller, n x n network
        self.sliding_window = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # box regression layer
        self.box_layer = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1, stride=1)

        # cls layer
        self.cls_layer = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)

        # no of anchor boxes
        k = 1

    def forward(self, x):

        # sliding window output
        y = self.sliding_window(x)

        # scores
        scores = self.cls_layer(y)
        scores = torch.sigmoid(scores)

        # coordinates
        coordinates = self.box_layer(y)

        # permute
        scores = scores.permute(1, 2, 0)
        coordinates = coordinates.permute(1, 2, 0)

        # reshape
        scores = scores.reshape(-1, 1)
        coordinates = coordinates.reshape(-1, 4)

        # return
        return scores, coordinates
    
    def propose_regions(self, region_scores: torch.Tensor, region_bounds: torch.Tensor):

        # NOTE: 0.9 hardcoded
        indices = region_scores > 0.9

        indices = indices.reshape(-1)

        region_bounds = region_bounds[indices]

        region_scores = region_scores[indices]

        return region_bounds, region_scores



class ROIPoolingLayer(nn.Module):
    
    def perform_nms(self):
        ...


class Classifier(nn.Module):
    ...



class FasterRCNN(nn.Module):
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.feature_extractor = FeatureExtractor()

        self.region_proposal_network = RegionProposalNetwork()

        self.roi_pooling_layer = ROIPoolingLayer()

        self.classifier = Classifier()
    
    # region proposal network train loop
    def train_rpn(self, num_epoch, dl_train, optim, criterion):
    
        for i_epoch in range(num_epoch):

            self.feature_extractor.train()
            self.region_proposal_network.train()

            for i_batch, (x, target_classes, ground_truth_boxes) in enumerate(dl_train):

                # NOTE: tmp, for speed
                if i_batch == 10:
                    break
                
                # zero out grads
                optim.zero_grad()

                # extract features
                feature_map = self.feature_extractor(x)

                # get region scores, offsets and dimensions
                region_scores, region_offsets = self.region_proposal_network(feature_map)

                # get dims of input and output
                _, H_in, W_in = x.shape
                _, H_out, W_out = feature_map.shape

                # get anchors
                anchors = get_anchors(H_in, W_in, H_out, W_out)

                # add region offsets to anchors
                offseted_regions = anchors + region_offsets.cpu()

                # relu the regions
                offseted_regions = torch.relu(offseted_regions)

                # offset regions score
                loss, max_iou_idx = criterion(region_scores.cpu(), offseted_regions, anchors.cpu(), ground_truth_boxes)

                # loss print
                if i_batch % 10 == 0:
                    print(f"minibatch {i_batch}, loss={loss.item()}")
                    #draw_region(x, offseted_regions[max_iou_idx])

                # bwd pass
                loss.backward()

                # optim step
                optim.step()

    def train_clsn(self, num_epoch, dl_train, optim, criterion):
        for i_epoch in range(num_epoch):

            self.feature_extractor.eval()
            self.region_proposal_network.eval()

            for i_batch, (x, target_classes, ground_truth_boxes) in enumerate(dl_train):

                # NOTE: tmp, for speed
                if i_batch == 10:
                    break
                
                # zero out grads
                optim.zero_grad()

                # extract features
                with torch.no_grad():
                    feature_map = self.feature_extractor(x)

                    # get region scores, offsets and dimensions
                    region_scores, region_offsets = self.region_proposal_network(feature_map)

                    # get dims of input and output
                    _, H_in, W_in = x.shape
                    _, H_out, W_out = feature_map.shape

                    # get anchors
                    anchors = get_anchors(H_in, W_in, H_out, W_out)

                    # add region offsets to anchors
                    offseted_regions = anchors + region_offsets.cpu()

                    # relu the regions
                    offseted_regions = torch.relu(offseted_regions)
                
                # propose regions
                proposed_regions = self.region_proposal_network.propose_regions(region_scores, offseted_regions)

                # pool regions
                self.roi_pooling_layer.perform_nms(proposed_regions)

                ...