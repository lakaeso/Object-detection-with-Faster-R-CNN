import torch
import torch.nn as nn

import numpy as np

from utils import calculate_IoU, parameterise

from torchvision import tv_tensors

from torchvision.ops import box_iou


class RPNLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.bb = nn.SmoothL1Loss()
        self.cls = nn.BCEWithLogitsLoss()
        
        # weight between L1 and CLS loss
        self.l = 10

        # iou treshold for positive
        self.iou_tresh_positive = 0.8

        # iou treshold for negative
        self.iou_tresh_negative = 0.3

        # num of anchor boxes per anchor
        self.k = 1

    def forward(self, scores: torch.Tensor, regions: torch.Tensor, anchors: torch.Tensor, ground_truth_boxes: list):

        # calc IOU scores
        iou_scores = []
        region_scores = []
        for r_i, region in enumerate(regions):
            ious = []
            regsc = []
            for b_i, gtb in enumerate(ground_truth_boxes): 
                iou = calculate_IoU(gtb, region)
                ious.append(iou)
                if iou > self.iou_tresh_positive:
                    regsc.append(1)
                elif iou < self.iou_tresh_negative:
                    regsc.append(-1)
                else:
                    regsc.append(0)
            iou_scores.append(ious)
            region_scores.append(regsc)
        iou_scores = torch.Tensor(iou_scores)
        region_scores = torch.Tensor(region_scores)

        # find max iou by each gtb and assign 1
        values, indices = torch.max(iou_scores, dim=0)
        for i, idx in enumerate(indices):
            region_scores[idx][i] = 1
            max_IoU_idx = idx

        # get filter for positive and negative
        positive_scores_filter = torch.max(region_scores, dim=1)[0] == 1
        negative_scores_filter = torch.max(region_scores, dim=1)[0] == -1

        # create y_true vector
        cls_true = torch.hstack([
                torch.ones_like(scores[positive_scores_filter].reshape(-1)),
                torch.zeros_like(scores[negative_scores_filter].reshape(-1))
        ])

        # create y_pred vector
        cls_pred = torch.hstack([
            scores[positive_scores_filter].reshape(-1),
            scores[negative_scores_filter].reshape(-1)
        ])
        
        # calculate cls loss
        cls_loss = self.cls(cls_true, cls_pred)
        
        # calculate bbox loss
        reg_loss = None
        reg_loss_cntr = 0
        
        for i, region in enumerate(regions):
            for j, gtb in enumerate(ground_truth_boxes):
                if region_scores[i][j] == 1:

                    # parameterise region
                    region_p = parameterise(region, anchors[i])

                    # parameterise gtb
                    ground_truth_box_p = parameterise(gtb, anchors[i])

                    # add loss
                    l = self.bb(region_p, ground_truth_box_p)
                    
                    # increment counter
                    reg_loss_cntr += 1
                    if reg_loss is None:
                        reg_loss = l
                    else:
                        reg_loss += l
        
        # sum L1 and CLS loss
        loss = cls_loss / len(cls_pred) + self.l * reg_loss / reg_loss_cntr / self.k

        # return loss and max IoU idx (for debugging)
        return loss, max_IoU_idx
