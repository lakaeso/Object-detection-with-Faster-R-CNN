import torch
import torch.nn as nn

import numpy as np

from utils import calculate_IoU, parameterise

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

        # assign scores
        anchor_scores = np.zeros((len(ground_truth_boxes), len(anchors)))

        for i, gtb in enumerate(ground_truth_boxes):

            max_IoU_score = None
            max_IoU_idx = None
            for j, region in enumerate(regions):
                
                IoU_score = calculate_IoU(gtb, region)

                if max_IoU_score is None or max_IoU_score < IoU_score:
                    max_IoU_score = IoU_score
                    max_IoU_idx = j

                if IoU_score > self.iou_tresh_positive:
                    anchor_scores[i][j] = 1
                if IoU_score < self.iou_tresh_negative:
                    anchor_scores[i][j] = -1
            
            # assign positivo to the largest IOU
            anchor_scores[i][max_IoU_idx] = 1
        
        # calc cls loss
        cls_loss = torch.tensor(0, dtype=torch.float)
        cls_loss_cntr = 0
        for i, _ in enumerate(ground_truth_boxes):
            for j, _ in enumerate(anchors):
                if anchor_scores[i][j] == 1:
                    cls_loss += self.cls(torch.tensor([1], dtype=torch.float), scores[j])
                    cls_loss_cntr += 1
                if anchor_scores[i][j] == -1:
                    cls_loss += self.cls(torch.tensor([0], dtype=torch.float), scores[j])
                    cls_loss_cntr += 1
        
        # reg loss ONLY for positive
        reg_loss = torch.tensor(0, dtype=torch.float)
        reg_loss_cntr = 0
        for i, _ in enumerate(ground_truth_boxes):
            for j, _ in enumerate(anchors):
                if anchor_scores[i][j] == 1:
                    
                    # parameterise region
                    region_p = parameterise(regions[j], anchors[i])

                    # parameterise gtb
                    ground_truth_box_p = parameterise(ground_truth_boxes[i], anchors[i])

                    # add loss
                    l = self.bb(region_p, ground_truth_box_p)
                    
                    # increment counter
                    reg_loss_cntr += 1
                    reg_loss += l
        
        # sum L1 and CLS loss
        loss = cls_loss / cls_loss_cntr + self.l * reg_loss / reg_loss_cntr / self.k

        # return loss and max IoU idx (for debugging)
        return loss, max_IoU_idx
