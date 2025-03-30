import torch

import numpy as np

import cv2

from PIL import Image, ImageDraw

from torchvision.transforms import v2
from torchvision import tv_tensors

DEVICE = torch.device('cpu') # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw_region(img: torch.Tensor, best_region: torch.Tensor):
    cv2.imwrite('./tmp.jpg', img.detach().permute(1, 2, 0).numpy())
    im = Image.open("./tmp.jpg") 
    imd = ImageDraw.Draw(im)
    vec = best_region.detach().numpy()
    x1, y1, x2, y2 = vec[0] - vec[2] / 2, vec[1] - vec[3] / 2, vec[0] + vec[2] / 2, vec[1] + vec[3] / 2
    shape = [(x1, y1), (x2, y2)]
    imd.rectangle(shape, fill=None, width = 2) 
    im.show()

def dense_to_one_hot(n, num_classes):
    return np.eye(num_classes)[n]

def non_max_supression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.4):
    
    scores_order_desc = torch.argsort(scores.reshape(-1), descending=True)
    indices = torch.arange(boxes.shape[0])
    visited = torch.zeros_like(indices)
    keep = torch.ones_like(indices)

    for i in indices:
        if keep[i] == 1:
            visited[i] = 1
            box = boxes[scores_order_desc[i]]

            for j in indices:

                if visited[j] == 0 and keep[j] == 1:
                    box_other = boxes[scores_order_desc[j]]

                    # compute IoU
                    IoU = calculate_IoU(box, box_other)

                    # if IoU > threshold set keep to 0
                    if IoU > iou_threshold:
                        keep[j] = 0 
    
    # indices
    indices = keep == 1

    # filter
    boxes = boxes[indices]

    # return
    return boxes

def calculate_IoU(ground_truth_box: torch.Tensor, anchor: torch.Tensor) -> float:
    
    # TODO: remove
    ground_truth_box = ground_truth_box.reshape(-1)
    anchor = anchor.reshape(-1)

    ground_truth_box = ground_truth_box.detach().numpy()
    anchor = anchor.detach().numpy()

    box1 = {}
    box1["left"] = ground_truth_box[0] - ground_truth_box[2] / 2
    box1["right"] = ground_truth_box[0] + ground_truth_box[2] / 2
    box1["top"] = ground_truth_box[1] - ground_truth_box[3] / 2
    box1["bottom"] = ground_truth_box[1] + ground_truth_box[3] / 2

    box2 = {}
    box2["left"] = anchor[0] - anchor[2] / 2
    box2["right"] = anchor[0] + anchor[2] / 2
    box2["top"] = anchor[1] - anchor[3] / 2
    box2["bottom"] = anchor[1] + anchor[3] / 2

    # calc intersection area
    intersection_width = min(box1["right"], box2["right"]) - max(box1["left"], box2["left"])
    intersection_height = min(box1["bottom"], box2["bottom"]) - max(box1["top"], box2["top"])
    
    if intersection_width <= 0 or intersection_height <= 0:
        return 0
    
    # calc intersection area
    intersection_area = intersection_width * intersection_height

    # calc union area
    box1_area = (box1["right"] - box1["left"]) * (box1["bottom"] - box1["top"])
    box2_area = (box2["right"] - box2["left"]) * (box2["bottom"] - box2["top"])
    
    union_area = box1_area + box2_area - intersection_area

    # calc iou
    iou = intersection_area / union_area
    
    # return
    return iou



def get_anchors(H_in, W_in, H_out, W_out):
    len_h = H_in / H_out
    len_w = W_in / W_out

    anchors = []
    for j in range(H_out):
        tmp = []
        for i in range(W_out):
            x_1 = (i / W_out) * W_in
            x_2 = ((i+1) / W_out) * W_in
            y_1 = (j / H_out) * H_in
            y_2 = ((j+1) / H_out) * H_in
            
            x_center = (x_1 + x_2) / 2
            y_center = (y_1 + y_2) / 2

            tmp.append((x_center, y_center, 50, 50)) # TODO: 50 x 50 for now
        
        anchors.append(tmp)
    
    # to tensor
    anchors = torch.tensor(anchors)

    # reshape
    anchors = anchors.reshape(-1, 4)

    # return
    return anchors



def collate_fn(batch):

    # works only for batchsize == 1
    assert len(batch) == 1

    (x, H, W), (gtb, labels) = batch[0]

    # init transform and bboxes
    transform = v2.Compose([v2.CenterCrop((640, 640))])
    
    boxes = tv_tensors.BoundingBoxes(gtb, format="CXCYWH", canvas_size=(H, W))
    
    # perform transform
    d = transform({"image": x, "boxes": boxes})
    
    # image
    image = d["image"]
    
    # boxes
    boxes = d["boxes"]
    
    # target classes
    target_classes = torch.tensor(labels)
        
    # return
    return image, target_classes, boxes


def parameterise(region: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    predicted_param = region.clone()
    predicted_param[0] = (region[0] - anchor[0]) / anchor[2]
    predicted_param[1] = (region[1] - anchor[1]) / anchor[3]
    predicted_param[2] = torch.log((region[2] + 1e-7) / (anchor[2] + 1e-7))
    predicted_param[3] = torch.log((region[3] + 1e-7) / (anchor[3] + 1e-7))

    return predicted_param