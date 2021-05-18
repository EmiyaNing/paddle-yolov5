import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

import math
import os
import random


def make_disvisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    new_size = make_disvisible(img_size, int(s))
    if new_size != img_size:
        print("-----------------Image Size couldn't be multiply by 32-----------------")
    return new_size

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:                           # no labels loaded
        return paddle.Tensor()

    labels = np.concatenate(labels, 0)              # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)           # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)    # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return paddle.to_tensor(weights)

def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    return image_weights

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[:, :, 0] = (x[:, :, 0] + x[:, :, 2]) / 2  # x center
    y[:, :, 1] = (x[:, :, 1] + x[:, :, 3]) / 2  # y center
    y[:, :, 2] = x[:, :, 2] - x[:, :, 0]  # width
    y[:, :, 3] = x[:, :, 3] - x[:, :, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[:, :, 0] = x[:, :, 0] - x[:, :, 2] / 2  # top left x
    y[:, :, 1] = x[:, :, 1] - x[:, :, 3] / 2  # top left y
    y[:, :, 2] = x[:, :, 0] + x[:, :, 2] / 2  # bottom right x
    y[:, :, 3] = x[:, :, 1] + x[:, :, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[:, :, 0] = w * (x[:, :, 0] - x[:, :, 2] / 2) + padw  # top left x
    y[:, :, 1] = h * (x[:, :, 1] - x[:, :, 3] / 2) + padh  # top left y
    y[:, :, 2] = w * (x[:, :, 0] + x[:, :, 2] / 2) + padw  # bottom right x
    y[:, :, 3] = h * (x[:, :, 1] + x[:, :, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[:, :, 0] = w * x[:, :, 0] + padw  # top left x
    y[:, :, 1] = h * x[:, :, 1] + padh  # top left y
    return y

def clip_coords(boxes, img_shape):
    boxes[:, 0] = paddle.clip(boxes[:, 0], min=0, max=img_shape[1])
    boxes[:, 1] = paddle.clip(boxes[:, 1], min=0, max=img_shape[0])
    boxes[:, 2] = paddle.clip(boxes[:, 2], min=0, max=img_shape[1])
    boxes[:, 3] = paddle.clip(boxes[:, 3], min=0, max=img_shape[0])
    
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad  = (img1_shape[1] - img0_shape[1] * gain)/2 , (img1_shape[0] - img0_shape[0] * gain) /2
    else:
        gain = ratio_pad[0][0]
        pad  = ratio_pad[1]


    list  = paddle.unbind(coords, axis=1)
    list[0] -= pad[0]
    list[2] -= pad[0]
    list[1] -= pad[1]
    list[3] -= pad[1]
    coords= paddle.stack(list, axis=1)  
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False,eps=1e-7):
    # Returns the IoU of box1 to box2. box1's shape is [4], box2's shape is [n, 4]
    # tensor box1 and tensor box2's dtype should be float32
    box2 = paddle.t(box2)
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    temp1 = paddle.cast(paddle.minimum(b2_x2, paddle.to_tensor(b1_x2)), dtype='float32').clip(min=0)
    temp2 = paddle.cast(paddle.minimum(b2_x1, paddle.to_tensor(b1_x1)), dtype='float32').clip(min=0)
    temp3 = paddle.cast(paddle.minimum(b2_y2, paddle.to_tensor(b1_y2)), dtype='float32').clip(min=0)
    temp4 = paddle.cast(paddle.minimum(b2_y1, paddle.to_tensor(b1_y1)), dtype='float32').clip(min=0)
    inter = (temp1 - temp2) * (temp3 - temp4)
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union  = w1 * h1 + w2 * h2 - inter + eps
    iou    = inter / union

    if GIoU or DIoU or CIoU:

        cw = paddle.maximum(b2_x2, paddle.to_tensor(b1_x2)) - paddle.minimum(b2_x1, paddle.to_tensor(b1_x1))
        ch = paddle.maximum(b2_y2, paddle.to_tensor(b1_y2)) - paddle.minimum(b2_y1, paddle.to_tensor(b1_y1))

        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2)/4

            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / math.pi ** 2) * paddle.pow(paddle.atan(w2 / h2) - paddle.atan(w1 / h1), 2)
                with paddle.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union)
    else:
        return iou

def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1, box2):
    """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(paddle.t(box1))
    area2 = box_area(paddle.t(box2))

    tbox1 = paddle.unsqueeze(box1, axis=1)
    tbox2 = paddle.unsqueeze(box2, axis=0)

    N     = box1.shape[0]
    M     = box2.shape[0]

    tbox1 = paddle.expand(tbox1, shape=[N, M, 4])
    tbox2 = paddle.expand(tbox2, shape=[N, M, 4])

    inter = (paddle.minimum(tbox1[:, :, 2:],tbox2[:, :, 2:]) - paddle.maximum(tbox1[:, :, :2], tbox2[:, :, :2])).clip(min=0)
    inter = paddle.prod(inter, axis=2)

    area1 = paddle.unsqueeze(area1, axis=1)
    area2 = paddle.unsqueeze(area2, axis=0)

    return inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    N   = wh1.shape[0]
    M   = wh2.shape[0]
    wh1 = paddle.unsqueeze(wh1, axis=1)
    wh2 = paddle.unsqueeze(wh2, axis=0)

    inter = paddle.minimum(wh1, wh2)

    twh1 = paddle.expand(wh1, shape=[N, M, 2])
    twh2 = paddle.expand(wh2, shape=[N, M, 2])

    inter = paddle.prod(paddle.minimum(twh1, twh2), axis=2)
    result = inter / (wh1.prod(2) + wh2.prod(2) - inter)
    return result


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,  multi_label=False, max_det=300):
    """
        Runs Non-Maximum Suppression (NMS) on inference results
            Paramaters:
                prediction: N * M * 85(xyhw + confidence + 80 class)

            Returns:
                list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        In this function, we can use the paddle.fluid.layers.multiclass_nms function.
        This function's input paramaters:
            bboxes: N(batch size) M(box number of image) 4 
            scores: N(batch size) C(number of classes) M(box number of image)
            background_label: ignore
            score_threshold: conf_thres
            nms_top_k: filter by score_threshold and keep k box
            nms_threshold: iou_thres
            nms_eta: igonre
            keep_top_k: max_det
            normalized: ether normalize..... 
    """
    n      = prediction.shape[0]
    bboxes = prediction[:, :, :4]
    bboxes = xywh2xyxy(bboxes)
    scores = prediction[:, :, 5:] * prediction[:, :, 4]
    scores = paddle.transpose(scores, shape=[0, 2, 1])
    output = paddle.fluid.multiclass_nms(bboxes, scores, score_threshold=conf_thres, nms_threshold=iou_thres, keep_top_k=max_det)
    return output

