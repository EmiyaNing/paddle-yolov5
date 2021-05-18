import sys
import paddle
import numpy as np
sys.path.append("..")
from utils.general import *

def test_bbox_iou():
    box1 = paddle.to_tensor([3, 3, 5, 5], dtype='float32')
    box2 = paddle.to_tensor([[1, 1, 4, 4], [2, 2, 6, 6], [3, 3, 6, 6]], dtype='float32')
    result = bbox_iou(box1, box2, x1y1x2y2=True, GIoU=True)
    print(result)

def test_box_iou():
    box1 = paddle.to_tensor([[3, 3, 5, 5], [4,4,9,9]], dtype='float32')
    box2 = paddle.to_tensor([[1, 1, 4, 4], [2, 2, 6, 6], [3, 3, 6, 6]], dtype='float32')
    result = box_iou(box1, box2)
    print(result)

def test_wh_iou():
    box1 = paddle.to_tensor([[4, 9],[5,6],[7,8]], dtype='float32')
    box2 = paddle.to_tensor([[3, 2],[9,4],[6,6],[5,5]], dtype='float32')
    result = wh_iou(box1, box2)
    print(result)

if __name__ == '__main__':
    #test_bbox_iou()
    #test_box_iou()
    test_wh_iou()

    