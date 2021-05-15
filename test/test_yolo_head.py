import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
sys.path.append("..")
from models.yolo_head import *

import numpy as np

def test_yolo_ahead():
    data1  = paddle.to_tensor(np.random.randn(4, 512, 64, 64), dtype='float32')    #end_layer
    data2  = paddle.to_tensor(np.random.randn(4, 512, 64, 64), dtype='float32')    #p5
    data3  = paddle.to_tensor(np.random.randn(4, 256, 128, 128), dtype='float32')   #p4
    data4  = paddle.to_tensor(np.random.randn(4, 128, 256, 256), dtype='float32')   #p3
    inputs = [data1, data2, data3, data4]
    model  = Yolo_ahead(512)
    result = model(inputs)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)
    detect = Detect_head(anchors=(
                                [10,13, 16,30, 33,23],  # P3/8
                                [30,61, 62,45, 59,119],  # P4/16
                                [116,90, 156,198, 373,326]  # P5/32
    ), ch=[128, 256, 512])
    result2 = detect(result)
    print(result2[0].shape)
    print(result2[1][0].shape)
    print(result2[1][1].shape)
    print(result2[1][2].shape)



if __name__ == '__main__':
    test_yolo_ahead()