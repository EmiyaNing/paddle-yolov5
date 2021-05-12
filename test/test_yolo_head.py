import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
sys.path.append("..")
from models.yolo_head import *

import numpy as np

if __name__ == '__main__':
    data1 = paddle.to_tensor(np.random.randn(3, 256, 128, 128), dtype='float32')
    data2 = paddle.to_tensor(np.random.randn(3, 512, 64, 64), dtype='float32')
    data3 = paddle.to_tensor(np.random.randn(3, 768, 32, 32), dtype='float32')
    data4 = paddle.to_tensor(np.random.randn(3, 1024, 16, 16), dtype='float32')
    test_data = [data1, data2, data3, data4]
    head  = Detect_head(anchors=([
                [ 19,27,  44,40,  38,94 ],
                [ 96,68,  86,152,  180,137 ],
                [ 140,301,  303,264,  238,542 ],
                [ 436,615,  739,380,  925,792 ],
    ]), ch=[256, 512, 768, 1024])

    result = head(test_data)

    print(result[0].shape)
    print(result[1][0].shape)
    print(result[1][1].shape)
    print(result[1][2].shape)
    print(result[1][3].shape)