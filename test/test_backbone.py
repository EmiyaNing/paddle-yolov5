import sys
import paddle
import numpy as np
sys.path.append("..")

from models.backbone import *
from models.yolo_head import *

def test_backbone():
    data   = paddle.to_tensor(np.random.randn(4, 3, 224, 224), dtype='float32')
    model  = CSPNet()
    result = model(data)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)
    print(result[3].shape)
    ahead  = Yolo_ahead(512)
    head_res = ahead(result)
    print(head_res[0].shape)
    print(head_res[1].shape)
    print(head_res[2].shape)
    detect_head = Detect_head(anchors=(
                                [10,13, 16,30, 33,23],  # P3/8
                                [30,61, 62,45, 59,119],  # P4/16
                                [116,90, 156,198, 373,326]  # P5/32
    ), ch=[128, 256, 512])
    final  = detect_head(head_res)
    print(final[0].shape)
    print(final[1][0].shape)
    print(final[1][1].shape)
    print(final[1][2].shape)


if __name__ == '__main__':
    test_backbone()