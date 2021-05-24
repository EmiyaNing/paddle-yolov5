import sys
import paddle
import numpy as np
sys.path.append("..")
from utils.loss import *
from models.yolo_head import *

def test_build_targets_paddle():
    p3 = paddle.to_tensor(np.random.rand(4, 512, 7, 7, 85), dtype='float32')
    p4 = paddle.to_tensor(np.random.rand(4, 256, 14, 14, 85), dtype='float32')
    p5 = paddle.to_tensor(np.random.rand(4, 128, 28, 28, 85), dtype='float32')
    p  = [p3, p4, p5]
    targets = paddle.to_tensor(np.random.rand(88, 6), dtype='float32')
    detect = Detect_head(anchors=(
                                [10,13, 16,30, 33,23],  # P3/8
                                [30,61, 62,45, 59,119],  # P4/16
                                [116,90, 156,198, 373,326]  # P5/32
    ), ch=[128, 256, 512])
    compute_loss = ComputeLoss(det = detect)
    tcls, tbox, indices, anchors = compute_loss.build_targets(p, targets)
    print(tcls[1].shape)
    print(tbox[1].shape)
    print(indices[1][0].shape)
    print(indices[1][1].shape)
    print(indices[1][2].shape)


if __name__ == '__main__':
    test_build_targets_paddle()