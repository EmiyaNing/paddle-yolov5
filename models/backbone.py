import paddle
import paddle.nn as nn
import paddle.nn.functional as F 

class CSPNet(nn.Layer):
    '''
        This class implement the yolov5s6's backbone
        [from, number, module, args]
            [ [ -1, 1, Focus, [ 64, 3 ] ],  # 0-P1/2
            [ -1, 1, Conv, [ 128, 3, 2 ] ],  # 1-P2/4
            [ -1, 3, C3, [ 128 ] ],
            [ -1, 1, Conv, [ 256, 3, 2 ] ],  # 3-P3/8
            [ -1, 9, C3, [ 256 ] ],
            [ -1, 1, Conv, [ 512, 3, 2 ] ],  # 5-P4/16
            [ -1, 9, C3, [ 512 ] ],
            [ -1, 1, Conv, [ 768, 3, 2 ] ],  # 7-P5/32
            [ -1, 3, C3, [ 768 ] ],
            [ -1, 1, Conv, [ 1024, 3, 2 ] ],  # 9-P6/64
            [ -1, 1, SPP, [ 1024, [ 3, 5, 7 ] ] ],
            [ -1, 3, C3, [ 1024, False ] ],  # 11
        ]
    '''
    def __init__(self, 
                 output_stride=(8, 16, 32, 64),
                 depth=0.33,
                 width=0.50):
        super().__init__()

    def forward(self, x):
        pass 
        
                