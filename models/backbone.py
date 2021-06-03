'''
    This file implement yolo v5's backbone.
    The normal size backbone have 30 layers.
'''
import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F 
sys.path.append("..")
from models.common import *

class CSPNet(nn.Layer):
    '''
        This class implement the yolov5s6's backbone
        [   
            [-1, 1, Focus, [64, 3]],  # 0-P1/2
            [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
            [-1, 3, C3, [128]],
            [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
            [-1, 9, C3, [256]],
            [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
            [-1, 9, C3, [512]],
            [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
            [-1, 1, SPP, [1024, [5, 9, 13]]],
            [-1, 3, C3, [1024, False]],  # 9
        ]
    '''
    def __init__(self, 
                 output_stride=(8, 16, 32),
                 channel_list=(64, 128, 256, 512, 1024),
                 depth=0.33,
                 width=0.50):
        super().__init__()
        self.output_stride = output_stride
        channel_list = [int(element * width) for element in channel_list] 
        self.focus   = Focus(3, channel_list[0], k=3)
        self.conv1   = Conv(channel_list[0], channel_list[1], k=3, s=2)
        self.cblock1 = nn.Sequential(* [ C3(channel_list[1], channel_list[1]) for i in range(int(3 * depth))])

        self.conv2   = Conv(channel_list[1], channel_list[2], k=3, s=2)                             #P3
        self.cblock2 = nn.Sequential(*[ C3(channel_list[2], channel_list[2]) for i in range(int(9 * depth))])

        self.conv3   = Conv(channel_list[2], channel_list[3], k=3, s=2)                            #P4
        self.cblock3 = nn.Sequential(*[ C3(channel_list[3], channel_list[3]) for i in range(int(9 * depth))])

        self.conv4   = Conv(channel_list[3], channel_list[4], k=3, s=2)                            #P5

        self.sppmodel= SPP(channel_list[4], channel_list[4], [5, 9, 13])
        self.cblock4 = nn.Sequential(* [ C3(channel_list[4], channel_list[4], shortcut=False) for i in range(int(3 * depth))])


    def forward(self, x):
        result = []
        x = self.focus(x)
        x = self.conv1(x)
        x = self.cblock1(x)
        
        x = self.conv2(x)
        #res4 = x
        if 8 in self.output_stride:
            result.append(x)
        x = self.cblock2(x)
        
        x = self.conv3(x)
        if 16 in self.output_stride:
            result.append(x)
        x = self.cblock3(x)

        x = self.conv4(x)
        if 32 in self.output_stride:
            result.append(x)

        x = self.sppmodel(x)
        result.append(x)

        return list(reversed(result))

        
                