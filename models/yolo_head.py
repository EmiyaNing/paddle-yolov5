'''
    This file implement the Detect head.
    The Detect head include a Conv head and a Operate Head.
    The Conv head used to get some sub-layer's result.
    The Operate head used to process the sub-layer's result and get output. 
'''
import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
sys.path.append('..')
from models.common import *

class Yolo_ahead(nn.Layer):
    def __init__(self, end_channel, channel_list=[512, 256, 512, 512], width=0.5):
        '''
            The channel_list represent the backbone's output layer channels.
            If the backbone output the p5(32 times), p4(16 times), p3(8 times)
        '''
        super().__init__()
        new_channel_list = [int(element*width) for element in channel_list]
        self.conv1   = nn.Conv2D(int(end_channel), int(new_channel_list[0]), kernel_size=1, stride=1)

        self.up1     = nn.UpsamplingNearest2D(scale_factor=2)
        self.cblock1 = C3(new_channel_list[0] * 2, new_channel_list[0], shortcut=False)

        self.conv2   = nn.Conv2D(new_channel_list[0], new_channel_list[1], kernel_size=1, stride=1)
        self.up2     = nn.UpsamplingNearest2D(scale_factor=2)
        self.cblock2 = C3(new_channel_list[1] * 2, new_channel_list[1], shortcut=False)

        self.conv3   = nn.Conv2D(new_channel_list[1], new_channel_list[2], kernel_size=3, stride=2, padding=1)
        self.cblock3 = C3(new_channel_list[2] * 2, new_channel_list[2], shortcut=False)

        self.conv4   = nn.Conv2D(new_channel_list[2], new_channel_list[2], kernel_size=3, stride=2, padding=1)
        self.cblock4 = C3(new_channel_list[2] * 3, new_channel_list[2] * 2, shortcut=False)


    def forward(self, inputs):
        '''
            paramaters:
                inputs: a tensor list of backbone's input.
                        the element should be [end_output, p5(32 times), p4(16 times), p3(8 times)]
        '''
        x = inputs[0]
        x = self.conv1(x)
        x = self.up1(x)
        x = paddle.concat([x, inputs[2]], axis=1)
        x = self.cblock1(x)
        temp = x

        x = self.conv2(x)
        x = self.up2(x)
        x = paddle.concat([x, inputs[3]], axis=1)
        x = self.cblock2(x)
        res1 = x
        
        x = self.conv3(x)
        x = paddle.concat([x, temp], axis=1)
        x = self.cblock3(x)
        res2 = x

        x = self.conv4(x)
        x = paddle.concat([x, inputs[1]], axis=1)
        x = self.cblock4(x)
        res3 = x

        return [res1, res2, res3]





class Detect_head(nn.Layer):
    '''
        These class is represent the yolov5's head.
        It relay on the hyperparameter stride to construct the output result...
    '''
    # the stride represent the downsize time...
    # the first stride = 8, represent the first outputhead will output the map with origin_high/8
    stride = [8, 16, 32]

    def __init__(self, 
                 number_class=80, 
                 anchors=(),
                 ch=()):
        super().__init__()
        self.number_class  = number_class
        self.number_output = number_class + 5
        self.number_layers = len(anchors)
        self.number_anchor = len(anchors[0]) // 2
        self.grid          = [paddle.to_tensor([0.0])] * self.number_layers
        a = paddle.to_tensor(anchors, dtype='float32')
        a = paddle.reshape(a, shape=[self.number_layers, -1, 2])
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', paddle.reshape(a.clone(), shape=[self.number_layers, 1, -1, 1, 1, 2]))
        self.m             = nn.LayerList(nn.Conv2D(in_channels=x, out_channels=self.number_output * self.number_anchor, kernel_size=1, bias_attr=self.initialize_biases(self.stride[xi])) for xi,x in enumerate(ch))
        self.training      = False

    def initialize_biases(self, s):
        '''
            This function will return a initialized bias parameters.
            The bias will be initialized by log function....
        '''
        data = paddle.zeros(shape=[self.number_anchor, self.number_output])
        data[:, 4] += math.log(8 / (640 / s) ** 2)
        data[:, 5:] += math.log(0.6 / (self.number_class - 0.99))
        data = paddle.reshape(data, shape=[-1])
        bias = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Assign(data)
        )
        return bias


    def forward(self, x):
        z = []
        for i in range(self.number_layers):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = paddle.reshape(x[i], shape=[bs, self.number_anchor, self.number_output, ny, nx])
            x[i] = paddle.transpose(x[i], perm=[0, 1, 3, 4, 2])

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)

                y = F.sigmoid(x[i])
                # Noramlize the result.
                y[:, :, :, :, 0:2] = (y[:, :, :, :, 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                y[:, :, :, :, 2:4] = (y[:, :, :, :, 2:4] * 2.0) ** 2 * self.anchor_grid[i]
            
                z.append(paddle.reshape(y, shape=[bs, -1, self.number_output]))

        return x if self.training else (paddle.concat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        '''
            This function used to get a grid index...
        '''
        yv, xv = paddle.meshgrid([paddle.arange(ny), paddle.arange(nx)])
        return paddle.cast(paddle.reshape(paddle.stack((xv, yv), 2), shape=[1, 1, ny, nx, 2]), dtype='float32')
