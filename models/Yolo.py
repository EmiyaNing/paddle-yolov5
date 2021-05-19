import paddle
import paddle.nn as nn
import paddle.nn.functional as F 
import sys
sys.path.append("..")
from models.yolo_head import *
from models.common import *
from models.backbone import *
from utils.general import *

class Yolo(nn.Layer):
    '''
        Top Level class of Yolo.
    '''
    def __init__(self, depth=0.33, width=0.5):
        super().__init__()
        self.backbone  = CSPNet(output_stride=(8, 16, 32), 
                               channel_list=(64, 128, 256, 512, 1024), 
                               depth=0.33,
                               width=0.5)
        self.yolo_head = Yolo_ahead(end_channel=1024 * width, 
                                    channel_list=[512, 256, 512, 512],
                                    width=width)
        self.detect    = Detect_head(number_class=80, 
                                     anchors=(
                                                [10,13, 16,30, 33,23],  # P3/8
                                                [30,61, 62,45, 59,119],  # P4/16
                                                [116,90, 156,198, 373,326]  # P5/32
                                            ),
                                     ch=[128, 256, 512])
        self.stride     = [8, 16, 32]
        self.nms        = NMS()

    def _descale_pred(self, p, flips, scale, img_size):
        '''
            de-scale predictions following augmented inference
            Inverse operation
        '''
        x, y, wh = p[:, :, 0]/scale, p[:, :, 1] / scale, p[:, :, 2:4] / scale
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        y = paddle.unsqueeze(y, axis=-1)
        x = paddle.unsqueeze(x, axis=-1)
        p = paddle.concat([x, y, wh, p[:, :, 4:]], axis=-1)
        return p 

    def forward_once(self, inputs, nms):
        x = self.backbone(inputs)
        x = self.yolo_head(x)
        x = self.detect(x)
        if nms:
            x = self.nms(x)
        return x


    def forward_augment(self, inputs):
        x        = inputs
        img_size = x.shape[-2:]
        s        = [1, 0.83, 0.67]
        f        = [None, 3, None]
        y        = []
        for si, fi in zip(s, f):
            xi   = scale_img(paddle.flip(x, axis=[fi]) if fi else x, si, gs=int(max(self.stride)))
            yi   = self.forward_once(xi, nms=False)[0]
            yi   = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # augmented inference, train
        return paddle.concat(y, axis=1)
    
    def forward(self, x, augment=False, nms=False):
        '''
            Use the augment forward or just forward once..
        '''
        if augment:
            return self.forward_augment(x)
        else:
            return self.forward_once(x, nms=nms)
