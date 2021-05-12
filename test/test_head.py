import torch
import torch.nn as nn
import numpy as np

class Detect(nn.Module):
    stride = [8, 16, 32]  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, 
                 nc=80, 
                 anchors=([
                     [10,13, 16,30, 33,23],
                     [30,61, 62,45, 59,119],
                     [116,90, 156,198, 373,326]
                 ]), 
                 ch=([256, 360, 512]), 
                 inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.training = False
        '''
            using the 1*1 conv to get the output...
        '''
        self.m = nn.ModuleList(nn.Conv2d(in_channels=x,out_channels=self.no * self.na, kernel_size=1) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        # x's shape is grid * bs * channel * h * w
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                print( "y.shape = " + str(y.shape))
                if self.inplace:
                    print("y[..., 0:2].shape = " + str(y[..., 0:2].shape))
                    print("y[..., 2:4].shape = " + str(y[..., 2:4].shape))
                    print("self.grid[i].shape = " + str(self.grid[i].shape))
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    '''
                        In these place, use the variable's square to make the derivative have the w and h
                    '''
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

if __name__ == '__main__':
    detect = Detect()
    #x      = torch.tensor(np.random.randn(3, 3, 256, 128, 128),dtype=torch.float32)
    x      = [torch.tensor(np.random.randn(3, 256, 19, 19),dtype=torch.float32), torch.tensor(np.random.randn(3, 360, 38, 38),dtype=torch.float32), torch.tensor(np.random.randn(3, 512, 76, 76),dtype=torch.float32)]
    result = detect(x)
    print(len(result))
    print("forward finished....")
    print(result[0].shape)
    print(result[1][0].shape)
    print(result[1][1].shape)
    print(result[1][2].shape)