import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Detect_head(nn.Layer):
    '''
        These class is represent the yolov5's head.
        It relay on the hyperparameter stride to construct the output result...
    '''
    # the stride represent the downsize time...
    # the first stride = 8, represent the first outputhead will output the map with origin_high/8
    stride = [8, 16, 32, 64]

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
        self.m             = nn.LayerList(nn.Conv2D(in_channels=x, out_channels=self.number_output * self.number_anchor, kernel_size=1) for x in ch)
        self.training      = False

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
                y[:, :, :, :, 0:2] = (y[:, :, :, :, 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                y[:, :, :, :, 2:4] = (y[:, :, :, :, 2:4] * 2.0) ** 2 * self.anchor_grid[i]
            
                z.append(paddle.reshape(y, shape=[bs, -1, self.number_output]))

        return x if self.training else (paddle.concat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = paddle.meshgrid([paddle.arange(ny), paddle.arange(nx)])
        return paddle.cast(paddle.reshape(paddle.stack((xv, yv), 2), shape=[1, 1, ny, nx, 2]), dtype='float32')
