import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs: paddle.Tensor):
        return inputs * F.sigmoid(inputs)

class Conv(nn.Layer):
    # Conv2D + BatchNorm2D + SiLU
    def __init__(self,
                 c1,                # ch_in
                 c2,                # ch_out
                 k=1,               # kernel
                 s=1,               # stride
                 p=None,            # padding
                 g=1,               # groups
                 act=True):
        super().__init__()
        self.conv = nn.Conv2D(c1, c2, k, s, autopad(k, p), groups=g)
        self.bn   = nn.BatchNorm2D(c2)
        self.act  = SiLU() if act is True else None

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x)) 


class Bottleneck(nn.Layer):
    # standard bottleneck
    def __init__(self, 
                 c1,                # ch_in
                 c2,                # ch_out
                 shortcut=True,     # shortcut
                 g=1,               # groups 
                 e=0.5):            # expansion
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # resdiual or not
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Layer):
    '''
        two branch:
            1. Conv2D + BatchNorm2D + SiLU + n*Bottleneck + Conv2D
            2. Conv2D + BatchNorm2D + SiLu
        then:
            concat(branch1, branch2)-> BatchNorm2D + LeakyReLU + Conv2D + BatchNorm2D + SiLU

    '''
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_       = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2D(c1, c_, 1, 1)
        self.cv3 = nn.Conv2D(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn  = nn.BatchNorm2D(2 * c_)
        self.act = nn.LeakyReLU(0.1)
        self.m   = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(paddle.concat([y1, y2], axis=1))))


class C3(nn.Layer):
    '''
        CSP Bottleneck with 3 convolutions
    '''
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_       = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m   = nn.Sequential(*[BottleneckCSP(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(paddle.concat([self.m(self.cv1(x)), self.cv2(x)], axis=1))

class SPP(nn.Layer):
    '''
        Spatial pyramid pooling layer.
        It is very different with KaiMing He's original SPP Module...
        The KaiMing He's SPP model's output is fixed shape..
    '''
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_       = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m   = nn.Sequential(*[nn.MaxPool2D(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(paddle.concat([x] + [m(x) for m in self.m], axis=1))

class Focus(nn.Layer):
    '''
        Focus wh information into c-space
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        '''
            x(b, c, w, h)->y(b, 4c, w/2, h/2)
        '''
        return self.conv(paddle.concat([x[:, :, ::2, ::2], x[:, :, 1::2, ::2], x[:, :, ::2, 1::2], x[:, :, 1::2, 1::2]], axis=1))