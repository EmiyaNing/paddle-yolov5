import sys
import paddle
import numpy as np 
sys.path.append("..")
from models.common import *


def test_focus():
    data   = paddle.to_tensor(np.random.randn(4, 3, 128, 128), dtype='float32')
    focus  = Focus(3, 64)
    result = focus(data)
    print(result.shape)

def test_SPP():
    data   = paddle.to_tensor(np.random.randn(4, 128, 64, 64), dtype='float32')
    spp    = SPP(128, 256)
    result = spp(data)
    print(result.shape)


def test_bottleneckcsp():
    data   = paddle.to_tensor(np.random.randn(4, 256, 32, 32), dtype='float32')
    model  = BottleneckCSP(256, 512, n=3)
    result = model(data)
    print(result.shape)

def test_bottleneck():
    data   = paddle.to_tensor(np.random.randn(4, 256, 32, 32), dtype='float32')
    model  = Bottleneck(256, 512)
    result = model(data)
    print(result.shape)

if __name__ == '__main__':
    #test_focus()
    #test_SPP()
    #test_bottleneckcsp()
    test_bottleneck()
    