import paddle
import paddle.nn as nn
import paddle.nn.functional as F 
import numpy as np
from copy import deepcopy

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA(object):
    '''
        Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
        Keep a moving average of everything in the model state_dict (parameters and buffers).
        This is intended to allow functionality like
        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        A smoothed version of the weights is necessary for some training schemes to perform well.
        This class is sensitive where it is initialized in the sequence of model init,
        GPU assignment and distributed training wrappers.
    '''
    def __init__(self, model ,decay=0.9999, updates=0):
        model.eval()
        self.ema     = deepcopy(model)
        self.updates = updates
        self.decay   = lambda x: decay * (1 - math.exp(-x / 2000))

        for k,p in self.ema.named_parameters():
            p.requires_grad = False

    def updates(self, model):
        with paddle.no_grad():
            self.updates += 1
            d            =  self.decay(self.updates)

            msd          = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype == 'float32':
                    v *= d
                    v += (1. - d) * msd[k]
    
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)
