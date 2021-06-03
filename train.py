import argparse
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F 
import paddle.optimizer as optim 
from paddle.optimizer.lr import LambdaDecay
from models.Yolo import Yolo
from utils.paddle_utils import ModelEMA
from utils.dataset import create_dataloader
from utils.loss    import ComputeLoss
from utils.metrics import *
from tests         import test

def train(opt):
    save_dir, epochs, batch_size, img_size, multi_scale, adam, workers, save_period = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.img_size, opt.multi_scale, opt.adam, opt.workers, opt.save_period
    total_batch_size = opt.total_batch_size
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    model = Yolo(number_class=opt.number_class)

    gs = max(int(max(model.stride)), 32)
    # Freezing some parameters....
    freeze = []
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print('Freezing % s' % k)
            v.requires_grad = False
    
    # Set optimizer 
    '''
        1. define the learning rate
        2. define the optimzer
    '''
    lf       = lambda x: (1 - x / (epochs - 1)) * (1.0 - opt.lrf) + opt.lrf
    sheduler = LambdaDecay(opt.lr0, lf)

    if adam:
        optimizer = optim.Adam(learning_rate=sheduler, beta1=opt.momentum, beta2=0.999, parameters=model.parameters())
    else:
        optimizer = optim.SGD(learning_rate=sheduler, momentum=opt.momentum, parameters=model.parameters())
    
    # Set Model Exponential Moving Average....
    #ema      = ModelEMA(model)

    # Resume.....

    nbs = 64
    # Create dataloader and dataset
    dataloader, dataset = create_dataloader(path=opt.train_path, img_size=img_size, batch_size=batch_size, stride=gs)
    number_batch = len(dataloader)

    testloader, testdataset = create_dataloader(path=opt.test_path, img_size=img_size, batch_size=batch_size, stride=gs)

    best_fitness = 0.0
    # Start training
    t0          = time.time()
    number_warm = max(round(opt.warmup_epochs * number_batch), 1000)
    maps        = np.zeros(opt.number_class)
    results     = (0, 0, 0, 0, 0, 0, 0)
    compute_loss= ComputeLoss(model.detect)

    print(f"Starting training for {epochs} epochs..............\n")

    for epoch in range(epochs):
        model.train()

        mloss = paddle.zeros([4])
        pbar = enumerate(dataloader)

        for i, (imgs, targets, _) in pbar:
            ni = i + number_batch * epoch
            imgs = imgs / 255

            '''if ni < number_warm:
                xi = [0, number_warm]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                # Fucking the paddlepaddle
                # This is a holy shit deeplearning framework
                optimizer.set_lr(np.interp(ni, xi, [opt.warmup_bias_lr, optimizer.get_lr() * lf(epochs)]))'''
            
            if multi_scale:
                sz = random.randrange(img_size * 0.5, img_size * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # forward
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            # Backward
            loss.backward()
            # Step the optimizer
            optimizer.step()
            # clear the model's gradient
            model.clear_gradients()
            mloss = (mloss * i + loss_items) / (i + 1)
        print(f"Now the {epoch} epochs's mloss = {mloss}")
        print(f"Now the learning rate = {optimizer.get_lr()}")
            
        final_epoch = epoch + 1 == epochs
        sheduler.step()
        if opt.notest or final_epoch:
            results, maps = test(batch_size  =batch_size,
                                img_size     =img_size,
                                model        =model,
                                dataloader   =testloader,
                                compute_loss =compute_loss)
            # update the best mAP
            fi            = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi

        # Save model
        if epoch % opt.save_period == 0:
            model_state_dict = model.state_dict()
            optimizer_dict   = optimizer.state_dict()
            paddle.save(model_state_dict, opt.save_dir + str(epoch) + '.pdparams')
            paddle.save(model_state_dict, opt.save_dir + str(epoch) + '.pdopt')

    # End training

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./result')
    parser.add_argument('--epochs', type=int, default='300')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--multi_scale', type=bool, default=True)
    parser.add_argument('--adam', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save_period', type=int, default=10)
    parser.add_argument('--lr0', type=float, default=0.0032)
    parser.add_argument('--lrf', type=float, default=0.12)
    parser.add_argument('--momentum', type=float, default=0.843)
    parser.add_argument('--weight_decay', type=float, default=0.00036)
    parser.add_argument('--train_path', type=str, default='./data/train_list.txt')
    parser.add_argument('--test_path', type=str, default='./data/val_list.txt')
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--warmup_momentum', type=float, default=0.5)
    parser.add_argument('--warmup_bias_lr', type=float, default=0.05)
    parser.add_argument('--number_class', type=int, default=20)
    parser.add_argument('--notest', type=bool, default=False)
    opt = parser.parse_args()
    opt.total_batch_size = opt.batch_size
    train(opt)