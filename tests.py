import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import paddle

from utils.dataset import create_dataloader
from utils.general import *
from utils.metrics import ap_per_class
from utils.loss    import ComputeLoss


def test(batch_size=32,
         img_size=32,
         conf_thres=0.001,
         iou_thres=0.6,
         augment=False,
         model=None,
         dataloader=None,
         compute_loss=None):
    '''
        Using this function to test the model's Mean AP....
    '''
    model.eval()
    number_class = model.number_class
    iouv         = np.linspace(0.5, 0.95, 10)
    niou         = paddle.numel(paddle.to_tensor(iouv)).numpy()
    seen         = 0


    # Define require result store variable
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss         = np.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []


    for batch_i, (img, targets, _) in enumerate(tqdm(dataloader, desc=s)):
        # normalize the img's pixel value to 0-1
        img /= 255.0
        number_batch, _, height, width = img.shape
        # forward the model
        out, train_out = model(img, augment=augment)

        # compute loss
        if compute_loss:
            compute_losses = ComputeLoss(model.detect)
            loss += compute_losses(train_out, targets)[1][:3]

        # Run NMS
        targets[:, 2:] *= paddle.to_tensor([width, height, width, height])
        out            = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
        # Statistics per image
        for si, pred in enumerate(out):
            # get each image's label
            targets= trgets.numpy()
            # Find all box, whose class id == si
            labels = targets[targets[:, 0] == si, 1:]
            num_la = len(labels)
            tcls   = labels[:, 0]
            seen   += 1
            
            # Enter the continue branch
            if len(pred) == 0:
                if num_la:
                    niou = np.zeros(0, dtype=np.bool)
                    stats.append((niou, paddle.Tensor(), paddle.Tensor(), tcls))
                continue

            predn = pred.clone().numpy()
            pred  = pred.numpy()
            scale_coords(img[si].shape[1:], predn[:, :4], _[si][0], _[si][1])
            correct = np.zeros(pred.shape[0], dtype=np.bool)
            niou    = correct

            if num_la:
                detected = []
                # Get this image's box..
                tbox     = labels[:, 1:5].numpy()
                # Reshape these boxees to original image's shape.exit()
                scale_coords(img[si].shape[1:], tbox, _[si][0], _[si][1])
                for cls in paddle.unique(tcls):
                    # Get the indice of each class's index in target and pred..
                    ti = np.reshape(np.nonzero(cls == tcls), shape=[-1])
                    pi = np.reshape(np.nonzero(cls == pred[:, 5]), shape=[-1])
                    if pi.shape[0]:
                        # caculate the ious of pred box and target box
                        # If pred box's number is n, target box's number is m
                        # The box_iou() function's result will be N * M matrix
                        # Then follow code will use np.max and np.argmax find the max value and max indice.
                        # ious should be M
                        # indice should be M
                        ious   = np.max(box_iou(predn[pi, :4], tbox[ti]), axis=1)
                        indice = np.argmax(box_iou(predn[pi, :4], tbox[ti]), axis=1)
                        for j in paddle.nonzero(paddle.to_tensor(ious > iouv[0]), as_tuple=False).numpy():
                            # Get the index of these element, whose value big than 0.5
                            # j should be a int value
                            # So use indice[j] should be a index of target.
                            d  = ti[indice[j]]
                            if d.item() not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break
            
            stats.append((correct, pred[:, 4], pred[:, 5], tcls))
                        

        # Compute statistic
        stats = [np.concatenate(x, 0) for x in zip(*stats)] 
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print result
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        model.float()
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        # Return result.
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps