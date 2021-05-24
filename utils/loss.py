import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
sys.path.append("..")

from utils.general import bbox_iou


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps

class BCEBlurWithLogitsLoss(nn.Layer):
    '''
        BCEwithLogitLoss() with reduced missing label effects.
    '''
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_function = nn.BCEwithLogitLoss(reduction='none')
        self.alpha         = alpha

    def forward(self, pred, true):
        loss = self.loss_function(pred, true)
        pred = F.sigmoid(pred)

        dx   = pred - true
        alpha_factor = 1 - paddle.exp((dx - 1) / (self.alpha + 1e-4))
        
        loss *= alpha_factor
        return paddle.mean(loss)


class FocalLoss(nn.Layer):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_function, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_function
        self.gamma    = gamma
        self.alpha    = alpha
        self.reduction= loss_function.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss         = self.loss_fcn(pred, true)

        pred_prob    = F.sigmoid(pred)
        p_t          = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss        *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return paddle.mean(loss)
        elif self.reduction == 'sum':
            return paddle.sum(loss)
        else:
            return loss

class QFocalLoss(nn.Layer):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.alpha    = alpha
        self.reduction= loss_function.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss      = self.loss_fcn(pred, true)
        preb_prob = F.sigmoid(preb)
        alpha_factor      = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = paddle.abs(true - pred_prob) ** self.gamma
        loss     *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return paddle.mean(loss)
        elif self.reduction == 'sum':
            return paddle.sum(loss)
        else:
            return loss

class ComputeLoss:
    def __init__(self, det, autobalance=False):
        super().__init__()
        BCEcls     = nn.BCEWithLogitsLoss(paddle.to_tensor([0.85]))
        BCEobj     = nn.BCEWithLogitsLoss(paddle.to_tensor([1.00]))

        self.cp, self.cn = smooth_BCE()

        g          = 1.5

        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.det          = det
        self.balance =  [4.0, 1.0, 0.4]
        self.ssi     = [8, 16, 32].index(16) if autobalance else 0
        self.anchor_t= 2.9

    def __call__(self, p, targets):
        pass


    def build_targets(self, p, targets):
        '''
            P should be         batch_size * 3 * gridx * gridy * number_class
            targets should be   num_targets * (image_index, class, x, y, h, w)
            To avoid the target's anchor be dispose all, I add a if case to break out.
            It may introduce some bug....
        '''
        na, nt = self.det.number_anchor,  targets.shape[0]
        tcls, tbox, indices, anchor = [], [], [], []
        gain   = paddle.ones([7])
        '''
            anchor = 3, there will change target's shape to 3*target
            ai is the anchor's index, to show which anchor is matched by bbox now..
        '''
        ai     = paddle.arange(na)
        ai     = paddle.unsqueeze(ai, axis=-1)
        ai     = paddle.expand(ai, shape=[ai.shape[0], nt])
        '''
            repeat the target's shape to 3 * num_targets * 6
            then concate it with anchor index...
        '''
        targets = paddle.unsqueeze(targets, axis=0)
        targets = paddle.expand(targets, shape=[na, targets.shape[1], targets.shape[2]])
        '''
            Using follow's code will add a anchor index to targets.
            This index's number is change from 0-2.
        '''
        targets = paddle.concat([targets, paddle.cast(paddle.unsqueeze(ai, axis=-1), dtype='float32')], axis=2)

        # g is the cell's center...
        g      = 0.5
        # off show the around cell...
        off    = paddle.to_tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype='float32') * g


        for i in range(self.det.number_layers):
            '''
                caculate each branch's anchor.
                In labels, the x y h w's range is (0,1), so in this process we should expand it to feature map's size.
                So we caculate the 'gain'.
            '''
            no_target = False
            anchors   = self.det.anchors[i]
            gain[2:6] = paddle.index_select(paddle.to_tensor(p[i].shape, dtype='float32'), paddle.to_tensor([3, 2, 3, 2]))
            t         = targets * gain

            if nt:
                r     = t[:, :, 4:6] / paddle.unsqueeze(anchors, axis=1)
                '''
                    Pointed out the estramely anchor box in targets.
                    To avoid those label infect the prediction result.
                '''
                j     = paddle.max(paddle.maximum(r, 1.0/r), axis=2) < self.anchor_t
                t     = paddle.to_tensor(t.numpy()[j.numpy()], dtype='float32')                    # filter
                if t.shape[0] != 0:
                    gxy     = t[:, 2:4]                                    # Get label's center point x,y. In grid
                    gxi     = paddle.index_select(gain, paddle.to_tensor([2, 3])) - gxy
                    np_gxy  = gxy.numpy()
                    np_gxi  = gxi.numpy()
                    j, k    = ((np_gxy % 1. < g) & (np_gxy > 1.)).T
                    l, m    = ((np_gxi % 1. < g) & (np_gxi < 1.)).T
                    j       = paddle.to_tensor(j, dtype='float32')
                    k       = paddle.to_tensor(k, dtype='float32')
                    l       = paddle.to_tensor(l, dtype='float32')
                    m       = paddle.to_tensor(m, dtype='float32')
                    j       = paddle.stack([paddle.ones_like(j), j, k, l, m])
                    j       = paddle.cast(j, dtype='bool')
                    t       = paddle.unsqueeze(t, axis=0)
                    t       = paddle.expand(t, shape=[5*t.shape[0], t.shape[1], t.shape[2]])
                    t       = paddle.to_tensor(t.numpy()[j.numpy()], dtype='float32')
                    zeros   = paddle.unsqueeze(paddle.zeros_like(gxy), axis=0)
                    off     = paddle.unsqueeze(off, axis=1)
                    offsets = zeros + off
                    offsets = paddle.to_tensor(offsets.numpy()[j.numpy()], dtype='float32')
                else:
                    no_target = True
                    t       = targets[0]
                    offsets = 0

            else:
                no_target = True
                t = targets[0]
                offsets = 0

            if not no_target:
                bc   = t[:, :2]
                #b, c = t.numpy().T[: , :2]
                b    = bc[:, 0]
                c    = bc[:, 1]
                bc   = paddle.t(bc)
                gxy  = t[:, 2:4]
                gwh  = t[:, 4:6]
                gij  = (gxy - offsets)
                gi, gj = paddle.t(gij)
                a    = t[:, 6]
                indices.append((b, a, paddle.clip(gj,0, gain[3]-1), paddle.clip(gi, 0, gain[2]-1)))
                tbox.append(paddle.concat((gxy - gij, gwh), axis=1))
                anchor.append(paddle.to_tensor(anchors.numpy()[a.numpy().astype(int)]))
                tcls.append(c)
            else:
                indices.append((paddle.zeros([0]), paddle.zeros([0]), paddle.zeros([0]), paddle.zeros([0])))
                tbox.append(paddle.zeros([0, 4]))
                tcls.append(paddle.zeros([0]))
                anchor.append(paddle.zeros([0, 2]))
        return tcls, tbox, indices, anchors            



        
