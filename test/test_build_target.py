import torch
import numpy as np

origin_anchors = ( [10,13, 16,30, 33,23],  # P3/8
            [30,61, 62,45, 59,119],  # P4/16
            [116,90, 156,198, 373,326] )

def build_targets(p, targets):
    global origin_anchors
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    origin_anchors = torch.tensor(origin_anchors, dtype=torch.float32).view(3, -1, 2)
    # targets nx6, 其中n是batch内所有图片的label拼接而成，6的第0维度表示当前是第几张图片的label = index classid xywh
    na, nt = 3, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    # anchor = 3, 将target 变成3xtarget格式，方便计算loss
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(3):
        anchors = origin_anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  #
        # targets 的xywh 本身是归一化的尺度，需要变成特征图尺度。
        # Match targets to anchors
        # targets is N * (images, classes, x, y, h, w, anchors)
        t = targets * gain
        # targets * gain to let x*7 y*7 h*7 w*7
        # t's shape = targets's shape
        if nt:
            # Matches
            # 计算当前target的wh和anchor的wh的比值
            # 如果大于预先设置的anchor_t值，则说明当前target和anchor匹配度不高，不应该强制回归，吧target丢到。
            # 物体被丢到说明物体的尺寸非常的极端。
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < 2.91  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter
            # Offsets
            # 网络的3个附件点，不再是落在哪个网格就计算该网格的anchor，而是依靠中心点的情况。
            # 选择最近的3个网格，作为落脚点，极大的增加正样本数。
            gxy = t[:, 2:4]  # grid xy
            # 7 - 7*x, 7 - 7 * y
            gxi = gain[[2, 3]] - gxy  # inverse
            # 通过两个条件选择出最靠近的两个邻居，加上自己就是三个网格
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            # 预设5个邻居，现在从中选出3个邻居。
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            # 下面的操作从中选出最近的3个网格。
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices
        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

if __name__ == '__main__':
    p3 = torch.tensor(np.random.rand(4, 3, 7, 7, 85))
    p4 = torch.tensor(np.random.rand(4, 3, 14, 14, 85))
    p5 = torch.tensor(np.random.rand(4, 3, 28, 28, 85))
    pred = [p3, p4, p5]
    targets = torch.tensor(np.random.rand(88, 6))
    tcls, tbox, indices, anch = build_targets(pred, targets)
    print(tcls[0].dtype)
    print(tbox[0].dtype)
    print(indices[0][0].dtype)
    print(indices[0][1].dtype)
    print(indices[0][2].dtype)
    print(indices[0][3].dtype)
    print(anch[0].dtype)

