import os
import sys
import cv2
import paddle
import paddle.nn.functional as F
import numpy as np
sys.path.append("..")

from utils.general import *

from paddle.io import Dataset
from paddle.io import DataLoader

from xml.dom.minidom import parse

def create_dataloader(path, img_size, batch_size, stride, word_size=1,augment=False, workers=8):
    dataset    = VocDataset(path, img_size, batch_size, stride, augment)
    batch_size = min(batch_size, len(dataset))
    nw         = min([os.cpu_count() // word_size, batch_size if batch_size > 1 else 0, workers])
    dataloader     = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=nw,
                                collate_fn=VocDataset.collate_fn)
    return dataloader, dataset




class VocDataset(Dataset):
    def __init__(self, path, img_size=224, batch_size=16, stride=32, augment=False):
        '''
            This class is a mapping class.
            Paramaters:
                path        train/test/val list file name, it's fix should be .txt
                img_size    Input image's shape
                batch_size  Batch's size.....
        '''
        self.img_size      = img_size
        self.augment       = augment
        self.mosaic        = self.augment 
        self.mosaic_border = [- img_size // 2, - img_size // 2]
        self.stride        = stride
        self.path          = path
        self.batch_size    = batch_size
        '''
            传入的path是一个string变量，变量里存放着数据集的描述文件的文件路径。
            接下来的代码将读取该描述文件，将文件中存放的图片路径，label描述文件路径读取出来。
        '''
        try:
            image_path = []
            label_path = []
            with open(path, 'r') as t:
                lines = t.readlines()
                # train set's size...
                self.number= len(lines)
                for line in lines:
                    image, label = line.split(' ')
                    # the label will contain a '\n' char, use follow's code to remove it.
                    label        = label.split('\n')[0]
                    image_path.append(image)
                    label_path.append(label)
        except Exception as e:
            raise Exception("There occur an error, when the code reading the describe file" + str(e))
        self.image_path = image_path
        self.label_path = label_path
        self.indices    = range(self.number)


    def __getitem__(self, index):
        '''
            Load image and labels by index.
            Return paddle.to_tensor(image), paddle.to_tensor(labels), shapes
        '''
        mosaic = self.mosaic and random.random() < 1.0       
        print(mosaic)
        if mosaic:
            img, labels = self.load_mosaic(index)
            shapes      = None

            if random.random() < 0.243:
                img2, labels2 = self.load_mosaic(random.randint(0, self.number - 1))
                r             = np.random.beta(8.0, 8.0)
                img           = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels        = np.concatenate((labels, labels2), axis=0)
        else:
            img, (h, w)        = self.load_image(index)
            shape              = self.img_size
            img, ratio, pad    = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes             = (self.img_size, self.img_size), ((h / self.img_size, w / self.img_size), pad)
            labels             = self.load_label(index)
            if labels.size:
                labels[:, 1:]      = xyxy2xywh(labels[:, 1:])
                labels[:, 1:]      = xywhn2xyxy(labels[:, 1:], ratio[0] * img.shape[0], ratio[1] * img.shape[1], padw=pad[0], padh=pad[1])

        

        if self.augment:
            if not mosaic:
                img, labels = random_perspective(img, labels, degrees=0.373, translate=0.245, shear=0.602, perspective=0.0)
                self.augment_hsv(img, hgain=0.0138, sgain=0.664, vgain=0.464)

        nL  = len(labels)
        if nL:

            labels          = labels.astype(np.float32)
            labels[:, 1:5]  = xyxy2xywh(labels[:, 1:5])
            labels[:,[2,4]] = labels[:,[2,4]] / img.shape[0]
            labels[:,[1,3]] = labels[:,[2,4]] / img.shape[1]

        if self.augment:
            if random.random() < 0.00856:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
            
            if random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = paddle.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = paddle.to_tensor(labels)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return paddle.to_tensor(img), labels_out, shapes

    def __len__(self):
        return len(self.image_path)

    def load_image(self, index):
        '''
            This function use the index to read the image file.
            Return the resize image, and the image's origin high, width.
            The return image's data type = np.uint8
        '''
        image_name = self.image_path[index].split("./")[1]
        image_name = os.path.join('../data',image_name)
        image = cv2.imread(image_name)
        h,w   = image.shape[0], image.shape[1]
        assert image is not None, 'Image Not Found' + image_name
        ratio = self.img_size / max(h,w)
        if ratio != 1:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA if ratio < 1 and not self.augment else cv2.INTER_LINEAR)
        return image, (h, w)

    def load_label(self, index):
        '''
            This function use the index to read the xml file.
            Return a 2 dimension tensor, whose shape is n * [index, class, x1, y1, x2, y2]
            Return type is numpy.int64
        '''
        label_name = self.label_path[index].split("./")[1]
        label_name = os.path.join('../data',label_name)
        label_file = parse(label_name)
        collection = label_file.documentElement
        objects    = collection.getElementsByTagName("object")
        sizes      = collection.getElementsByTagName("size")
        h          = int(sizes[0].getElementsByTagName("height")[0].childNodes[0].data)
        w          = int(sizes[0].getElementsByTagName("width")[0].childNodes[0].data)

        list       = []
        for object in objects:
            name = object.getElementsByTagName("name")[0].childNodes[0].data 
            x1   = float(object.getElementsByTagName("xmin")[0].childNodes[0].data) / w
            y1   = float(object.getElementsByTagName("ymin")[0].childNodes[0].data) / h
            x2   = float(object.getElementsByTagName("xmax")[0].childNodes[0].data) / w
            y2   = float(object.getElementsByTagName("ymax")[0].childNodes[0].data) / h
            temp = np.array([index, x1, y1, x2, y2], dtype='float32')
            list.append(temp)
        return np.array(list)


    def augment_hsv(self, img, hgain=0.5, sgain=0.5, vgain=0.5):
        '''
            Augment the image in image's hsv format.
        '''
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype   = img.dtype
        x       = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


    def hist_equalize(self, image, clahe=True, bgr=False):
        '''
            Equalize histogram on BGR image with img.shape(n, m, 3) and range 0-255
        '''
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)

        if clahe:
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)


    def load_mosaic(self, index):
        labels4 = []
        s       = self.img_size
        yc, xc  = [int(random.uniform(-x, 2*s + x)) for x in self.mosaic_border]
        indices = [index] + random.choices(self.indices, k=3)
        for i,index in enumerate(indices):
            img, original_shape = self.load_image(index)
            h, w = img.shape[:2]

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            labels = self.load_label(index)
            labels[:, 1:] = xyxy2xywh(labels[:, 1:])
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)
        
        labels4 = np.concatenate(labels4, axis=0)
        labels4[:, 1:] = np.clip(labels4[:, 1:], 0, 2*s)
        return img4, labels4
            

    def load_mosaic9(self,index):
        labels9 = []
        s       = self.img_size
        indices = [index] + random.choices(self.indices, k=8)
        for i, index in enumerate(indices):
            img, original_shape = self.load_image(index)
            h, w = img.shape[:2]
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp
            
            padx, pady = c[:2]
            x1, y1, x2, y2  = [max(x,0) for x in c]
            labels          = self.load_label(index)
            labels[:, 1:]   = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)
            labels9.append(labels)

            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]
            hp, wp          = h, w
        yc, xc  = [int(random.uniform(0, s)) for _ in self.mosaic_border]
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        labels9 = np.concat(labels9, axis=0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        return img9, labels9
    
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return paddle.stack(img, 0), paddle.concat(label, 0), path, shapes


def letterbox(img, new_shape=(224, 224), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    '''
        This function is used to Resize and pad image while meeting stride-multiple constraints..
    '''
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r     = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh    = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio  = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img         = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2
    width  = img.shape[1] + border[1] * 2

    # Used to get the center 
    C      = np.eye(3)
    C[0, 2]= -img.shape[1] / 2
    C[1, 2]= -img.shape[0] / 2

    # Perspective
    P      = np.eye(3)
    P[2, 0]= random.uniform(-perspective, perspective)
    P[2, 1]= random.uniform(-perspective, perspective)

    # This matrix used to rotation and scale
    R      = np.eye(3)
    a      = random.uniform(-degrees, degrees)
    s      = random.uniform(1 - scale, 1 + scale)
    R[:2]  = cv2.getRotationMatrix2D(angle=a, center=(0,0), scale=s)

    # Shear
    S      = np.eye(3)
    S[0, 1]= math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0]= math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Translation
    T      = np.eye(3)
    T[0, 2]= random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2]= random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined rotation matrix
    # @ is the matrix dot operation
    M      = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    n      = len(targets)
    if n:
        new = np.zeros((n, 4))

        xy  = np.ones((n * 4, 3))
        # Using follow code to get the x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy  = xy @ M.T
        xy  = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

        # Get new boxes
        x   = xy[:, [0, 2, 4, 6]]
        y   = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        i  = box_candidates(box1 = targets[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

