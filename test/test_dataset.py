import paddle
import sys
sys.path.append("..")

from utils.dataset import *


def test_dataset():
    dataset = VocDataset("../data/train_list.txt")
    for img, labels, shape in dataset:
        print(labels)

if __name__ == '__main__':
    test_dataset()
