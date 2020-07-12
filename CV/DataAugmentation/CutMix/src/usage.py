from cutmix import CutMix
from utils import CutMixCrossEntropyLoss
import torch
import torchvision
import pretrainedmodels
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (Rotate, Flip, OneOf, Compose)
from torch.utils.data import Dataset, DataLoader
import argparse


def arg_parser():
    parser = argparse.ArgumentParser('ArgParser for CutMix example')
    parser.add_argument('--cifarpath', default='./data', type=str, help='File path to the cifar100 dataset')
    parser.add_argument('--epoch', default=200, type=int, help='The number of epochs')
    parser.add_argument('--model_name', default='', type=str, help='The name of the deep learning model')
    return parser.parse_args()



if __name__ == "__main__":
    args = arg_parser()

    example_train_augmentation = Compose([
        Rotate(20),
        ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR100(args.cifarpath, train=True, download=True, transform=example_train_augmentation)
    dataset = CutMix(dataset, num_class=100, beta=1.0, prob=0.5, num_mix=2)    # this is paper's original setting for cifar.


    criterion = CutMixCrossEntropyLoss(True)

    epoch = args.epoch
    model_name = args.model_name

    loader = DataLoader(dataset, shuffle=True, num_workers=4, batch_size=128)

    model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epoch):
        for input, target in loader:    # input is cutmixed image's normalized tensor and target is soft-label which made by mixing 2 or more labels.
            output = model(input)
            loss = criterion(output, target)
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
