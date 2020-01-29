import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import argparse

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-no", type=int,
            help="cpu: -1, gpu: 0~N", default=0)
    
    parser.add_argument("--train-flag", action="store_true",
            help="train the network", default=False)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arg = parser.parse_args()


    dataset = torchvision.datasets.IamgeFolder(root=args.data,
            transform=transforms.Compose([
                transforms.Resize(args.imsize),
                transforms.RandomCrop(args.cropsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))

    dataloader = torchvision.utils.data.DataLoader(dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)


