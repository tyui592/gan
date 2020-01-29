import os
import argparse

from train import train_network
from evaluate import evaluate_network

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-no", type=int,
            help="cpu: -1, gpu: 0~N", default=0)

    parser.add_argument("--train-flag", action="store_true",
            help="train the network", default=False)
    
    parser.add_argument("--unet-flag", action="store_true",
            help="apply U-Net architecture of generator", default=False)

    parser.add_argument("--lr", type=float,
            help="learning rate", default=0.0002)
    
    parser.add_argument("--beta1", type=float,
            help="bata 1 of adam optimizer", default=0.5)

    parser.add_argument("--beta2", type=float,
            help="bata 2 of adam optimizer", default=0.999)
    
    parser.add_argument("--data-A", type=str,
            help="data path for Domain A", default=None)
    
    parser.add_argument("--data-B", type=str,
            help="data path for Domain B", default=None)
    
    parser.add_argument("--imsize", type=int,
            help="size to resize the image", default=286)
    
    parser.add_argument("--cropsize", type=int,
            help="size to crop the image", default=256)
    
    parser.add_argument("--cencrop", action='store_true',
            help="center crop flag, default: random crop", default=False)

    parser.add_argument("--epoch", type=int,
            help="number of training epochs", default=200)
    
    parser.add_argument("--check-iter", type=int,
            help="check interval iteration", default=10)
    
    parser.add_argument("--batch-size", type=int,
            help="batch size", default=8)
    
    parser.add_argument("--num-workers", type=int,
            help="num workers", default=1)    
    
    parser.add_argument("--discriminator-weight", type=float,
            help="discriminator loss weight", default=0.5)    
    
    parser.add_argument("--l1-weight", type=float,
            help="L1 lsos weight of generator", default=1.0)

    parser.add_argument("--save-path", type=str,
            help="save path", default=None)

    parser.add_argument("--check-point", type=str,
            help="check point path to load trained model", default=None)
    
    parser.add_argument("--image", type=str,
            help="test image path", default=None)

    args = parser.parse_args()

    print("-*-"*10, "arguments", "-*-"*10)
    for key, value in vars(args).items():
        print("%s: %s"%(key,value))

    if args.gpu_no >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)

    
    if args.train_flag:
        discriminator, generator = train_network(args)
    else:
        evaluate_network(args)
