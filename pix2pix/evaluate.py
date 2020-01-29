import torch


from utils import imload, imsave
from network import Generator


def evaluate_network(args):
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')
    check_point = torch.load(args.check_point)

    network = Generator(args.unet_flag).to(device).eval()
    network.load_state_dict(check_point['g_state_dict'])

    image = imload(args.image, args.imsize, args.cropsize, args.cencrop).to(device)

    output = network(image)

    imsave(output, 'output.jpg')
