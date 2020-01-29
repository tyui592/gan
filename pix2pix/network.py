import torch
import torch.nn as nn

def weights_init(m):
    """
        All networks were trained from scratch.
        Weights were inistialized from a Gaussian (mean:0, std:0.02)
    """
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.02)
    return None

class Generator(nn.Module):
    def __init__(self, unet_flag=False):
        super(Generator, self).__init__()
        self.unet_flag = unet_flag

        """
            encoder: C64-C128-C256-C512-C512-C512-C512-C512 
            Ck: Convolution(kernel:4, stride:2)-BatchNorm-LeakyReLU(0.2)
            * BatchNorm is not applied to the first 'C64' layer
        """
        # def _make_basic_module(self, conv, in_ch, out_ch, norm, drop, act):
        self.encoder = nn.ModuleList()
        self.encoder.append(make_basic_module('down', 3, 64, False, False, 'lrelu'))
        in_ch = 64
        for out_ch in [128, 256, 512, 512, 512, 512]:
            self.encoder.append(make_basic_module('down', in_ch, out_ch, True, False, 'lrelu'))
            in_ch = out_ch
        self.encoder.append(make_basic_module('down', in_ch, out_ch, False, False, 'lrelu'))
        in_ch = out_ch


        """
            decoder: CD512-CD512-CD512-C512-C256-C128-C64 + C3(Tanh)
            CD: Convolution-BatchNorm-Dropout(0.5)-LeakyReLU(0.2)
            * U-Net decoder: CD512-CD1024-CD1024-C1024-C512-C256-C128 + C3(Tanh)
        """
        self.decoder = nn.ModuleList()
        self.decoder.append(make_basic_module('up', in_ch, 512, True, True, 'relu'))
        drop_out = True
        for i, out_ch in enumerate([512, 512, 512, 256, 128, 64]):
            if unet_flag:
                in_ch *= 2
            if i == 2:
                drop_out = False
            self.decoder.append(make_basic_module('up', in_ch, out_ch, True, drop_out, 'relu'))
            in_ch = out_ch
        if unet_flag:
            in_ch *= 2
        self.decoder.append(make_basic_module('up', in_ch, 3, False, False, 'tanh'))

        self.apply(weights_init)

    def forward(self, x):
        if self.unet_flag:
            feature = []

        for layer in self.encoder:
            x = layer(x)
            if self.unet_flag:
                feature.append(x)

        for i, layer in enumerate(self.decoder):
            if self.unet_flag and i > 0:
                x = torch.cat([x, feature[-i-1]], dim=1)
            x = layer(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        """
            discriminator: C64-C128-C256-C512 + C1(Sigmoid)
            Ck: Convolution(kernel:4, stride:2)-BatchNorm-LeakyReLU(0.2)
            * BatchNorm is not applied to the first 'C64' layer
        """
        self.layer = nn.ModuleList()
        self.layer.append(make_basic_module('down', 6, 64, False, False, 'lrelu'))
        in_ch = 64
        for out_ch in [128, 256, 512]:
            self.layer.append(make_basic_module('down', in_ch, out_ch, True, False, 'lrelu'))
            in_ch = out_ch
        self.layer.append(make_basic_module('down', in_ch, 1, False, False, 'sigmoid'))

        self.apply(weights_init)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        for layer in self.layer:
            x = layer(x)
        return x

def make_basic_module(conv, in_ch, out_ch, norm, drop, act, k=4, s=2):
    basic_module = []
    if conv == 'up':
        basic_module.append(nn.ConvTranspose2d(in_ch, out_ch, k, s, 1))
    elif conv == 'down':
        basic_module.append(nn.Conv2d(in_ch, out_ch, k, s, 1))
    else:
        raise NotImplementedError("Not implemented conv type")
    
    if norm:
        basic_module.append(nn.BatchNorm2d(out_ch, affine=True))

    if drop:
        basic_module.append(nn.Dropout(0.5))


    if act == 'relu':
        basic_module.append(nn.ReLU())
    elif act == 'lrelu':
        basic_module.append(nn.LeakyReLU(0.2))
    elif act == 'tanh':
        basic_module.append(nn.Tanh())
    elif act == 'sigmoid':
        basic_module.append(nn.Sigmoid())
    else:
        raise NotImplementedError("Not implemented activation type")
    
    return nn.Sequential(*basic_module)
