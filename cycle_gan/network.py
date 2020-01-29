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
    if isinstance(m, nn.InstanceNorm2d):
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.02)

    return None

class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        """
            discriminator: C64-C128-C256-C512 + C1(Sigmoid)
            Ck: Convolution(kernel:4, stride:2)-BatchNorm-LeakyReLU(0.2)
            * InstanceNorm is not applied to the first 'C64' layer
        """
        layers = []
        layers.append(make_conv_module('zero', 'down', 3, 64, False, 'lrelu', 4, 2, 1))
        in_ch = 64
        for out_ch in [128, 256, 512]:
            layers.append(make_conv_module('zero', 'down', in_ch, out_ch, True, 'lrelu', 4, 2, 1))
            in_ch = out_ch
        layers.append(make_conv_module('zero', 'down', in_ch, 1, False, 'linear', 4, 2, 1))
        
        self.layers = nn.Sequential(*layers)

        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x)
 
def make_conv_module(pad, conv, in_ch, out_ch, norm, act, k=3, s=2, p=1):
    basic_module = []
    if pad == 'reflect':
        basic_module.append(nn.ReflectionPad2d(p))
    elif pad == 'zero':
        basic_module.append(nn.ZeroPad2d(p))
    elif pad == 'none':
        pass
    else:
        raise NotImplementedError("Not implemented pad type")
        
    if conv == 'up':
        basic_module.append(nn.ConvTranspose2d(in_ch, out_ch, k, s, 1, 1))
    elif conv == 'down':
        basic_module.append(nn.Conv2d(in_ch, out_ch, k, s))
    else:
        raise NotImplementedError("Not implemented conv type")
    
    if norm:
        basic_module.append(nn.InstanceNorm2d(out_ch, affine=True))

    if act == 'relu':
        basic_module.append(nn.ReLU())
    elif act == 'lrelu':
        basic_module.append(nn.LeakyReLU(0.2))
    elif act == 'tanh':
        basic_module.append(nn.Tanh())
    elif act == 'sigmoid':
        basic_module.append(nn.Sigmoid())
    elif act == 'linear':
        pass
    else:
        raise NotImplementedError("Not implemented activation type")
    
    return nn.Sequential(*basic_module)

class ResidualLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualLayer, self).__init__()

        self.conv1 = make_conv_module('reflect', 'down', in_ch, out_ch, True, 'relu', 3, 1, 1)
        self.conv2 = make_conv_module('reflect', 'down', out_ch, out_ch, True, 'linear', 3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x

class Generator(nn.Module):
    def __init__(self, r_blocks=9):
        super(Generator, self).__init__()
        layers = []

        # down sampling layers
        layers.append(make_conv_module('reflect', 'down', 3, 64, True, 'relu', 7, 1, 3))
        in_ch = 64
        for out_ch in [128, 256]:
            layers.append(make_conv_module('reflect', 'down', in_ch, out_ch, True, 'relu', 3, 2, 1))
            in_ch = out_ch

        # residual layers
        for i in range(r_blocks):
            layers.append(ResidualLayer(in_ch, in_ch))

        # up sampling layers
        for out_ch in [128, 64]:
            layers.append(make_conv_module('none', 'up', in_ch, out_ch, True, 'relu', 3, 2))
            in_ch = out_ch
        layers.append(make_conv_module('reflect', 'down', in_ch, 3, True, 'tanh', 7, 1, 3)) 

        self.layers = nn.Sequential(*layers)

        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x)
