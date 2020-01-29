import time

import torch
import torch.nn as nn

from network import Generator, Discriminator
from utils import FacadeFolder, imsave, lastest_arverage_value

def train_network(args):
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')

    generator = Generator(unet_flag=args.unet_flag).to(device).train()
    discriminator = Discriminator().to(device).train()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    bce_criterion = nn.BCELoss()
    l1_criterion = nn.L1Loss()

    dataset = FacadeFolder(args.data_A, args.data_B, args.imsize, args.cropsize, args.cencrop)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    loss_seq = {'d_real':[], 'd_fake':[], 'g_gan':[], 'g_l1':[]}
    for epoch in range(args.epoch):
        for iteration, (item_A, item_B) in enumerate(dataloader, 1):
            item_A, item_B = item_A.to(device), item_B.to(device)

            """
                Train Discriminator
            """
            fake_B = generator(item_A)
            fake_discrimination = discriminator(fake_B.detach(), item_A)
            fake_loss = bce_criterion(fake_discrimination, torch.zeros_like(fake_discrimination).to(device))

            real_discrimination = discriminator(item_B, item_A)
            real_loss = bce_criterion(real_discrimination, torch.ones_like(real_discrimination).to(device))

            discriminator_loss = (real_loss + fake_loss) * args.discriminator_weight
            loss_seq['d_real'].append(real_loss.item())
            loss_seq['d_fake'].append(fake_loss.item())
            
            optimizer_d.zero_grad()
            discriminator_loss.backward()
            optimizer_d.step()

            """
                Train Generator
            """
            fake_B = generator(item_A)
            fake_discrimination = discriminator(fake_B, item_A)
            fake_loss = bce_criterion(fake_discrimination, torch.ones_like(fake_discrimination).to(device))
            l1_loss = l1_criterion(fake_B, item_B)

            generator_loss = fake_loss + l1_loss * args.l1_weight
            loss_seq['g_gan'].append(fake_loss.item())
            loss_seq['g_l1'].append(l1_loss.item())

            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()

            """
                Check training loss
            """
            if iteration % args.check_iter == 0:
                check_str = "%s: epoch:[%d/%d]\titeration:[%d/%d]"%(time.ctime(), epoch, args.epoch, iteration, len(dataloader))
                for key, value in loss_seq.items():
                    check_str += "\t%s: %2.2f"%(key, lastest_arverage_value(value))
                print(check_str)
                imsave(torch.cat([item_A, item_B, fake_B], dim=0), args.save_path+'training_image.jpg')

        # save check point
        torch.save({'iteration':iteration,
                    'g_state_dict':generator.state_dict(),
                    'd_state_dict':discriminator.state_dict(),
                    'loss_seq':loss_seq}, args.save_path+'check_point.pth')

    return discriminator, generator
