import time

import torch
import torch.nn as nn

from network import Generator, Discriminator
from utils import FacadeFolder, imsave, lastest_arverage_value, BufferN

def train_network(args):
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')

    G_A2B = Generator().to(device).train()
    G_B2A = Generator().to(device).train()
    print("Load Generator", G_A2B)

    D_A = Discriminator().to(device).train()
    D_B = Discriminator().to(device).train()
    print("Load Discriminator", D_A)

    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    dataset = FacadeFolder(args.data_A, args.data_B, args.imsize, args.cropsize, args.cencrop)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    loss_seq = {'G_A2B':[], 'G_B2A':[], 'D_A':[], 'D_B':[], 'Cycle':[]}

    buffer_fake_A = BufferN(50)
    buffer_fake_B = BufferN(50)
    for epoch in range(args.epoch):
        """
            learning rate schedule
        """
        for iteration, (item_A, item_B) in enumerate(dataloader, 1):
            item_A, item_B = item_A.to(device), item_B.to(device)
            """
                train discriminators
            """
            with torch.no_grad():
                fake_B = G_A2B(item_A)
                fake_A = G_B2A(item_B)

            buffer_fake_B.push(fake_B.detach())
            N_fake_B_discrimination = D_B(torch.cat(buffer_fake_B.get_buffer(), dim=0))
            real_B_discrimination = D_B(item_B)

            buffer_fake_A.push(fake_A.detach())
            N_fake_A_discrimination = D_A(torch.cat(buffer_fake_A.get_buffer(), dim=0))
            real_A_discrimination = D_A(item_A)

            loss_D_B_fake = mse_criterion(N_fake_B_discrimination, torch.zeros_like(N_fake_B_discrimination).to(device))
            loss_D_B_real = mse_criterion(real_B_discrimination, torch.ones_like(real_B_discrimination).to(device))
            loss_D_B = (loss_D_B_fake + loss_D_B_real) * args.discriminator_weight
            loss_seq['D_B'].append(loss_D_B.item())

            loss_D_A_fake = mse_criterion(N_fake_A_discrimination, torch.zeros_like(N_fake_A_discrimination).to(device))
            loss_D_A_real = mse_criterion(real_A_discrimination, torch.ones_like(real_A_discrimination).to(device))
            loss_D_A = (loss_D_A_fake + loss_D_A_real) * args.discriminator_weight
            loss_seq['D_A'].append(loss_D_A.item())

            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()

            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()

            """
                train generators
            """
            fake_B = G_A2B(item_A)
            fake_A = G_B2A(item_B)
            fake_B_discrimination = D_B(fake_B)
            fake_A_discrimination = D_A(fake_A)
            B_from_fake_A = G_A2B(fake_A)
            A_from_fake_B = G_B2A(fake_B)

            loss_G_A2B_gan = mse_criterion(fake_B_discrimination, torch.ones_like(fake_B_discrimination).to(device))
            loss_G_B2A_gan = mse_criterion(fake_A_discrimination, torch.ones_like(fake_A_discrimination).to(device))
            loss_seq['G_A2B'].append(loss_G_A2B_gan.item())
            loss_seq['G_B2A'].append(loss_G_B2A_gan.item())
            
            loss_cyc = l1_criterion(B_from_fake_A, item_B) + l1_criterion(A_from_fake_B, item_A)
            loss_seq['Cycle'].append(loss_cyc.item())

            loss_G = loss_G_A2B_gan + loss_G_B2A_gan + args.cyc_weight * loss_cyc

            optimizer_G_A2B.zero_grad()
            optimizer_G_B2A.zero_grad()
            loss_G.backward()
            optimizer_G_A2B.step()
            optimizer_G_B2A.step()


            """
                check training loss
            """
            if iteration % args.check_iter == 0:
                check_str = "%s: epoch:[%d/%d]\titeration:[%d/%d]"%(time.ctime(), epoch, args.epoch, iteration, len(dataloader))
                for key, value in loss_seq.items():
                    check_str += "\t%s: %2.2f"%(key, lastest_arverage_value(value))
                print(check_str)
                imsave(torch.cat([item_A, item_B, fake_B], dim=0), args.save_path+'training_image_A2B.jpg')
                imsave(torch.cat([item_B, item_A, fake_A], dim=0), args.save_path+'training_image_B2A.jpg')

        # save networks
        torch.save({'iteration':iteration,
                    'G_A2B_state_dict':G_A2B.state_dict(),
                    'G_B2A_state_dict':G_B2A.state_dict(),
                    'D_A_state_dict':D_A.state_dict(),
                    'D_B_state_dict':D_B.state_dict(),
                    'loss_seq':loss_seq}, args.save_path+'check_point.pth')

    return None
