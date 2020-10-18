# load packages
import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid

from model import Generator, Discriminator, weights_init
from fid_score import calculate_fid_given_paths

# Set random seed for reproducibility
# manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--nz', type=int, default=100)

args = parser.parse_args()
batch_size = args.batch_size
image_size = args.image_size
num_epochs = args.num_epochs
lr = args.lr
beta1 = args.beta1
nz = args.nz

####################### CONSTANTS, VARIABLES #########################
DATAROOT = "celeba"
WORKERS = 2
NGPU = 1
REAL_LABEL = 1.
FAKE_LABEL = 0.

img_list = []
G_losses = []
D_losses = []
iters = 0
############################## FUNCTION ##############################
def save_image_list(dataset, real):
    if real:
        base_path = './img/real'
    else:
        base_path = './img/fake'
    
    dataset_path = []
    
    for i in range(len(dataset)):
        save_path =  f'{base_path}/image_{i}.png'
        dataset_path.append(save_path)
        vutils.save_image(dataset[i], save_path, normalize=True)
    
    return base_path

############################## DATALOADER ##############################
dataset = dset.ImageFolder(root=DATAROOT,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=WORKERS)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")

############################## GENERATOR ##############################
netG = Generator(NGPU).to(device)
if (device.type == 'cuda') and (NGPU > 1):
    netG = nn.DataParallel(netG, list(range(NGPU)))
netG.apply(weights_init)

############################## DISCRIMINATOR ##########################
netD = Discriminator(NGPU).to(device)
if (device.type == 'cuda') and (NGPU > 1):
    netD = nn.DataParallel(netD, list(range(NGPU)))

#################### OPTIMIZER, NOISE, LOSS FUNC #######################
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=lr, betas=(beta1,0.9))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

scheduler_d = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

criterion = nn.BCELoss()

############################## TRAINING ##############################
print(f"Using device: {device}")
print("Starting Training Loop...")
for epoch in range(num_epochs+1):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(FAKE_LABEL)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(REAL_LABEL)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1

    scheduler_d.step()
    scheduler_g.step()
    
    # Check pointing for every epoch
    torch.save(netG.state_dict(), './checkpoint/netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), './checkpoint/netD_epoch_%d.pth' % (epoch))

torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict()
            }, './checkpoint/model_final.pth')

print("Done Training!")

####################### GENERATE FAKE IMG ##########################
with torch.no_grad():
    noise = torch.randn(50, nz, 1, 1, device=device)
    fake_dataset = netG(noise).detach().cpu()
    fake_image_path_list = save_image_list(fake_dataset, False)

    # true images
    test_dataset = dset.ImageFolder(root="./celeba",
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=2)

    for i, (data, _) in enumerate(dataloader):
        real_dataset = data
        break
        

    real_image_path_list = save_image_list(real_dataset, True)

############################## EVALUATE ##############################
fid_value = calculate_fid_given_paths([real_image_path_list, fake_image_path_list],
                                                          50, 
                                                          False,
                                                          2048)

print (f"FID score: {fid_value}")