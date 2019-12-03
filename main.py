#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models import *
from utils import *
from losses import *
from inception_score import inception_score, FID_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset, cifar10 or cifar100')
parser.add_argument('--dataroot', type=str, default='.', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=str, default='32x32', help='the height / width of the input image to network')
parser.add_argument('--noisestd', type=float, default=0, help='standard diviation of the noise in the MCMC objective')
parser.add_argument('--gsteps', type=int, default=1, help='number of G updates')
parser.add_argument('--dsteps', type=int, default=1, help='number of D updates')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=256, help='number features for G')
parser.add_argument('--ndf', type=int, default=256, help='number of features for D')
parser.add_argument('--niter', type=int, default=10**6, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for G, default=0.0002')
parser.add_argument('--lrd', type=float, default=0.0004, help='learning rate for D, default=0.0004')
parser.add_argument('--gamma', type=float, default=0, help='regularization strength for the MCMC objective')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='save dir')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--augmentation', default=False, action='store_true', help='whether to apply data augmentation')
parser.add_argument('--monitor', default='fisher_IS', type=str, help='metrics to monitor, choice includes fisher, fid, IS, joind by _')
parser.add_argument('--printevery', default=1000, type=int, help='frequency of printing')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

imageSize = map(int, opt.imageSize.split('x'))
if opt.augmentation:
    import torchvision.transforms.functional as TF
    transform=transforms.Compose([
        transforms.RandomCrop(imageSize, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: TF.rotate(x, random.choice([-90, 0, 90, 180]))),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
else:
    transform=transforms.Compose([
        transforms.Resize(imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
if opt.dataset == 'cifar10':
    dataset_train = dset.CIFAR10(root=opt.dataroot, train=True, download=True, 
                                   transform=transform)
    dataset_val = dset.CIFAR10(root=opt.dataroot, train=False, download=True, 
                                 transform=transform)
elif opt.dataset == 'cifar100':
    dataset_train = dset.CIFAR100(root=opt.dataroot, train=True, download=True,
                           transform=transform)
    dataset_val = dset.CIFAR100(root=opt.dataroot, train=False, download=True,
                           transform=transform)
else:
    raise NotImplementedError

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize,
                                               shuffle=True, num_workers=0)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=0)

device = torch.device('cuda:0')
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

Generator = gen_dict[opt.imageSize]
Discriminator = disc_dict[opt.imageSize]

netG = Generator(nz, ngf).to(device)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

netG_avg = Generator(nz, ngf).to(device)
exp_avg(netG_avg, netG, beta=0)
nn
netD = Discriminator(ndf).to(device)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

noise_shape = (opt.batchSize, nz, 1, 1)
fixed_noise = get_noise(noise_shape).to(device)

max_g_steps = opt.gsteps
max_d_steps = opt.dsteps
# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrd, betas=(0, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0, 0.999))
optimizerD.zero_grad()
optimizerG.zero_grad()
lr_sched_d = optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.8)
lr_sched_g = optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.8)
paramsD = [p for p in netD.parameters() if p.requires_grad]
g_label = 0
nupdates = 0
for epoch in range(10**6):
    # lr_sched_d.step()
    # lr_sched_g.step()
    if nupdates >= opt.niter:
        break
    for i, data in enumerate(dataloader_train, 0):
        if data[0].size(0) < opt.batchSize:
            continue
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)

        steps = 0
        while True:
            noise = get_noise(noise_shape).to(device)
            fake = netG(noise)            

            with torch.no_grad():
                fake_old = netG_avg(noise)
            output = netD(fake)

            errG = gloss(output, fake, fake_old, opt)

            netG.zero_grad()
            errG.backward()
            optimizerG.step()
            netG.zero_grad()
            exp_avg(netG_avg, netG, beta=0.9)
            steps += 1
            nupdates += 1
            if steps == max_g_steps:
                break

        steps = 0
        while True:
            output_real = netD(real_cpu)
            D_x = output_real.mean().item()

            noise = get_noise(noise_shape).to(device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            errD = dloss(output_real, output_fake, opt)
            optimizerD.zero_grad()
            errD.backward()
            optimizerD.step()
            optimizerD.zero_grad()
            D_G_z = output_fake.mean().item()
            steps += 1
            if steps == max_d_steps:
                break

        if nupdates % opt.printevery == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            with torch.no_grad():
                fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%07d.png' % (opt.outf, nupdates),
                    normalize=True)
            with torch.no_grad():
                fake_avg = netG_avg(fixed_noise)
            vutils.save_image(fake_avg.detach(),
                    '%s/fake_avg_samples_epoch_%07d.png' % (opt.outf, nupdates),
                    normalize=True)

            delta_gz = torch.mean((fake.detach() - fake_avg.detach())**2)
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f, delta_gz %.4f'
              % (nupdates, opt.niter, i, len(dataloader_train), errD.item(), errG.item(), D_x, D_G_z, delta_gz))

            monitor_list = opt.monitor.split('_')
            fsim_train = fsim_val = fid_train = fid_val = d_train = d_val = d_fake = ISmean = ISstd = 0
            if ('fisher' in monitor_list) or ('fid' in monitor_list):
                real_train = []
                fake = []
                for i, data in enumerate(dataloader_train, 0):
                    real_train.append(data[0].to(device))
                    noise = get_noise(noise_shape).to(device)
                    with torch.no_grad():
                        fake.append(netG(noise))
                    if (i + 1) * opt.batchSize >= 512:
                        break
                real_train = torch.cat(real_train, dim=0)
                fake = torch.cat(fake, dim=0)
                real_val = []
                for i, data in enumerate(dataloader_val, 0):
                    real_val.append(data[0].to(device))
                    if (i + 1) * opt.batchSize >= 512:
                        break
                real_val = torch.cat(real_val, dim=0)
                if 'fisher' in monitor_list:
                    fsim_train, d_train, d_fake = fisher_sim(netD, paramsD, real_train, fake)
                    fsim_val, d_val, d_fake = fisher_sim(netD, paramsD, real_val, fake)
                if 'fid' in monitor_list:
                    fid_train = FID_score(real_train, fake)
                    fid_val = FID_score(real_val, fake)
            if 'IS' in monitor_list:
                ISmean, ISstd = inception_score(dataloader_IS(netG_avg, device, opt.nz, 64))
            print('============ \n fisher sim train %.4f fisher sim val %.4f inception score %.4f train fid %.4f val fid %.4f \n=================' %
                  (fsim_train, fsim_val, ISmean, fid_train, fid_val))

            # do checkpointing
            torch.save(netG_avg.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, nupdates))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, nupdates))

            if nupdates >= opt.niter:
                break


