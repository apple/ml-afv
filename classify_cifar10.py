#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import os
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='.')
parser.add_argument('--imageSize', type=str, default='32x32')
parser.add_argument('--batchSize', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--ndf', type=int, default=256)
parser.add_argument('--ngf', type=int, default=256)
parser.add_argument('--nz', type=int, default=256)
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--sample', type=int, default=1)
opt = parser.parse_args()

imageSize = map(int, opt.imageSize.split('x'))
train_set =  dset.CIFAR10(root=opt.dataroot, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.RandomCrop(imageSize, padding=4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

test_set =  dset.CIFAR10(root=opt.dataroot, train=False, download=True,
                          transform=transforms.Compose([
                              transforms.Resize(imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=True, num_workers=8)
dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=1,
                                               shuffle=True, num_workers=8)


Discriminator = disc_dict[opt.imageSize]
Generator = gen_dict[opt.imageSize]

device = torch.device("cuda:0")
netD = Discriminator(opt.ndf).to(device)
netD.load_state_dict(torch.load(opt.netD))
paramsD = [p for p in netD.parameters() if p.requires_grad]

netG = Generator(opt.nz, opt.ngf).to(device)
netG.load_state_dict(torch.load(opt.netG))

def flatten(tensor_list, sample):
    return torch.cat([t.flatten() for t in tensor_list])[::sample]

class Clf(nn.Module):
    def __init__(self, nparams, opt):
        super(Clf, self).__init__()
        self.main = nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(nparams, 10, bias=False)
        )
        self.main[-1].weight.data.zero_()

    def forward(self, x):
        return self.main(x)

print('calculating sample mean....')

noise = torch.randn(1024, opt.nz, 1, 1).to(device)
with torch.no_grad():
    fake = netG(noise)
nexamples = 0
mean = 0
for i in range(fake.size(0)):
    nexamples += 1
    data = fake[i:i+1]
    output = netD(data)
    grad = torch.autograd.grad([output.mean()], paramsD)
    grad = flatten(grad, opt.sample)
    mean = mean + grad

mean = mean / nexamples
print('done! mean mean %.6f' % mean.abs().mean().item())

print('calculating sample var....')
nexamples = 0
var = 0
for i in range(fake.size(0)):
    nexamples += 1
    data = fake[i:i+1]
    output = netD(data)
    grad = torch.autograd.grad([output.mean()], paramsD)
    grad = flatten(grad, opt.sample)
    var = var + (grad - mean)**2

print('done! mean std %.6f' % var.sqrt().mean().item())
var = var / nexamples

nparams = mean.size(0)
print('num parameters %d' % nparams)

def get_feat(data, netD, mean, var):
    output = netD(data)
    grad = torch.autograd.grad([output.mean()], paramsD)
    grad = flatten(grad, opt.sample)
    feat = (grad - mean ) / (var.sqrt() + 1e-6)
    return feat

clf = Clf(nparams, opt).to(device)
optimizer = optim.Adam(clf.parameters(), lr=opt.lr, weight_decay=0)

def criterion(pred, labels):
    eye = torch.eye(10).to(device)
    labels_onehot = eye[labels]
    labels_signed = 2 * (labels_onehot - 0.5)
    return torch.mean(F.relu(1 - labels_signed * pred)**2)
        
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], 0.5)
print('Starting training....')
for epoch in range(200):
    lr_sched.step()
    avg_loss = 0
    avg_acc = 0
    images = []
    labels = []
    nexamples = 0
    clf.train()
    for data in dataloader_train:
        images.append(get_feat(data[0].to(device), netD, mean, var))
        labels.append(data[1].to(device))
        if len(images) == opt.batchSize:
            images = torch.stack(images, dim=0)
            labels = torch.cat(labels, dim=0)
            pred = clf(images)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nexamples += opt.batchSize
            acc = (labels == pred.max(dim=1)[1]).float().mean()
            avg_acc += acc
            images = []
            labels = []
    print('epoch %d train loss %.4f acc %.4f' % (epoch, loss.item(),
                                                 avg_acc.item() * opt.batchSize / nexamples))
    avg_acc = 0
    images = []
    labels = []
    nexamples = 0
    clf.eval()
    for data in dataloader_test:
        images.append(get_feat(data[0].to(device), netD, mean, var))
        labels.append(data[1].to(device))
        if len(images) == opt.batchSize:
            images = torch.stack(images, dim=0)
            labels = torch.cat(labels, dim=0)
            pred = clf(images)
            loss = criterion(pred, labels)
            nexamples += images.size(0)
            avg_acc += (labels == pred.max(dim=1)[1]).sum().float().item()
            images = []
            labels = []
    print('epoch %d test loss %.4f acc %.4f\n' % (epoch, loss.item(), avg_acc / nexamples))
    
