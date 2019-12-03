#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn.functional as F

def dloss_ls(output_real, output_fake, opt):
    return torch.mean((output_real - 1)**2) + torch.mean(output_fake**2)

def gloss_ls(output_fake, fake, fake_old, opt):
    if opt.noisestd > 0:
        epsilon = torch.empty_like(fake).normal_(0, opt.noisestd)
    else:
        epsilon = torch.zeros_like(fake)
    return opt.gamma * torch.mean((fake - fake_old.detach() + epsilon)**2) + torch.mean((output_fake - 1)**2)

def dloss_hinge(output_real, output_fake, opt):
    return torch.mean(F.relu(1 - output_real)) + torch.mean(F.relu(1 + output_fake))

def gloss_hinge(output_fake, fake, fake_old, opt):
    epsilon = torch.empty_like(fake).normal_(0, opt.noisestd)
    return opt.gamma * torch.mean((fake - fake_old.detach() + epsilon)**2) - torch.mean(output_fake)

dloss = dloss_ls
gloss = gloss_ls
