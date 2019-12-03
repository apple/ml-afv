#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvD(nn.Module):
    def __init__(self, inchan, outchan, kernel, stride, pad, bias=True):
        super(ConvD, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(inchan, outchan, kernel, stride, pad, bias=bias))

    def forward(self, x):
        return self.conv(x)

class ResBlockGpre(nn.Module):
    def __init__(self, nin, nout=None):
        if nout is None:
            nout = nin
        super(ResBlockGpre, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(nin),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.ReLU(True),
            nn.Conv2d(nout, nout, 3, 1, 1),            
        )
        self.shortcut = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(nin, nout, 3, 1, 1)
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
        
    
class ResBlockDpre(nn.Module):
    def __init__(self, ndf, down=False, first=False):
        super(ResBlockDpre, self).__init__()
        if down:
            pool = nn.AvgPool2d(2)
        else:
            pool = nn.Sequential()
        if first:
            first_relu = nn.Sequential()
            nin = 3
        else:
            first_relu = nn.ReLU(True)
            nin = ndf
        self.conv = nn.Sequential(
            first_relu,
            ConvD(nin, ndf, 3, 1, 1),
            nn.ReLU(True),
            ConvD(ndf, ndf, 3, 1, 1),
            pool
        )
        self.shortcut = nn.Sequential(
            ConvD(nin, ndf, 3, 1, 1),
            pool
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class Generator32(nn.Module):
    def __init__(self, nz=128, ngf=128):
        super(Generator32, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf, 4, 1, 0),
            ResBlockGpre(ngf),
            ResBlockGpre(ngf),
            ResBlockGpre(ngf),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator32(nn.Module):
    def __init__(self, ndf=128):
        super(Discriminator32, self).__init__()
        self.main = nn.Sequential(
            ResBlockDpre(ndf, down=True, first=True),
            ResBlockDpre(ndf, down=True, first=False),
            ResBlockDpre(ndf, down=False, first=False),
            ResBlockDpre(ndf, down=False, first=False),
            nn.ReLU(True),
            nn.AvgPool2d(8),
            ConvD(ndf, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(x.size(0), -1)

    def get_pool(self, x):
        outs = []
        for block in self.main[:4]:
            x = block(x)
            outs.append(x.mean(dim=-1).mean(dim=-1))
        return torch.cat(outs, dim=1)

disc_dict = {'32x32': Discriminator32}
gen_dict = {'32x32': Generator32}
