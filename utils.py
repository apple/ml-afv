#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import torch

class ReplayMemory(object):
    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, samples):
        """Saves a transition."""
        for sample in samples:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = sample.cpu().detach()
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return torch.stack(samples, dim=0)

    def __len__(self):
        return len(self.memory)

def get_noise(shape):
    return torch.randn(*shape)

def exp_avg(model_old, model_new, beta=0.9):
    for p, q in zip(model_old.parameters(), model_new.parameters()):
        p.data.mul_(beta).add_((1 - beta) * q)

def fisher_sim(netD, paramsD, real, fake):
    fake = fake.detach()
    n = fake.size(0)
    mean = 0
    var = 0
    fake_d = 0
    for i in range(n):
        output = netD(fake[i:i+1])
        grad = torch.autograd.grad([output.mean()], paramsD)
        mean = mean + torch.cat([p.flatten() for p in grad])
        fake_d += output.mean()
    mean = mean / n
    fake_d = fake_d / n
    for i in range(n):
        output = netD(fake[i:i+1])
        grad = torch.autograd.grad([output.mean()], paramsD)
        var = var + (torch.cat([p.flatten() for p in grad]) - mean)**2
    var = var / n

    grad = 0
    real_d = 0
    bsize = 64
    nb = real.size(0) / bsize
    for i in range(nb):
        output = netD(real[i*bsize : (i+1)*bsize])
        real_d = real_d + output.mean()
        grad_ = torch.autograd.grad([output.mean()], paramsD)
        grad = grad + torch.cat([p.flatten() for p in grad_])
    grad = grad / nb
    real_d = real_d / nb

    return torch.exp(-10*((grad - mean)**2 / (var + 1e-12)).mean()).item(), real_d.item(), fake_d.item()
    
def dataloader_IS(netG, device, nz, batchSize, nbatches=100):
    counter = 0
    while True:
        if counter == nbatches:
            break
        z = torch.randn(batchSize, nz, 1, 1).to(device)
        with torch.no_grad():
            fake = netG(z)
        counter += 1
        fake = 0.5 * fake + 0.5
        fake[:, 0] = (fake[:, 0] - 0.485) / 0.229
        fake[:, 1] = (fake[:, 1] - 0.456) / 0.224
        fake[:, 2] = (fake[:, 2] - 0.406) / 0.225
        yield fake

