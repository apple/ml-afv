#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
from inception_utils import numpy_calculate_frechet_distance, WrapInception

import numpy as np
from scipy.stats import entropy

def inception_score(dataloader, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    # Load inception model
    device = torch.device('cuda:0')
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    def get_pred(x):
        x = F.interpolate(x, size=(299, 299), mode='nearest')
        with torch.no_grad():
            x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = []

    for i, batch in enumerate(dataloader, 0):
        if type(batch) in [list, tuple]:
            batch = batch[0]
        batchv = batch.to(device)
        batch_size_i = batch.size(0)

        preds.append(get_pred(batchv))

    preds = np.concatenate(preds, axis=0)
    N = len(preds)
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def FID_score(real, fake, batchSize=64):
    """Computes FID score
    """
    # Load inception model
    device = torch.device('cuda:0')
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_pool = WrapInception(inception_model).to(device)
    inception_pool.eval()
    def get_pred(inputs):
        preds = []
        nbatches = inputs.size(0) / batchSize
        for i in range(nbatches):
            x = inputs[i * batchSize : (i + 1) * batchSize]
            x = F.interpolate(x, size=(299, 299), mode='nearest')
            with torch.no_grad():
                x = inception_pool(x)[0]
            preds.append(x)
        return torch.cat(preds, dim=0)

    # Get predictions
    pool_real = get_pred(real).cpu().numpy()
    pool_fake = get_pred(fake).cpu().numpy()
    def get_moments(pool):
        mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
        return mu, sigma
    mu1, sigma1 = get_moments(pool_real)
    mu2, sigma2 = get_moments(pool_fake)
    fid = numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

