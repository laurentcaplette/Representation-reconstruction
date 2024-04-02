# Creation of experimental stimuli
# Copyright (c) 2021 Laurent Caplette

import numpy as np
import torch
from torch import nn
import torchvision
import os, h5py
from robustness.datasets import RestrictedImageNet, ImageNet
from robustness.model_utils import make_and_restore_model
import scipy.stats as stats
from sklearn.covariance import ledoit_wolf

np.random.seed() # randomly initialize the seed

imagedir = '' ### path of Imagenet validation set images
stimdir = '' ### path of stimuli files

# load robust ResNet-50
ds = ImageNet('/tmp')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds, parallel=False,
                                  resume_path='') ### here include path of imagenet_l2_3_0.pt file
model.cuda()
model.eval()
pass


# extract output of any residual block ('sublayer') within any stage ('layer')
class FeatExtract_mid(nn.Module):
    def __init__(self, net, layer_nb, sublayer_nb):
        super(FeatExtract_mid, self).__init__()
        self.name = 'middle_of_stage'
        self.net = net
        self.layer_nb = layer_nb
        self.sublayer_nb = sublayer_nb
        for p in self.net.parameters():
            p.requires_grad = False
        featnetlist = list(self.net.children())
        featnetlist[self.layer_nb] = featnetlist[self.layer_nb][:self.sublayer_nb + 1]
        self.features = nn.Sequential(*featnetlist[:self.layer_nb + 1])

    def forward(self, x):
        return self.features(x)


imagenet_mean = torch.tensor(np.array([0.485, 0.456, 0.406])).type(torch.float32).cuda() # from ImageNet training set
imagenet_std = torch.tensor(np.array([0.229, 0.224, 0.225])).type(torch.float32).cuda() # from ImageNet training set

for param in model.model.parameters():
    param.requires_grad = False  # don't compute gradients for parameters

color_corr_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                  [0.27, 0.00, -0.05],
                                  [0.27, -0.09, 0.03]]).astype("float32") # from ImageNet training set
max_norm_svd_sqrt = np.max(np.linalg.norm(color_corr_svd_sqrt, axis=0))
color_corr_norm = color_corr_svd_sqrt / max_norm_svd_sqrt
COLOR_CORR_NORM = torch.tensor(color_corr_norm, dtype=torch.float32).cuda()


def decorrelate_colors(image, COLOR_CORR_NORM):
    image2 = image.permute(1, 2, 0)
    image_flat = torch.reshape(image2, (-1, 3))  ####
    image_flat = torch.matmul(image_flat, torch.t(COLOR_CORR_NORM))
    image2 = torch.reshape(image_flat, np.shape(image2))
    image = image2.permute(2, 0, 1)

    return image


fe = FeatExtract_mid(model.model, 6, 4).cuda() # sampled layer

NFILES = 15
SIZE = 224 # size of images
NIMAGES = 150
NUNITS = 1024
lr = 0.05 # learning rate
wd = 0.1 # weight decay
epsilon = 1e-8
nIters = 1500

# Fourier 1/f scaling
d = 1
decay_power = 1
fy = np.fft.fftfreq(SIZE, d=d)[:, None]
fx = np.fft.fftfreq(SIZE, d=d)[:SIZE // 2 + 1]
frequencies = np.sqrt(fx * fx + fy * fy) ** decay_power
spectrum_scale = 1.0 / np.maximum(frequencies, 1.0 / (SIZE * d))
spectrum_scale = np.repeat(np.expand_dims(np.repeat(np.expand_dims(spectrum_scale, axis=-1), 2, axis=-1), axis=0), 3, axis=0)
spectrum_scale = torch.tensor(spectrum_scale).cuda()


os.chdir(imagedir)

f = h5py.File('RRN50_R3dot4_avg.h5', 'r')  # load spatially averaged activations to ImageNet validation set
X = f['activ'][:]
f.close()

Xstd = np.std(X, axis=0, keepdims=True) + epsilon
Xmean = np.mean(X, axis=0, keepdims=True)
XZ = (X - Xmean) / Xstd # standardized activations

cov = ledoit_wolf(XZ, assume_centered=True)[0]  # estimate cov mat with optimal shrinkage
U, Lambda, _ = np.linalg.svd(cov)
W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))  # ZCA whitening transform
T = np.dot(XZ, W.T)  # whitened activations
T = T.astype(np.float32)

os.chdir(stimdir)

for file_nb in range(NFILES):

    Tnew = np.zeros((NIMAGES, NUNITS))
    for feat in range(NUNITS):
        t = T[:, feat]
        kernel = stats.gaussian_kde(t)  # distribution of activations in whitened space
        Tnew[:, feat] = kernel.resample(NIMAGES)[0, :]  # random samples from this distribution

    Winv = np.linalg.pinv(W) # (pseudo)inverse of whitening transform: coloring transform
    Fvecs = np.dot(Tnew, Winv) # apply coloring transform
    Fvecs = Fvecs * Xstd + Xmean # destandardize: back to channel space

    R2 = np.zeros(NIMAGES)
    stim = np.zeros((NIMAGES, SIZE, SIZE, 3), dtype=np.uint8)
    for im in range(NIMAGES):

        init_coeffs = np.random.normal(size=(3, SIZE, SIZE // 2 + 1, 2), scale=0.01)
        f_vec = torch.tensor(np.expand_dims(Fvecs[im, :], axis=0), dtype=torch.float32).cuda() # target feature vector
        coeffs = torch.tensor(init_coeffs,dtype=torch.float32).cuda().requires_grad_() # initial random values (Fourier coeffs)
        optimizer = torch.optim.AdamW([coeffs], lr=lr, weight_decay=wd) # optimization initialization

        idx = -1
        for ii in range(nIters):

            optimizer.zero_grad()

            spectrum = torch.mul(coeffs, spectrum_scale) # 1/f scaling
            img = torch.stack( # inverse Fourier transform
                [torch.irfft(spectrum[i, :, :, :], signal_ndim=2, signal_sizes=(224, 224), onesided=True,
                             normalized=True) for i in range(coeffs.shape[0])], dim=0).type(torch.float32)
            img = decorrelate_colors(img, COLOR_CORR_NORM) # decorrelate color channels
            img = torch.sigmoid(img)

            # collect model activations to current optimized image
            img = torchvision.transforms.functional.normalize(img, imagenet_mean, imagenet_std) # normalize before feeding to model
            activ = fe(img.unsqueeze(0))
            act = torch.mean(torch.mean(activ, axis=-1), axis=-1) # spatially averaged activations
            loss = torch.mean((f_vec - act) ** 2) # MSE loss

            # compute R-squared
            act2 = act.cpu().detach().numpy()
            Fvec = f_vec.cpu().detach().numpy()
            SStot = np.sum((act2 - np.mean(act2)) ** 2)
            SSres = np.sum((act2 - Fvec) ** 2)
            R2[im] = 1 - (SSres / SStot)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(coeffs, 1)
            optimizer.step()

        # reapply these steps to final results to get an image
        spectrum = torch.mul(coeffs, spectrum_scale)
        img = torch.stack(
            [torch.irfft(spectrum[i, :, :, :], signal_ndim=2, signal_sizes=(SIZE, SIZE), onesided=True, normalized=True)
             for i in range(coeffs.shape[0])], dim=0).type(torch.float32)
        img = decorrelate_colors(img, COLOR_CORR_NORM)
        img = torch.sigmoid(img)

        stim[im] = (255 * np.transpose(img.cpu().detach().numpy(), (1, 2, 0))).astype(np.uint8) # final stimulus

    # save everything
    output_file = h5py.File('rndstim_RRN50_R3dot4_final_' + str(file_nb) + '.h5', 'w')
    output_file.create_dataset('stim', data=stim) # stimuli
    output_file.create_dataset('features', data=Fvecs) # original target features
    output_file.create_dataset('R2', data=R2) # R-squareds
    output_file.close()

