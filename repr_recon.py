# Reconstruction of representations
# Copyright (c) 2021 Laurent Caplette

import numpy as np
import torch
from torch import nn
import torchvision
import h5py
from robustness.datasets import RestrictedImageNet, ImageNet
from robustness.model_utils import make_and_restore_model
import pickle
from kornia.augmentation import RandomAffine

np.random.seed() # randomly initialize the seed

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


def data_augmentation(image, padding=16, scaling_factors=(1, 0.975, 1.025, 0.95, 1.05),
                      angles=[-5., 5.], jitter_1=16, jitter_2=16,
                      crop_type=2, crop_size=None):
    image_rank = len(image.size())

    if image_rank != 3:
        raise ValueError("Unsupported image rank")

    # remember original height and width
    if crop_size is not None:
        height = crop_size
        width = crop_size
    else:
        height = image.size()[-2]
        width = image.size()[-1]

    # convert scaling factors and angles into Tensors
    scaling_factors = torch.tensor(scaling_factors, dtype=torch.float32)
    if angles is not None:
        angles = torch.tensor(angles, dtype=torch.float32)

    # apply padding
    if padding is not None:
        if image_rank == 3:
            paddings = (32, 32, 32, 32)
        else:
            raise ValueError("Unsupported image rank")

        image = torch.nn.functional.pad(image, paddings)

    # first jitter
    if jitter_1 is not None:
        # random_jitter_1 = torch.distributions.uniform.Uniform(-jitter_1,jitter_1+1).rsample([2]).type(torch.int32)
        ox, oy = np.random.randint(-jitter_1, jitter_1 + 1, 2)
        image = torch.roll(torch.roll(image, ox, 1), oy, 2)

    # random scaling
    if scaling_factors is not None:
        random_scale_idx = np.random.randint(0, len(scaling_factors), 1)
        random_scale_height = (image.size()[-2] * scaling_factors[random_scale_idx]).type(torch.int32)
        random_scale_width = (image.size()[-1] * scaling_factors[random_scale_idx]).type(torch.int32)

        image = \
        torch.nn.functional.interpolate(image.unsqueeze(0), (random_scale_height, random_scale_width), mode='bilinear')[0]

    else:
        # no scaling performed
        random_scale_height = image.shape[-2]
        random_scale_width = image.shape[-1]

    # random rotation #####
    if angles is not None:
        rot_fn = RandomAffine([-5., 5.])
        image = rot_fn(image.unsqueeze(0))
        image = image[0, :, :, :]

    # second jitter
    if jitter_2 is not None:
        ox, oy = np.random.randint(-jitter_2, jitter_2 + 1, 2)
        image = torch.roll(torch.roll(image, ox, 1), oy, 2)

    # crop out an image of the same size as the original image
    if crop_type == 3:  # random
        raise ValueError("Unsupported crop type for now")
    elif crop_type == 2:  # center
        return image[:, (random_scale_height - height) // 2:height + ((random_scale_height - height) // 2),
               (random_scale_width - width) // 2:width + ((random_scale_width - width) // 2)]
    elif crop_type == 1:  # none
        return image
    else:
        raise ValueError("Unsupported crop type")

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
fe3 = FeatExtract_mid(model.model, 7, 0).cuda() # slightly higher layer

NFILES = 1
SIZE = 224 # image size
lr = 0.05 # learning rate
wd = 0.1 # weight decay
epsilon = 1e-8
nIters = 2000
alpha = 4 # arbitrary exponent for caricature objective function

# Fourier 1/f scaling
d = 1
decay_power = 1
fy = np.fft.fftfreq(SIZE, d=d)[:, None]
fx = np.fft.fftfreq(SIZE, d=d)[:SIZE // 2 + 1]
frequencies = np.sqrt(fx * fx + fy * fy) ** decay_power
spectrum_scale = 1.0 / np.maximum(frequencies, 1.0 / (SIZE * d))
spectrum_scale = np.repeat(np.expand_dims(np.repeat(np.expand_dims(spectrum_scale, axis=-1), 2, axis=-1), axis=0), 3, axis=0)
spectrum_scale = torch.tensor(spectrum_scale).cuda()

comp_obj = pickle.load(open('pca_feats_object_final.obj', 'rb'))
Fvecs = np.load('most_named_words_comps_final.npy') # visual PCs for each word
Fvecs = comp_obj.inverse_transform(Fvecs) # transform to channel activations
NIMAGES = np.shape(Fvecs)[0]

init_coeffs = np.random.normal(size=(3, SIZE, SIZE // 2 + 1, 2), scale=0.01) # initial random values

for file_nb in range(NFILES):

    R2 = np.zeros(NIMAGES)
    vis = np.zeros((NIMAGES, SIZE, SIZE, 3), dtype=np.uint8)
    for im in range(NIMAGES):

        f_vec = torch.tensor(np.expand_dims(Fvecs[im, :], axis=0), dtype=torch.float32).cuda() # target feature vector
        coeffs = torch.tensor(init_coeffs, dtype=torch.float32).cuda().requires_grad_() # initial random values (Fourier coeffs)
        optimizer = torch.optim.AdamW([coeffs], lr=lr, weight_decay=wd) # optimization initialization

        for ii in range(nIters):

            optimizer.zero_grad()

            spectrum = torch.mul(coeffs, spectrum_scale) # 1/f scaling
            img = torch.stack( # inverse Fourier transform
                [torch.irfft(spectrum[i, :, :, :], signal_ndim=2, signal_sizes=(SIZE, SIZE), onesided=True,
                             normalized=True) for i in range(coeffs.shape[0])], dim=0).type(torch.float32)
            img = decorrelate_colors(img, COLOR_CORR_NORM) # decorrelate color channels
            img = torch.sigmoid(img)

            # collect model activations to current optimized image
            img = torchvision.transforms.functional.normalize(img, imagenet_mean, imagenet_std) # normalize before feeding to model
            img = data_augmentation(img) # small transformations to improve synthesis
            act = torch.empty((0)).cuda()
            activ = fe(img.unsqueeze(0))
            act = torch.cat((act,torch.mean(torch.mean(activ[0], axis=-1), axis=-1)),0) # spatially averaged sampled layer
            activ = fe3(img.unsqueeze(0))
            act = torch.cat((act,torch.mean(torch.mean(activ[0],axis=-1),axis=-1)),0) # spatially averager higher layer

            cossim = torch.dot(f_vec[0],act)/torch.mul(torch.sqrt(torch.sum(f_vec[0]**2)),torch.sqrt(torch.sum(act**2)))
            loss = -torch.mul(torch.dot(f_vec[0],act),cossim**alpha) # maximize caricature objective function

            loss.backward()
            torch.nn.utils.clip_grad_norm_(coeffs, 1)
            optimizer.step()

        # reapply these steps to final results to get an image
        spectrum = torch.mul(coeffs, spectrum_scale)
        img = torch.stack(
            [torch.irfft(spectrum[i, :, :, :], signal_ndim=2, signal_sizes=(SIZE, SIZE), onesided=True, normalized=True)
             for i in
             range(coeffs.shape[0])], dim=0).type(torch.float32)
        img = decorrelate_colors(img, COLOR_CORR_NORM)
        img = torch.sigmoid(img)

        vis[im] = (255 * np.transpose(img.cpu().detach().numpy(), (1, 2, 0))).astype(np.uint8) # final image

        # save after each image in case of failure
        output_file = h5py.File('vis_mostnamed_final.h5', 'w')
        output_file.create_dataset('vis', data=vis)
        output_file.close()