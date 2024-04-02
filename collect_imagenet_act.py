# Collect layer activations to ImageNet images
# Copyright (c) 2021 Laurent Caplette

import torch
from torchvision import transforms
from torch import nn
import os, h5py, glob
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import numpy as np

imagenet_mean = torch.tensor(np.array([0.485, 0.456, 0.406])).type(torch.float32).cuda()
imagenet_std = torch.tensor(np.array([0.229, 0.224, 0.225])).type(torch.float32).cuda()

imagedir = '/gpfs/milgram/project/turk-browne/projects/dnn_revcorr/ILSVRC_2012_val' ###

# here load pretrained robust model from Madry Lab (robustness toolbox)
ds = ImageNet('/tmp')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds, parallel=False,
             resume_path='') #### here include path of imagenet_l2_3_0.pt file
model.cuda()
model.eval() # so that in test mode

for param in model.parameters():
    param.requires_grad = False  # don't compute gradients for parameters


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


layer_nb = 6
sublayer_nb = 4
fe = FeatExtract_mid(model.model, layer_nb, sublayer_nb).cuda()

NUNITS = 1024 # nb of channels
NSPATIAL = 14 # nb of units in a channel
SIZE = 224 # size of images
IM_NUM = 50000 # total nb of images
MB_SIZE = 500 # size of chunks to process one at a time

os.chdir(imagedir)
for f in sorted(glob.glob("im[0-9][0-9][0-9][0-9][0-9].h5")):
    #if os.path.isfile(f[0:7]+'_RRN50_R3dot4.h5'):
        #continue

    images_file = h5py.File(f, 'r')
    images = images_file['data'][:]
    images_file.close()
    NIMAGES = images.shape[0]

    activ = np.zeros((NIMAGES,NUNITS,NSPATIAL,NSPATIAL),dtype=np.float32) ###
    for im in range(NIMAGES): # could be multiple at the same time but unnecessary

        img =  images[im]

        img_tensor = torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1)) / 255).type(torch.float32).cuda() # [0,1]
        img_tensor = transforms.functional.normalize(img_tensor, imagenet_mean, imagenet_std) # in normalized space
        act = fe(img_tensor.unsqueeze(0)) # features
        activ[im] = act[0,:].cpu().numpy()

    output_file = h5py.File(f[0:7] + '_RRN50_R3dot4.h5', 'w') ###
    output_file.create_dataset('activ', data=activ)
    output_file.close()


# put all outputs in one file
idx = 0
flist = sorted(glob.glob('*_RRN50_R3dot4.h5'))
activ_all = np.zeros((IM_NUM,NUNITS),dtype=np.float32)
for f in flist:
    temp_file = h5py.File(f, 'r')
    activ = temp_file['activ'][:]
    temp_file.close()

    activ_all[idx:idx+activ.shape[0],:] = np.mean(np.mean(activ,axis=-2),axis=-1)

    idx += activ.shape[0]

activ_all = np.delete(activ_all,np.arange(idx,IM_NUM),axis=0) # delete superfluous empty images if any

out_file = h5py.File('RRN50_R3dot4_avg.h5','w')
out_file.create_dataset('activ',data=activ_all)
out_file.close()

