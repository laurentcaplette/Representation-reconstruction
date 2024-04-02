# Prepare images and labels for validation #3
# Copyright (c) 2023 Laurent Caplette

import numpy as np
import os, csv, h5py
from PIL import Image

np.random.seed()
homedir = '' ### path of home directory

imSize = 224

nRecon = 25 # number of concepts / recons
nppim = 3 # number of perm images per recon
nIms = int((nRecon * (nppim+1))) # 100
nTrials = nIms
nBlocks = 5
nTrialsInBlock = np.round(nIms/nBlocks).astype(int)

mn_words = np.load('most_named_words_final.npy')

# real reconstructions
f = h5py.File('vis_mostnamed_ultimatefinal.h5', 'r')
mn_im = f['vis'][:]
f.close()

# permutation reconstructions
f1 = h5py.File(homedir+'main_expt_data/vis_perm_final.h5', 'r')
f2 = h5py.File(homedir+'/vis_perm_add.h5', 'r')
p_im = np.concatenate((f1['vis'][:],f2['vis'][:]),axis=0)
f1.close()
f2.close()

# select the first 100
perm_ims = p_im.reshape((nIms,3,imSize,imSize,3))
recon_ims = mn_im[:nIms]
labels = mn_words[:nIms]

# select random 25 among these
np.random.seed(2020)
selection = np.sort(np.random.choice(nIms,nRecon,replace=False))
recon_ims2 = recon_ims[selection]
perm_ims2 = perm_ims[selection]
labels2 = labels[selection]

# create variables for experiment
all_ims = np.concatenate((recon_ims2,perm_ims2.reshape((nRecon*nppim,imSize,imSize,3))),axis=0)
all_labels = np.tile(labels2,nppim+1)
all_cond = np.concatenate((np.ones(nRecon),np.zeros(nRecon*nppim)))
all_recon_nbs = np.tile(selection,nppim+1)
all_permID = np.concatenate((np.zeros(nRecon),np.ones(nRecon),np.ones(nRecon)*2,np.ones(nRecon)*3))

# save png images for experiment
os.mkdir(homedir+'/DNN_noise_validation3_expt/html/resources/Images')
os.chdir(homedir+'/DNN_noise_validation3_expt/html/resources/Images/')
for ii in range(nTrials):
    im = Image.fromarray(all_ims[ii])
    im.save('valid3_' + str(ii) + '.png', 'png')

os.chdir('../')

# create csv files for experiment
for suj in range(100): # make for more subjects than necessary

    image_nbs = np.random.permutation(nTrials).astype(np.int32)  # image presentation order for subject
    all_recon_nbs_ordered = all_recon_nbs[image_nbs]
    all_cond_ordered = all_cond[image_nbs]
    all_permID_ordered = all_permID[image_nbs]

    cond_file_list = [['label_list_valid3_' + str(suj) + '_' + str(block) + '.csv']
                      for block in range(nBlocks - 1)] # exclude last block because will be in another loop
    with open('cond_file_list_valid3_' + str(suj) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['label_file'])  # header
        writer.writerows(cond_file_list)

    for block in range(nBlocks):
        trial_nbs = np.arange(nTrialsInBlock) # trial nb in order
        im_nbs = image_nbs[block * nTrialsInBlock:(block + 1) * nTrialsInBlock] # image vector idx
        recon_nbs = all_recon_nbs_ordered[block * nTrialsInBlock:(block + 1) * nTrialsInBlock] # concept ID (0-100)
        cond = all_cond_ordered[block * nTrialsInBlock:(block + 1) * nTrialsInBlock] # 0 if perm, 1 if recon
        permID = all_permID_ordered[block * nTrialsInBlock:(block + 1) * nTrialsInBlock] # 0 if recon, 1-3 if perm
        # don't use labels in these files so that subjects cant see them in the code

        with open('label_list_valid3_' + str(suj) + '_' + str(block) + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['trialInBlock_nb', 'im_nb', 'recon_nb', 'cond', 'permID'])
            rows = list(zip(trial_nbs, im_nbs, recon_nbs, cond, permID))
            writer.writerows(rows)

os.chdir(homedir)
