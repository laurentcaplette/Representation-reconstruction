# Prepare images and labels for validation #2
# Copyright (c) 2022 Laurent Caplette

import numpy as np
import os, csv, h5py
from PIL import Image

np.random.seed()
homedir = '' ### path of home directory

nBlocks = 5
nTrialsInBlock = 20
nTrials = nBlocks * nTrialsInBlock

f = h5py.File('vis_mostnamed_ultimatefinal.h5', 'r')
mn_im = f['vis'][:]
f.close()
mn_words = np.load('most_named_words_final.npy')

ims = mn_im[:100]
labels = mn_words[:100]

os.mkdir(homedir+'/DNN_noise_validation2_expt/html/resources/Images')
os.chdir(homedir+'/DNN_noise_validation2_expt/html/resources/Images/')
for ii in range(nTrials):
    im = Image.fromarray(ims[ii])
    im.save('valid2_' + str(ii) + '.png', 'png')

os.chdir('../')

for suj in range(100): # make for more subjects than necessary

    image_nbs = np.random.permutation(nTrials).astype(np.int32)  # image presentation order for subject

    cond_file_list = [['label_list_valid2_' + str(suj) + '_' + str(block) + '.csv']
                      for block in range(nBlocks - 1)] # exclude last block because will be in another loop
    with open('cond_file_list_valid2_' + str(suj) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['label_file'])  # header
        writer.writerows(cond_file_list)

    for block in range(nBlocks):
        trial_nbs = np.arange(nTrialsInBlock)
        im_nbs = image_nbs[block * nTrialsInBlock:(block + 1) * nTrialsInBlock]
        # don't use labels in these files so that subjects cant see them in the code

        with open('label_list_valid2_' + str(suj) + '_' + str(block) + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['trialInBlock_nb', 'im_nb'])
            rows = list(zip(trial_nbs, im_nbs))
            writer.writerows(rows)

os.chdir(homedir)
