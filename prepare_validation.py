# Prepare images and labels for validation
# Copyright (c) 2021 Laurent Caplette

import numpy as np
import os, csv, h5py
from PIL import Image

np.random.seed()
homedir = '' ### path of home directory

nBlocks = 7
nTrialsInBlock = 50
nTrials = nBlocks * nTrialsInBlock

f = h5py.File('vis_visgen_ultimatefinal.h5', 'r')
visgen_im = f['vis'][:]
f.close()
visgen_words = np.load('most_common_visgen_final.npy')

f = h5py.File('vis_mostnamed_ultimatefinal.h5', 'r')
mn_im = f['vis'][:]
f.close()
mn_words = np.load('most_named_words_final.npy') # all words named >=10x

ind = np.where([word not in mn_words for word in visgen_words])[0] # which words don't overlap

ims = np.concatenate((mn_im[:250], visgen_im[ind[:100]]), axis=0) # take 250 most named & top 100 VisGen that were named <10x
labels = np.concatenate((mn_words[:250], visgen_words[ind[:100]]))

os.mkdir(homedir+'/DNN_noise_validation_final_expt/html/resources/Images')
os.chdir(homedir+'/DNN_noise_validation_final_expt/html/resources/Images/')
for ii in range(nTrials):
    im = Image.fromarray(ims[ii])
    im.save('finalval_' + str(ii) + '.png', 'png')

os.chdir('../')

for suj in range(100): # make for more subjects than necessary

    image_nbs = np.random.permutation(nTrials).astype(np.int32)  # image presentation order for subject

    cond_file_list = [['label_list_finalval_' + str(suj) + '_' + str(block) + '.csv']
                      for block in range(nBlocks - 1)] # exclude last block because will be in another loop
    with open('cond_file_list_finalval_' + str(suj) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['label_file'])  # header
        writer.writerows(cond_file_list)

    for block in range(nBlocks):
        trial_nbs = np.arange(nTrialsInBlock)
        im_nbs = image_nbs[block * nTrialsInBlock:(block + 1) * nTrialsInBlock]
        ts = np.random.choice([0, 1], nTrialsInBlock)  # whether true is on the left (0) or right (1)
        falsenbs = [np.random.choice(image_nbs[image_nbs != ii]) for ii in im_nbs] # choose random label among wrong ones
        falselabels = labels[falsenbs]
        truelabels = labels[im_nbs]
        labels_left = [truelabels[ii] if ts[ii] == 0 else falselabels[ii] for ii in range(nTrialsInBlock)]
        labels_right = [falselabels[ii] if ts[ii] == 0 else truelabels[ii] for ii in range(nTrialsInBlock)]

        with open('label_list_finalval_' + str(suj) + '_' + str(block) + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['trialInBlock_nb', 'im_nb', 'label_left', 'label_right', 'ts'])
            rows = list(zip(trial_nbs, im_nbs, labels_left, labels_right, ts))
            writer.writerows(rows)

os.chdir(homedir)

np.save('final_validation_labels.npy', labels)

