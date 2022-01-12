# Creation of binary visual-semantic mapping (no semantic embedding) and further analyses
# Copyright (c) 2021 Laurent Caplette

import numpy as np
import pickle

ncomps_feat = 213
nperms = 1000

# load visual features
all_im_words = np.load('all_im_words_final.npy', allow_pickle=True)
all_feats = np.load('all_feats_final.npy')
all_y_human = np.load('all_y_final.npy') # just to get empty trials
aa = np.where(np.isnan(all_y_human))
all_feats2 = np.delete(all_feats, np.unique(aa[0]), axis=0)
pca_feats = pickle.load(open('pca_feats_object_final.obj','rb'))
feats_pc = pca_feats.transform(all_feats2) # same as original analysis

# collect responses
thewords = np.load('most_named_words_final.npy')  # all the words named 10x or more
all_y = np.zeros((10000, len(thewords)), dtype=np.float16)
for im_nb in range(10000):
    words = [word for word in all_im_words[im_nb]]
    for count, theword in enumerate(thewords):
        if theword in words:
            all_y[im_nb, count] = 1.0

# Regression (weighted sum), all at the same time (binary embedding)
all_y2 = np.delete(all_y, np.unique(aa[0]), axis=0) # take out trials without valid answers, same as original analysis
all_yZ = (all_y2 - np.mean(all_y2, axis=0, keepdims=True)) / np.std(all_y2, axis=0, keepdims=True) # standardize each variable
noembed_comps = all_yZ.T@feats_pc # categories x visual features
np.save('noembed_comps_final.npy', noembed_comps) # save for reconstructions
noembed_comps_perm = np.zeros((nperms,len(thewords),ncomps_feat))
for perm in range(nperms):
    noembed_comps_perm[perm] = all_yZ[np.random.permutation(np.shape(all_yZ)[0])].T @ feats_pc  # simple weighted sum

# correlate feature vectors with original ones (they are both in the same space)
categ_comps = np.load('most_named_words_comps_final.npy')
nwords = np.shape(categ_comps)[0]
categ_chans = pca_feats.inverse_transform(categ_comps)
noembed_chans = pca_feats.inverse_transform(noembed_comps)
noembed_chans_perm = np.zeros((nperms,nwords,3072))
for count in range(nwords):
    noembed_chans_perm[:,count] = pca_feats.inverse_transform(noembed_comps_perm[:,count]) # back to dnn channels
thecorr = np.zeros(nwords)
thecorr_perm = np.zeros((nperms, nwords))
thecorr_boot = np.zeros((nperms, nwords))
for word_idx in range(nwords):
    thecorr[word_idx] = np.corrcoef(categ_chans[word_idx],noembed_chans[word_idx])[0,1]
    for perm in range(nperms):
        thecorr_perm[perm, word_idx] = np.corrcoef(categ_chans[word_idx],noembed_chans_perm[perm,word_idx])[0,1] # permute one variable (more conservative)
for boot in range(nperms):
    idx = np.random.choice(3072, 3072)
    for word_idx in range(nwords):
        thecorr_boot[boot, word_idx] = np.corrcoef(categ_chans[word_idx,idx],noembed_chans[word_idx,idx])[0,1]
np.save('thecorr_noembed.npy', thecorr)
np.save('thecorr_perm_noembed.npy', thecorr_perm)
np.save('thecorr_boot_noembed.npy', thecorr_boot)

q = np.quantile(np.amax(thecorr_perm,axis=-1),.95) # threshold for statistical significance

# assess relationship between frequency of word and correlation value
freqs = np.load('most_named_words_freqs_final.npy')
x = np.log(freqs) # log of frequency (so that relationship is approx. linear)
y = np.arctanh(thecorr) # Fisher-transformed correlation (so that it is unbounded)
y_perm = np.arctanh(thecorr_perm)
R2 = np.corrcoef(x,y)[0,1]**2 # R2=0.64 (0.80 correlation)
R2_perm = np.zeros(nperms)
for perm in range(nperms):
    R2_perm[perm] = np.corrcoef(x,y_perm[perm])[0,1]**2

q = np.quantile(R2_perm,.95) # threshold for statistical significance


