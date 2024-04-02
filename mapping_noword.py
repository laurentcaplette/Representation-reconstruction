# Analyses without using the concerned words for each concept
# Copyright (c) 2021 Laurent Caplette

import numpy as np
from sklearn.decomposition import PCA
import h5py, json, pickle

np.random.seed() # randomly initialize the seed

homedir = '' ### path of home directory

nwords = 10
ncomps_feat = 213
nperms = 1000

thewords = np.load('most_named_words_final.npy')[:nwords]
all_feats = np.load('all_feats_final.npy')
all_im_words = np.load('all_im_words_final.npy', allow_pickle=True)
all_y_human = np.load('all_y_final.npy') # just to get empty trials
aa = np.where(np.isnan(all_y_human))
all_feats2 = np.delete(all_feats, np.unique(aa[0]), axis=0)
pca_feats = pickle.load(open('pca_feats_object_final.obj','rb'))

# load word embedding
glove_dim = 300
vico_dir = homedir+'/pretrained_vico/glove_300_vico_linear_100/'
f = h5py.File(vico_dir+'visual_word_vecs.h5py','r') # glove = 300 first dimensions of this ViCo embedding
embed_mat = f['embeddings'][()]
f.close()
word_to_idx = json.load(open(vico_dir+'visual_word_vecs_idx.json','r'))

# recover word features when word was not used in the mapping creation
noword_comps = np.zeros((nwords, ncomps_feat))
noword_comps_perm = np.zeros((nperms, nwords, ncomps_feat))
for count, theword in enumerate(thewords):

    print(count)

    all_y = np.empty((0, glove_dim), dtype=np.float32)
    for im_nb in range(10000):
        words = [word for word in all_im_words[im_nb] if word != theword] # exclude responses that are the word

        embed = np.empty((0, glove_dim))
        for word in words:
            embed = np.append(embed, np.expand_dims(embed_mat[word_to_idx[word], :glove_dim], axis=0), axis=0)

        all_y = np.append(all_y, np.mean(embed, axis=0, keepdims=True), axis=0)

    bb = np.where(np.isnan(all_y))
    all_y2 = np.delete(all_y, np.unique(bb[0]), axis=0)
    all_feats_temp = np.delete(all_feats, np.unique(bb[0]), axis=0)
    feats_pc_temp = pca_feats.transform(all_feats_temp) # transform vis chans to comps used in main analysis
    pca_embed = PCA(whiten=True)
    y_pc_temp = pca_embed.fit_transform(all_y2) # words x components

    ncomps_y = np.where(np.cumsum(pca_embed.explained_variance_ratio_) > .9)[0][0] + 1 # to maintain 90% variance

    beta_temp = feats_pc_temp.T @ y_pc_temp[:,:ncomps_y]
    beta_perm_temp = np.zeros((nperms,*np.shape(beta_temp)))
    for perm in range(nperms):
        beta_perm_temp[perm] = feats_pc_temp.T @ y_pc_temp[np.random.permutation(np.shape(y_pc_temp)[0]),:ncomps_y]

    word_sem_comps = pca_embed.transform(np.expand_dims(embed_mat[word_to_idx[theword], :glove_dim], axis=0))[0,:ncomps_y]
    noword_comps[count] = beta_temp @ word_sem_comps # compute vis comps vector for category
    for perm in range(nperms):
        noword_comps_perm[perm, count] = beta_perm_temp[perm] @ word_sem_comps

np.save('noword_comps_final.npy', noword_comps) # save for reconstructions

# correlate feature vectors with original ones
word_comps = np.load('most_named_words_comps_final.npy')[:nwords]
word_chans = pca_feats.inverse_transform(word_comps) # back to dnn channels
noword_chans = pca_feats.inverse_transform(noword_comps) # back to dnn channels
noword_chans_perm = np.zeros((nperms,nwords,3072))
for count in range(nwords):
    noword_chans_perm[:,count] = pca_feats.inverse_transform(noword_comps_perm[:,count]) # back to dnn channels
thecorr = np.zeros(nwords)
thecorr_perm = np.zeros((nperms, nwords))
thecorr_boot = np.zeros((nperms, nwords))
for word_idx in range(nwords):
    thecorr[word_idx] = np.corrcoef(word_chans[word_idx],noword_chans[word_idx])[0,1]
    for perm in range(nperms):
        thecorr_perm[perm, word_idx] = np.corrcoef(word_chans[word_idx],noword_chans_perm[perm,word_idx])[0,1] # permute one variable (more conservative)
for boot in range(nperms):
    idx = np.random.choice(3072, 3072)
    for word_idx in range(nwords):
        thecorr_boot[boot, word_idx] = np.corrcoef(word_chans[word_idx,idx],noword_chans[word_idx,idx])[0,1]
np.save('thecorr_noword.npy', thecorr)
np.save('thecorr_perm_noword.npy', thecorr_perm)
np.save('thecorr_boot_noword.npy', thecorr_boot)

thecorrZ = (thecorr-np.mean(thecorr_perm,axis=0))/np.std(thecorr_perm,axis=0)
thecorr_permZ = (thecorr_perm-np.mean(thecorr_perm,axis=0,keepdims=True))/np.std(thecorr_perm,axis=0,keepdims=True)
q = np.quantile(np.amax(thecorr_permZ,axis=-1),.95) # threshold for statistical significance