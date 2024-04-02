# Representational similarity analysis of Nishida et al. (2021) data and our data
# Copyright (c) 2021 Laurent Caplette

import numpy as np
import h5py, csv, json, pickle
from nltk.corpus import wordnet as wn
from scipy.stats import spearmanr
from scipy.stats import rankdata
from scipy.spatial.distance import squareform
#nltk.download()

homedir = '' ### path of home directory

np.random.seed()

nIms = 100 # nb of images per participant
SIZE = 224  # image size (square)
nresp = 3
nperms = 1000
nsuj = 36 # in behavioral similarity judgment data
nnouns = 60 # in behavioral similarity judgment data
ndist = (nnouns*nnouns-nnouns)//2

# load glove word embedding
glove_dim = 300
vico_dir = homedir+'/pretrained_vico/glove_300_vico_linear_100/' # from ViCo website and elsewhere
f = h5py.File(vico_dir+'visual_word_vecs.h5py','r') # glove = 300 first dimensions of this ViCo embedding
embed_mat = f['embeddings'][()]
f.close()
word_to_idx = json.load(open(vico_dir+'visual_word_vecs_idx.json','r')) # from ViCo website and elsewhere

with open(homedir+'/inv_mds_data/RSA_nouns.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    headers = next(reader,None)
    column = {}
    for h in headers:
        column[h] = []
    for row in reader:
        for h, v in zip(headers,row):
            column[h].append(v)

nouns = column['ENGLISH']

# what are the hypernyms for each concept
hyper = [[] for i in nouns]
for idx, im in enumerate(nouns):
    hyper[idx].append(im)
    temp = wn.synsets(im)
    while len(temp)!=0:
        hyper[idx].append(temp[0].name()[:-5])
        temp = temp[0].hypernyms()

# take hypernym if name is not recognized by embedding (only TV & PC -> television & computer)
embeds = np.zeros((len(hyper),glove_dim))
all_im_words = np.zeros(len(hyper),dtype='object')
for idx, label in enumerate(hyper):
    for level in label:
        try:
            embeds[idx] = embed_mat[word_to_idx[level], :glove_dim]
            all_im_words[idx] = level # record if it is used
            break
        except:
            continue

sem_dist = 1 - np.corrcoef(embeds) # semantic distance

pca_embed = pickle.load(open('pca_embed_object_final.obj','rb'))
pca_feats = pickle.load(open('pca_feats_object_final.obj','rb'))
beta_human = np.load('beta_human_final.npy')
word_comps = pca_embed.transform(embeds)
vis_comps = word_comps @ beta_human.T
vis_feats = pca_feats.inverse_transform(vis_comps)

vis_dist = 1 - np.corrcoef(vis_feats) # visual distance

# collect behavioral distanes from similarity judgment data
all_dist = np.zeros((nsuj, ndist))
for suj in range(nsuj):

    f = h5py.File(homedir + '/inv_mds_data/P' + str(suj + 1).zfill(3) + '_Nouns.h5', 'r')
    dist = f['disparities'][:]
    f.close()

    all_dist[suj] = rankdata(dist) # rank before averaging subjects because only order is interpretable

behav_dist = squareform(np.mean(all_dist,axis=0)) # behavioral distance

tri_ind = np.triu_indices(nnouns,1)

# how much each model RDM fits (correlates to) behavioral data
sem_corr = spearmanr(sem_dist[tri_ind],behav_dist[tri_ind])[0]
vis_corr = spearmanr(vis_dist[tri_ind],behav_dist[tri_ind])[0]
corr_diff = vis_corr - sem_corr
np.save('rsa_sem_corr.npy', sem_corr)
np.save('rsa_vis_corr.npy', vis_corr)

# test significance with permuted data
sem_corr_perm = np.zeros(nperms)
vis_corr_perm = np.zeros(nperms)
for perm in range(nperms):
    idx = np.random.permutation(nnouns)
    rdm_temp = np.zeros((nnouns,nnouns))
    for ii in range(nnouns):
        rdm_temp[ii,:] = behav_dist[idx[ii],idx]
    sem_corr_perm[perm] = spearmanr(sem_dist[tri_ind],rdm_temp[tri_ind])[0]
    vis_corr_perm[perm] = spearmanr(vis_dist[tri_ind],rdm_temp[tri_ind])[0]

corr_diff_perm = vis_corr_perm - sem_corr_perm
np.save('rsa_sem_corr_perm.npy', sem_corr_perm)
np.save('rsa_vis_corr_perm.npy', vis_corr_perm)

qv = np.quantile(vis_corr_perm, .95) # stat thresh: is visual RDM fitting the data?
qd = np.quantile(corr_diff_perm, .975) # stat thresh: is visual RDM fitting better than semantic RDM?
