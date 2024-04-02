# Analyses of individual observer representations
# Copyright (c) 2021 Laurent Caplette

import numpy as np
from sklearn.decomposition import PCA
import nltk
import csv, os, re, json
import torch
from torch import nn
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import h5py, pickle
import pkg_resources
from symspellpy import SymSpell, Verbosity

np.random.seed()

homedir = '' ### path of home directory
os.chdir(homedir)

sym_spell = SymSpell(max_dictionary_edit_distance=2) # for correction of spelling mistakes
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

nsuj = 8 # nb of subjects
nIms = 125 # nb of images per participant
SIZE = 224  # image size (square)

ds = ImageNet('/tmp') # load robust ResNet-50 (from robustness toolbox / Madry Lab website)
net, _ = make_and_restore_model(arch='resnet50', dataset=ds, parallel=False, resume_path=homedir + '/imagenet_l2_3_0.pt')
net.eval()  # so that in test mode

imagenet_mean = torch.tensor(np.array([0.485, 0.456, 0.406])).type(torch.float32) # from ImageNet training set
imagenet_std = torch.tensor(np.array([0.229, 0.224, 0.225])).type(torch.float32) # from ImageNet training set
imagenet_mean_batch = np.tile(imagenet_mean.numpy()[np.newaxis, :, np.newaxis, np.newaxis], (nIms, 1, SIZE, SIZE))
imagenet_std_batch = np.tile(imagenet_std.numpy()[np.newaxis, :, np.newaxis, np.newaxis], (nIms, 1, SIZE, SIZE))


class FeatExtract_mid(
    nn.Module):  # get features from the first layer of a submodule (here, 1st res block of a submodule)
    def __init__(self, net, layer_nb, sublayer_nb):
        super(FeatExtract_mid, self).__init__()
        self.name = 'middle_of_stage'
        self.net = net
        self.layer_nb = layer_nb
        self.sublayer_nb = sublayer_nb
        for p in self.net.parameters():
            p.requires_grad = False
        featnetlist = list(self.net.children())
        featnetlist[self.layer_nb] = featnetlist[self.layer_nb][
                                     :self.sublayer_nb + 1]  # only take first X modules of the stage
        self.features = nn.Sequential(*featnetlist[:self.layer_nb + 1])

    def forward(self, x):
        return self.features(x)


for param in net.parameters():
    param.requires_grad = False  # don't compute gradients for parameters

# Defines layer from which to extract features
fe = FeatExtract_mid(net.model, 6, 4)
fe3 = FeatExtract_mid(net.model, 7, 0)

# load glove word embedding
glove_dim = 300
vico_dir = homedir+'/pretrained_vico/glove_300_vico_linear_100/' # from ViCo website and elsewhere
f = h5py.File(vico_dir+'visual_word_vecs.h5py','r') # glove = 300 first dimensions of this ViCo embedding
embed_mat = f['embeddings'][()]
f.close()
word_to_idx = json.load(open(vico_dir+'visual_word_vecs_idx.json','r')) # from ViCo website and elsewhere

# visual words (from Visual Genome database)
obj_synsets = json.load(open('object_synsets.json','r')) # from Visual Genome website
obj_synsets_list = list(obj_synsets.values())
osl = [name[:-5] for name in obj_synsets_list]
all_objects = list(set(osl))

# list of stopwords to ignore (from NTLK)
stop_words = nltk.corpus.stopwords.words('english')

# collect features of stimuli shown in expt (same 750 for all subjects)
all_feats = np.empty((0, 1024 + 2048), dtype=np.float32)
for session in range(6):
    f = h5py.File(homedir + '/rnd_stim/rndstim_RRN50_R3dot4_' + str(160 + session) + '.h5', 'r')
    stim = f['stim'][:nIms]
    f.close()

    imgs = np.transpose(stim.astype(np.float16), (0, 3, 1, 2)) / 255  # [0,1]
    imgs = torch.tensor((imgs - imagenet_mean_batch) / imagenet_std_batch)

    feats = np.empty((nIms, 0))
    feats = np.append(feats, np.mean(np.mean(fe(imgs).numpy(), axis=3), axis=2), axis=1)
    feats = np.append(feats, np.mean(np.mean(fe3(imgs).numpy(), axis=3), axis=2), axis=1)

    all_feats = np.append(all_feats, feats, axis=0)

np.save('all_feats_ind_final.npy', all_feats)


os.chdir(homedir + '/DNN_noise_naming_ind_expt/html/final_data')

# collect responses of each subject
for participant in range(nsuj):

    print(participant)

    pattern = str(participant) + '_DNN.+csv'
    resfiles = []
    for file in os.listdir():
        match = re.fullmatch(pattern, file)
        if match:
            resfiles.append(file)

    all_y = np.empty((0, glove_dim), dtype=np.float32)
    all_im_words = np.array([])
    all_words = []
    sessions = np.array([])
    for file_idx, filename in enumerate(resfiles): # sessions

        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            headers = next(reader, None)
            column = {}
            for h in headers:
                column[h] = []
            for row in reader:
                for h, v in zip(headers, row):
                    column[h].append(v)

        label0 = np.array(column['label0'])
        label1 = np.array(column['label1'])
        label2 = np.array(column['label2'])
        resCont0 = np.array(column['resCont0'])
        resCont1 = np.array(column['resCont1'])
        resCont2 = np.array(column['resCont2'])
        imID = np.array(column['im_ID'])
        sessions = np.append(sessions, int(column['session'][1]))

        # delete rows without data
        label0 = np.delete(label0, [0, 1, 2, 3, 4, 30, 56, 82, 108])
        label1 = np.delete(label1, [0, 1, 2, 3, 4, 30, 56, 82, 108])
        label2 = np.delete(label2, [0, 1, 2, 3, 4, 30, 56, 82, 108])
        resCont0 = np.delete(resCont0, [0, 1, 2, 3, 4, 30, 56, 82, 108])
        resCont1 = np.delete(resCont1, [0, 1, 2, 3, 4, 30, 56, 82, 108])
        resCont2 = np.delete(resCont2, [0, 1, 2, 3, 4, 30, 56, 82, 108])
        imID = np.delete(imID, [0, 1, 2, 3, 4, 30, 56, 82, 108])

        imID = np.array([int(ii) for ii in imID])

        y = np.empty((0, glove_dim), dtype=np.float32)
        all_im_words_temp = [[] for ii in range(nIms)]
        for im_nb in range(nIms):

            embed = np.empty((0, glove_dim))
            for lbl in range(3):

                label = eval('label' + str(lbl))
                resCont = eval('resCont' + str(lbl))

                responses = label[im_nb]

                responses2 = responses.lower()  # make lowercase
                tokens = nltk.word_tokenize(responses2)  # tokenize everything in words
                words = [w for w in tokens if not w in stop_words]  # reject if stopword #####
                words = [w for w in words if w not in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']]

                # correct words if necessary
                for idx, word in enumerate(words):

                    # additional step to correct when subject most likely typed enter instead of tab but it's still a (most likely obscure) word
                    if (word == words[-1]) & (word not in all_objects) & (word + resCont[im_nb] in all_objects):
                        words[-1] = word + resCont[im_nb]

                    # check if numbers and reject if so
                    isnumber = False
                    for letter in word:
                        try:
                            int(letter)
                        except:
                            continue
                        else:
                            isnumber = True
                    if isnumber:
                        words[idx] = ''

                    if (word not in word_to_idx) or (len(word) < 2):
                        if (word == words[-1]) & (word + resCont[im_nb] in word_to_idx):
                            words[-1] = word + resCont[im_nb]
                        else:
                            accepted = False
                            if len(word) > 2:  # if nonexistent word is just 1 or 2 letters, don't try to correct it
                                # check for spelling mistakes
                                suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                                # put visual suggestions before the others
                                suggestion_vis = []
                                suggestion_nonvis = []
                                for suggestion in suggestions:
                                    if suggestion.term in all_objects:
                                        suggestion_vis.append(suggestion.term)
                                    else:
                                        suggestion_nonvis.append(suggestion.term)
                                suggestion_terms = suggestion_vis + suggestion_nonvis
                                for suggestion in suggestion_terms: # if first suggestion is not in vocab, try next (in decreasing frequency)
                                    if suggestion in word_to_idx:
                                        words[idx] = suggestion
                                        accepted = True
                                        break
                                    else:
                                        continue
                            if not accepted:  # if word impossible or too short to correct, remove it
                                words[idx] = ''

                # if there's more than one word (after stopword & nonword removal), try joining them
                if len(words) > 1:
                    if '-'.join(words) in word_to_idx:
                        words = ['-'.join(words)]
                    elif ('-'.join(words)).capitalize() in word_to_idx:
                        words = [('-'.join(words)).capitalize()]
                    elif '-'.join([w.capitalize() for w in words]) in word_to_idx:
                        words = ['-'.join([w.capitalize() for w in words])]

                words = [w for w in words if w != '']

                for word in words:
                    embed = np.append(embed, np.expand_dims(embed_mat[word_to_idx[word], :300], axis=0), axis=0)
                    all_im_words_temp[im_nb].append(word)
                    all_words.append(word)

            y = np.append(y, np.mean(embed, axis=0, keepdims=True), axis=0)

        # rearrange images in order before concatenating sessions
        all_y = np.append(all_y, y[np.argsort(imID)], axis=0)
        all_im_words_temp = np.array(all_im_words_temp, dtype='object')
        all_im_words = np.concatenate((all_im_words, all_im_words_temp[np.argsort(imID)]))

    os.chdir(homedir)
    np.save('all_words_ind' + str(participant) + '_final.npy', all_words)
    unique_words = list(set(all_words))
    np.save('unique_words_ind' + str(participant) + '_final.npy', unique_words)

    # rearrange sessions in order
    all_im_words2 = np.zeros_like(all_im_words)
    all_y2 = np.zeros_like(all_y)
    for sess in range(6):
        idx = np.squeeze(np.where(sessions==sess))
        all_im_words2[nIms*sess:nIms*(sess+1)] = all_im_words[nIms*idx:nIms*(idx+1)]
        all_y2[nIms*sess:nIms*(sess+1)] = all_y[nIms*idx:nIms*(idx+1)]
    np.save('sorted_all_im_words_ind' + str(participant) + '_final.npy', all_im_words2)
    np.save('sorted_y_ind' + str(participant) + '_final.npy', all_y2)
    os.chdir(homedir + '/DNN_noise_naming_ind_expt/html/final_data')

os.chdir(homedir)

# count all the words
def CountFrequency(my_list):
    count = {}
    for i in my_list:
        count[i] = count.get(i, 0) + 1
    return count

all_the_words = np.array([])
for participant in range(nsuj):
    all_words = np.load('all_words_ind' + str(participant) + '_final.npy')
    all_the_words = np.append(all_the_words, all_words, axis=0)

word_freqs = CountFrequency(all_the_words)
sorted_freqs = {k: v for k, v in sorted(word_freqs.items(), key=lambda item: item[1], reverse=True)}
most_named_words_ind = list(sorted_freqs.keys())[:294] # named >= 10x
most_named_freqs_ind = list(sorted_freqs.values())[:294] # named >= 10x
np.save('most_named_words_ind_final.npy', most_named_words_ind)
np.save('most_named_words_ind_freqs_final.npy', most_named_freqs_ind)

nperms = 1000
ncomps_feat = 150 # to keep 90% variance

# perform PCA on visual features (same for all subjects)
pca_feats = PCA(whiten=True, n_components=ncomps_feat, random_state=2020)
feats_pc_group = pca_feats.fit_transform(all_feats) # images x components
pickle.dump(pca_feats, open('pca_feats_object_ind_final.obj', 'wb'))

words = np.load('most_named_words_ind_final.npy')

# for each participant, perform PCA on semantic features, weighted sum, and creation of feature vectors for reconstructions
for participant in range(nsuj):

    # load data and delete trials invalid for this subject
    all_y = np.load('sorted_y_ind' + str(participant) + '_final.npy')
    aa = np.where(np.isnan(all_y)) # which trials are invalid (no valid response)
    all_y2 = np.delete(all_y, np.unique(aa[0]), axis=0)
    feats_pc_ind = np.delete(feats_pc_group, np.unique(aa[0]), axis=0)

    # perform PCA on semantic features for this subject
    pca_embed_temp = PCA(whiten=True, random_state=2020)
    pca_embed_temp.fit(all_y2)
    ncomps_y_ind = np.where(np.cumsum(pca_embed_temp.explained_variance_ratio_)>.9)[0][0]+1
    pca_embed_ind = PCA(whiten=True, n_components=ncomps_y_ind, random_state=2020)
    y_pc_ind = pca_embed_ind.fit_transform(all_y2) # trials x components
    pickle.dump(pca_embed_ind, open('pca_embed_object_ind' + str(participant) + '_final.obj', 'wb'))

    # perform weighted sum for this subject (in subject-specific semantic space)
    beta_ind = feats_pc_ind.T @ y_pc_ind # weighted sum
    beta_perm_ind = np.zeros((nperms,*np.shape(beta_ind)))
    for perm in range(nperms):
        beta_perm_ind[perm] = feats_pc_ind.T @ y_pc_ind[np.random.permutation(np.shape(y_pc_ind)[0])]
    np.save('beta_ind' + str(participant) + '_final.npy', beta_ind)
    np.save('beta_perm_ind' + str(participant) + '_final.npy', beta_perm_ind)

    # retrieve subject-specific visual representations of most named words
    vis_comps = np.zeros((len(words), ncomps_feat))
    for idx, word in enumerate(words):
        word_embedding = embed_mat[word_to_idx[word], :glove_dim]
        word_comps = pca_embed_ind.transform(np.expand_dims(word_embedding, axis=0)) # in subject-specific semantic PC space
        vis_comps[idx] = word_comps @ beta_ind.T  # words x vis PC
    np.save('most_named_words_comps_ind' + str(participant) + '_final.npy', vis_comps)

# prepare inter-individual differences analyses
idx = np.array(np.arange(125*3))
idx1 = np.zeros(125*6).astype(np.bool)
idx1[idx] = True # 1st and 2nd half of stimuli for both sets (shown in different orders for different subjects)
idx2 = np.ones(125*6).astype(np.bool)
idx2[idx] = False

pca_embed_main = pickle.load(open('pca_embed_object_final.obj', 'rb')) # semantic space from main expt
ncomps_y_main = 127 # nb of semantic PCs in main study

# create feature vectors and same-space betas for data halves for inter-individual differences analyses
beta_reproj = np.zeros((nsuj, 2, ncomps_feat, ncomps_y_main))
beta_reproj_perm = np.zeros((nsuj, 2, nperms, ncomps_feat, ncomps_y_main))
for p in range(nsuj):

    # load files
    all_y = np.load('sorted_y_ind' + str(p) + '_final.npy')
    pca_embed_ind = pickle.load(open('pca_embed_object_ind' + str(p) + '_final.obj', 'rb'))

    # invalid trials for this subject
    aa = np.where(np.isnan(all_y))
    aa1 = np.where(np.isnan(all_y[idx1])) # for half 1 only
    aa2 = np.where(np.isnan(all_y[idx2])) # for half 2 only
    all_y[np.unique(aa[0])] = 0 # temporarily replace by zeros (will be deleted later)

    # project all trials on semantic space from previous study
    y_pc_reproj = pca_embed_main.transform(all_y)

    # divide all data into halves
    feats_pc1 = feats_pc_group[idx1]
    feats_pc2 = feats_pc_group[idx2]
    y_pc_reproj1 = y_pc_reproj[idx1]
    y_pc_reproj2 = y_pc_reproj[idx2]

    # now remove invalid trials from each half
    feats_pc1 = np.delete(feats_pc1, np.unique(aa1[0]), axis=0)
    feats_pc2 = np.delete(feats_pc2, np.unique(aa2[0]), axis=0)
    y_pc_reproj1 = np.delete(y_pc_reproj1, np.unique(aa1[0]), axis=0)
    y_pc_reproj2 = np.delete(y_pc_reproj2, np.unique(aa2[0]), axis=0)

    # perform weighted sum for each half, in semantic space from main study
    beta_reproj[p,0] = feats_pc1.T @ y_pc_reproj1
    for perm in range(nperms):
        beta_reproj_perm[p,0,perm] = feats_pc1.T @ y_pc_reproj1[np.random.permutation(np.shape(y_pc_reproj1)[0])]
    beta_reproj[p,1] = feats_pc2.T @ y_pc_reproj2
    for perm in range(nperms):
        beta_reproj_perm[p,1,perm] = feats_pc2.T @ y_pc_reproj2[np.random.permutation(np.shape(y_pc_reproj2)[0])]

# check if visual-semantic mappings overall are individually unique (using betas in semantic space from main study)
corrmat = np.zeros((nsuj,nsuj))
corrmat_perm = np.zeros((nperms,nsuj,nsuj))
for suj1 in range(nsuj):
    for suj2 in range(nsuj):
        term1 = np.ravel(pca_feats.inverse_transform(beta_reproj[suj1,0].T)) # transform to DNN channels before corr
        term2 = np.ravel(pca_feats.inverse_transform(beta_reproj[suj2,1].T))
        corrmat[suj1,suj2] = np.corrcoef(term1,term2)[0,1] # corr between feat vecs from different halves
        for perm in range(nperms):
            term1 = np.ravel(pca_feats.inverse_transform(beta_reproj_perm[suj1,0,perm].T))
            term2 = np.ravel(pca_feats.inverse_transform(beta_reproj_perm[suj2,1,perm].T))
            corrmat_perm[perm,suj1,suj2] = np.corrcoef(term1,term2)[0,1]

uniqcoeff = np.zeros((nsuj,nsuj))
uniqcoeff_perm = np.zeros((nperms,nsuj,nsuj))
for s1 in range(nsuj): # for each subject pair, are representations individually unique?
    for s2  in range(nsuj):
        uniqcoeff[s1,s2] = np.mean([corrmat[s1,s1],corrmat[s2,s2]])-np.mean([corrmat[s1,s2],corrmat[s2,s1]])
        for perm in range(nperms):
            uniqcoeff_perm[perm,s1,s2] = np.mean([corrmat_perm[perm,s1,s1],corrmat_perm[perm,s2,s2]])-np.mean([corrmat_perm[perm,s1,s2],corrmat_perm[perm,s2,s1]])

meancoeff = np.mean(uniqcoeff[np.triu_indices(nsuj,k=1)]) # mean of coefficients (matrix is symmetric)
meancoeff_perm = np.zeros(nperms)
for perm in range(nperms):
    temp = uniqcoeff_perm[perm]
    meancoeff_perm[perm] = np.mean(temp[np.triu_indices(nsuj,k=1)])
np.save('indivdiff_beta_meancoeff_final.npy', meancoeff)
np.save('indivdiff_beta_meancoeff_perm_final.npy', meancoeff_perm)

q = np.quantile(meancoeff_perm,.95) # statistical threshold
