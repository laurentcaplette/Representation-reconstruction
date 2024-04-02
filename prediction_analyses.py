# Prediction analyses
# Copyright (c) 2021 Laurent Caplette

import numpy as np
import nltk
import csv, os, re
import torch
from torch import nn
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import h5py, pickle
from sklearn.metrics.pairwise import cosine_similarity as cossim
import pkg_resources
from symspellpy import SymSpell, Verbosity
import json
from PIL import Image
import torchvision
# nltk.download()

np.random.seed()

##### PREDICTION OF SEMANTIC CONTENT #####

stop_words = nltk.corpus.stopwords.words('english')

homedir = '' ### path of home directory
os.chdir(homedir)

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

obj_synsets = json.load(open('object_synsets.json', 'r'))
obj_synsets_list = list(obj_synsets.values())
osl = [name[:-5] for name in obj_synsets_list]
all_objects = list(set(osl))

nIms = 100
nperms = 1000

beta_human = np.load('beta_human_final.npy')
pca_embed = pickle.load(open('pca_embed_object_final.obj','rb'))
pca_feats = pickle.load(open('pca_feats_object_final.obj','rb'))
ncomps_feat = 213
ncomps_y = 127

layer_nb = 6  # submodule nb
sublayer_nb = 4
NUNITS = 1024  # nb of channels in layer
NSPATIAL = 14  # nb of spatial positions in layer
SIZE = 224  # image size (square)

imagenet_mean = torch.tensor(np.array([0.485, 0.456, 0.406])).type(torch.float32)
imagenet_std = torch.tensor(np.array([0.229, 0.224, 0.225])).type(torch.float32)
imagenet_mean_batch = np.tile(imagenet_mean.numpy()[np.newaxis, :, np.newaxis, np.newaxis], (nIms, 1, SIZE, SIZE))
imagenet_std_batch = np.tile(imagenet_std.numpy()[np.newaxis, :, np.newaxis, np.newaxis], (nIms, 1, SIZE, SIZE))

ds = ImageNet('/tmp')
net, _ = make_and_restore_model(arch='resnet50', dataset=ds, parallel=False,
                                resume_path=homedir + '/imagenet_l2_3_0.pt')
net.eval()  # so that in test mode


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

# Defines layer from which you extract features (which feature extractor + layer nb)
fe = FeatExtract_mid(net.model, 6, 4)
fe3 = FeatExtract_mid(net.model, 7, 0)

glove_dim = 300

vico_dir = homedir + '/pretrained_vico/glove_300_vico_linear_100/'
f = h5py.File(vico_dir + 'visual_word_vecs.h5py', 'r')
word_to_idx = json.load(open(vico_dir + 'visual_word_vecs_idx.json', 'r'))
visual_words = json.load(open(vico_dir + 'visual_words.json', 'r'))
embed_mat = f['embeddings'][()]

unique_words = np.load('unique_words_final.npy')
all_embed = np.array([embed_mat[word_to_idx[word], :glove_dim] for word in unique_words])

os.chdir(homedir + '/DNN_noise_naming_expt/html/final_data')

pattern = '[0-9]{1,3}_DNN.+csv'
resfiles = []
for file in os.listdir():
    match = re.fullmatch(pattern, file)
    if match:
        resfiles.append(file)

# collect labels given to practice stimuli
labels = [[] for i in range(5)]
stimfilenbs = np.array([])
prolificids = np.array([], 'f')
for file_idx, filename in enumerate(resfiles):

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
    stimfilenbs = np.append(stimfilenbs, int(column['participant'][1]))
    prolificids = np.append(prolificids, column['id'][1])

    # 5 first rows are practice stimuli
    label0 = label0[:5]
    label1 = label1[:5]
    label2 = label2[:5]

    for im in range(5):

        for lbl in range(3):

            label = eval('label' + str(lbl))
            resCont = eval('resCont' + str(lbl))

            responses = label[im]

            responses2 = responses.lower()  # make lowercase
            tokens = nltk.word_tokenize(responses2)  # tokenize everything in words
            words = [w for w in tokens if not w in stop_words]  # reject if stopword
            words = [w for w in words if w not in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']]

            # correct words if necessary
            for idx, word in enumerate(words):

                # additional step to correct when subject most likely typed enter instead of tab but it's still a (most likely obscure) word
                if (word == words[-1]) & (word not in all_objects) & (word + resCont[im] in all_objects):
                    words[-1] = word + resCont[im]

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
                    # if last word and there is suggestion, combine them
                    if (word == words[-1]) & (word + resCont[im] in word_to_idx):
                        words[-1] = word + resCont[im]
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
                            for suggestion in suggestion_terms:
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
                labels[im].append(word)

all_labels = labels[0] + labels[1] + labels[2] + labels[3] + labels[4]

# draw random samples with replacement for null distribution
labels_p = {}
for im in range(5):
    labels_p[im] = np.random.choice(all_labels, (nperms, len(labels[im])))

os.chdir(homedir + '/DNN_noise_naming_expt/html/resources/Images/practice')

# collect visual features and predicted semantic features of 5 stimuli
modelpreds = {}
embed_vecs = np.empty((5, glove_dim))
for nb in range(5):
    image = np.array(Image.open('prac_im_' + str(nb) + '.png'))
    img = torch.tensor(image / 255).permute((2, 0, 1)).type(torch.float32)

    img = torchvision.transforms.functional.normalize(img, imagenet_mean, imagenet_std)
    act = torch.empty((0))
    activ = fe(img.unsqueeze(0))
    act = torch.cat((act, torch.mean(torch.mean(activ[0], axis=-1), axis=-1)), 0)
    activ = fe3(img.unsqueeze(0))
    act = torch.cat((act, torch.mean(torch.mean(activ[0], axis=-1), axis=-1)), 0)

    vis_comps = pca_feats.transform(np.expand_dims(act.numpy(),axis=0))
    sem_comps = vis_comps @ beta_human
    embed_vecs[nb] = pca_embed.inverse_transform(np.expand_dims(sem_comps,axis=0))

# actual semantic features of responses to stimuli
meanembed = np.zeros((5, glove_dim))
for im in range(5):
    ii = 0
    for lbl in labels[im]:
        try:
            ii += 1
            meanembed[im] += embed_mat[word_to_idx[lbl], :glove_dim]
        except:
            print(lbl)
    meanembed[im] /= ii  # not really necessary because cosine sim

# cosine similarity between observed and predicted semantic features
goodcossim = np.zeros(5)
for im in range(5):
    goodcossim[im] = cossim[meanembed[im],embed_vecs[im]]
meancossim = np.mean(goodcossim)

# bootstrap cosine similarity to measure uncertainty
goodcossim_boot = np.zeros((nperms,5))
for boot in range(nperms):
    idx = np.random.choice(glove_dim, glove_dim, replace=True)
    for im in range(5):
        goodcossim_boot[boot,im] = cossim(np.expand_dims(meanembed[im,idx],axis=0),np.expand_dims(embed_vecs[im,idx],axis=0))
meancossim_boot = np.mean(goodcossim_boot,axis=1)

# null semantic features
meanembed_p = np.zeros((5,nperms,glove_dim))
for perm in range(nperms):
    for im in range(5):
        for lbl in labels_p[im][perm]:
            meanembed_p[im,perm] += embed_mat[word_to_idx[lbl],:glove_dim]
        meanembed_p[im,perm] /= len(labels[im]) # not even necessary because cosine sim

# cosine similarity between null and predicted semantic features
goodcossim_p = np.zeros((nperms,5))
for im in range(5):
    goodcossim_p[:,im] = cossim(np.expand_dims(embed_vecs[im],axis=0),meanembed_p[im])

q = np.quantile(np.mean(goodcossim_p,axis=1),.95) # threshold for statistical significance

np.save('sempred_cossim.npy', goodcossim)
np.save('sempred_cossim_perm.npy', goodcossim_p)
np.save('sempred_cossim_boot.npy', goodcossim_boot)

##### PREDICTION OF STIMULI #####

# function to efficiently compute correlations with 2d arrays
def corr2_coeff(A, B):
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

nwords = 10
nstim = 750

os.chdir(homedir)

# load labels from indiv expt
total_words = np.zeros(8, dtype='object')
for participant in range(8):
    total_words[participant] = np.load('sorted_all_im_words_ind' + str(participant) + '_final.npy', allow_pickle=True)

labels = np.empty(nstim, dtype='object')
for im in range(nstim):
    labels[im] = list([])
    for participant in range(8):
        labels[im] += total_words[participant][im]

words = np.load('most_named_words_final.npy') # most answered words in main expt
all_feats = np.load('all_feats_ind_final.npy') # vis feats of stim in indiv expt

dice = np.zeros(nwords)
permdice = np.zeros((nwords, nperms))
for w in range(nwords):

    word = words[w]
    word_embedding = embed_mat[word_to_idx[word], :glove_dim]
    word_comps = pca_embed.transform(np.expand_dims(word_embedding, axis=0))
    vis_comps = beta_human @ word_comps[0]
    vis_feats = pca_feats.inverse_transform(np.expand_dims(vis_comps, axis=0))[0]

    totalpres = np.zeros(nstim).astype(np.bool)
    for ii in range(nstim): # is the word present in responses to stim?
        totalpres[ii] = word in labels[ii]

    vals = np.argsort(corr2_coeff(np.expand_dims(vis_feats, axis=0), all_feats).ravel()) # corr between vis feats of word and vis feats of stims

    nImPred = nstim - np.sum(totalpres).astype(np.int32)
    valpred = np.ones(nstim).astype(np.bool)
    for ii in range(nImPred): # which stims correlate most with vis feats?
        valpred[vals[ii]] = False

    intersection = np.logical_and(valpred, totalpres)
    dice[w] = 2. * intersection.sum() / (valpred.sum() + totalpres.sum())

    for perm in range(nperms): # repeat while permuting one Boolean array
        intersection = np.logical_and(valpred, np.random.permutation(totalpres))
        permdice[w, perm] = 2. * intersection.sum() / (valpred.sum() + totalpres.sum())

diceZ = (dice - np.mean(permdice, axis=-1)) / np.std(permdice, axis=-1)
permdiceZ = (permdice - np.mean(permdice, axis=-1, keepdims=True)) / np.std(permdice, axis=-1, keepdims=True)

q = np.quantile(np.amax(permdiceZ, axis=0), .95)
