# Analyses of the DNN representations
# Copyright (c) 2021 Laurent Caplette

import numpy as np
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import wordnet as wn
import csv, os, re
import torch
from torch import nn
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import h5py, json, pickle, urllib
#nltk.download()

np.random.seed() # randomly initialize the seed

homedir = '' ### path of home directory

classes = pickle.load(urllib.request.urlopen( # load ImageNet class names
'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))

nIms = 100 # nb of images per participant
SIZE = 224  # image size (square)

ds = ImageNet('/tmp') # load robust ResNet-50 (from robustness toolbox / Madry Lab website)
net, _ = make_and_restore_model(arch='resnet50', dataset=ds, parallel=False, resume_path=homedir + '/imagenet_l2_3_0.pt')
net.eval()  # so that in test mode

imagenet_mean = torch.tensor(np.array([0.485, 0.456, 0.406])).type(torch.float32) # from ImageNet training set
imagenet_std = torch.tensor(np.array([0.229, 0.224, 0.225])).type(torch.float32) # from ImageNet training set
imagenet_mean_batch = np.tile(imagenet_mean.numpy()[np.newaxis, :, np.newaxis, np.newaxis], (nIms, 1, SIZE, SIZE))
imagenet_std_batch = np.tile(imagenet_std.numpy()[np.newaxis, :, np.newaxis, np.newaxis], (nIms, 1, SIZE, SIZE))


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


for param in net.parameters():
    param.requires_grad = False  # don't compute gradients for parameters

# Defines layer from which to extract features
fe = FeatExtract_mid(net.model, 6, 4)
fe3 = FeatExtract_mid(net.model, 7, 0)

# list of stopwords to ignore (from NTLK)
stop_words = nltk.corpus.stopwords.words('english')

# load glove word embedding
glove_dim = 300
vico_dir = homedir+'/pretrained_vico/glove_300_vico_linear_100/' # from ViCo website and elsewhere
f = h5py.File(vico_dir+'visual_word_vecs.h5py','r') # glove = 300 first dimensions of this ViCo embedding
embed_mat = f['embeddings'][()]
f.close()
word_to_idx = json.load(open(vico_dir+'visual_word_vecs_idx.json','r')) # from ViCo website and elsewhere

os.chdir(homedir + '/DNN_noise_naming_expt/html/final_data')

# retrieve appropriate stimuli files...
pattern = '[0-9]{1,3}_DNN.+csv'
resfiles = []
for file in os.listdir():
    match = re.fullmatch(pattern, file)
    if match:
        resfiles.append(file)

stimfilenbs = np.array([])
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

    stimfilenbs = np.append(stimfilenbs, int(column['participant'][1]))

os.chdir(homedir)

nresp = 3  # nb of responses per image
nSuj = 100
nIms = 100

model_embeds = np.zeros((nSuj, nIms, glove_dim))
all_im_words = [[] for ii in range(nSuj*nIms)]
for ii in range(nSuj):

    print(ii)

    f = h5py.File(homedir + '/rnd_stim/rndstim_RRN50_R3dot4_' + str(stimfilenbs[ii].astype(np.int32)) + '.h5', 'r')
    stim = f['stim'][:nIms]
    f.close()

    imgs = np.transpose(stim.astype(np.float16), (0, 3, 1, 2)) / 255  # [0,1]
    imgs = torch.tensor((imgs - imagenet_mean_batch) / imagenet_std_batch)

    logits = net.model(imgs) # model predictions (1000-dimensional logit vector for each image)

    for im_nb in range(nIms):

        preds = torch.nn.functional.softmax(logits[im_nb]).numpy() # 1000-dim probability vector

        labels = [classes[i].lower() for i in np.argsort(preds)[-nresp:]]

        labels2 = []
        for label in labels:
            label = label.split(',')[0]  # use first part of label which is the most usual or basic name
            labels2.append('-'.join(label.split(' ')))  # if multiple words, join by dash

        # find all wordnet hypernyms associated with label
        hyper = [[] for label in labels2]
        for idx, label in enumerate(labels2):
            hyper[idx].append(label)
            temp = wn.synsets(label)
            while len(temp) != 0:
                tempname = temp[0].name()[:-5]
                tempname.replace('_','-') # replace underscores by dashes to fit with word embedding
                hyper[idx].append(tempname)
                temp = temp[0].hypernyms()

        # if label is too specific for word embedding (network is sometimes), replace with wordnet hypernym
        embed = np.zeros(glove_dim)
        for label in hyper:
            for level in label:
                try:
                    embed += embed_mat[word_to_idx[level], :glove_dim]
                    all_im_words[im_nb].append(level) # record if it is used
                    break
                except:
                    continue
        embed /= nresp # not really necessary since using cosine sim, but to be safe

        model_embeds[ii, im_nb] = embed

np.save('model_embeds_final.npy', model_embeds)

all_feats = np.load('all_feats_final.npy')
pca_feats = pickle.load(open('pca_feats_object_final.obj','rb'))
feats_pc = pca_feats.transform(all_feats) # images x components

model_embeds = np.load('model_embeds_final.npy')
model_embeds = model_embeds.reshape((nSuj*nIms,glove_dim))
aa = [np.count_nonzero(model_embeds[ii])==0 for ii in range(10000)]
model_embeds = np.delete(model_embeds, aa, axis=0)
feats_pc = np.delete(feats_pc, aa, axis=0)

ncomps_embed_model = 118 # to retain 90% variance
pca_embed_model = PCA(whiten=True, n_components=ncomps_embed_model, random_state=2020)
model_embed_pc = pca_embed_model.fit_transform(model_embeds) # trials x components
pickle.dump(pca_embed_model, open('pca_embed_model_object_final.obj', 'wb'))

# creation of mapping (weighted sum)
beta_model = feats_pc.T @ model_embed_pc
np.save('beta_model_final.npy', beta_model)

# null permutation distribution of mappings
nperms = 1000
beta_perm_model = np.zeros((nperms,*np.shape(beta_model)))
for perm in range(nperms):
    beta_perm_model[perm] = feats_pc.T @ model_embed_pc[np.random.permutation(np.shape(model_embed_pc)[0])]
np.save('beta_perm_model_final.npy', beta_perm_model)

# transform coefficients in z-scores for stats
betaZ = (beta_model-np.mean(beta_perm_model,axis=0))/np.std(beta_perm_model,axis=0)
beta_permZ = (beta_perm_model-np.mean(beta_perm_model,axis=0))/np.std(beta_perm_model,axis=0)

# statistical significance thresholds
ncomps_feat = 213
qmax = np.quantile(np.amax(beta_permZ.reshape((nperms, ncomps_feat * ncomps_embed_model)), axis=1), .975)
qmin = np.quantile(np.amin(beta_permZ.reshape((nperms, ncomps_feat * ncomps_embed_model)), axis=1), .025)

nw = 100

# feature vectors for most common Visual Genome words
words = np.load('most_common_visgen_final.npy')[:nw]
vis_comps = np.zeros((len(words),ncomps_feat))
for idx, word in enumerate(words):
    word_embedding = embed_mat[word_to_idx[word],:glove_dim]
    word_comps = pca_embed_model.transform(np.expand_dims(word_embedding,axis=0))
    vis_comps[idx] = word_comps @ beta_model.T
np.save('visgen_comps_model_final.npy', vis_comps)

all_feats = np.load('all_feats_final.npy')

# inter-group differences analysis
all_feats = np.load('all_feats_final.npy')
pca_feats = pickle.load(open('pca_feats_object_final.obj','rb'))
pca_embed_model = pickle.load(open('pca_embed_model_object_final.obj','rb'))
pca_embed_human = pickle.load(open('pca_embed_object_final.obj','rb'))
model_embeds = np.load('model_embeds_final.npy')
model_embeds = model_embeds.reshape((nSuj*nIms,glove_dim))
human_embeds = np.load('all_y_final.npy')
ncomps_embed_human = 127
ncomps_embed_model = 118

idx = np.array(np.arange(10000//2))
idx1 = np.zeros(10000).astype(np.bool)
idx1[idx] = True
idx2 = np.ones(10000).astype(np.bool)
idx2[idx] = False

# invalid trials
human_invalid = np.unique(np.where(np.isnan(human_embeds))[0])
human_invalid1 = np.unique(np.where(np.isnan(human_embeds[idx1]))[0]) # for half 1 only
human_invalid2 = np.unique(np.where(np.isnan(human_embeds[idx2]))[0]) # for half 2 only
human_embeds[human_invalid] = 0 # temporarily replace by zeros (will be deleted later)
model_invalid = np.array([np.count_nonzero(model_embeds[ii])==0 for ii in range(10000)])
model_invalid1 = model_invalid[idx1]
model_invalid2 = model_invalid[idx2]

# transform to PCs
feats_pc = pca_feats.transform(all_feats)
model_embed_pc = pca_embed_model.transform(model_embeds)
human_embed_pc = pca_embed_human.transform(human_embeds)

# divide into halves
feats_pc1 = feats_pc[idx1]
feats_pc2 = feats_pc[idx2]
y_model_pc1 = model_embed_pc[idx1]
y_model_pc2 = model_embed_pc[idx2]
y_human_pc1 = human_embed_pc[idx1]
y_human_pc2 = human_embed_pc[idx2]

# now remove invalid trials from each half
feats_pc_human1 = np.delete(feats_pc1, human_invalid1, axis=0)
feats_pc_human2 = np.delete(feats_pc2, human_invalid2, axis=0)
feats_pc_model1 = np.delete(feats_pc1, model_invalid1, axis=0)
feats_pc_model2 = np.delete(feats_pc2, model_invalid2, axis=0)
y_human_pc1 = np.delete(y_human_pc1, human_invalid1, axis=0)
y_human_pc2 = np.delete(y_human_pc2, human_invalid2, axis=0)
y_model_pc1 = np.delete(y_model_pc1, model_invalid1, axis=0)
y_model_pc2 = np.delete(y_model_pc2, model_invalid2, axis=0)

categ_feats_model = np.zeros((2,nw,ncomps_feat))
categ_feats_model_perm = np.zeros((2,nperms,nw,ncomps_feat))
categ_feats_human = np.zeros((2,nw,ncomps_feat))
categ_feats_human_perm = np.zeros((2,nperms,nw,ncomps_feat))

# model vectors, fold 1
beta_model1 = feats_pc_model1.T@y_model_pc1 # simple weighted sum
beta_model1_perm = np.zeros((nperms,ncomps_feat,ncomps_embed_model))
for perm in range(nperms):
    beta_model1_perm[perm] = feats_pc_model1.T@y_model_pc1[np.random.permutation(np.shape(y_model_pc1)[0])]
for ii, word in enumerate(words):
    word_embedding = embed_mat[word_to_idx[word],:glove_dim]
    word_comps = pca_embed_model.transform(np.expand_dims(word_embedding,axis=0))
    categ_feats_model[0,ii] = beta_model1 @ word_comps[0]
    for perm in range(nperms):
        categ_feats_model_perm[0,perm,ii] = beta_model1_perm[perm] @ word_comps[0]

# model vectors, fold 2
beta_model2 = feats_pc_model2.T@y_model_pc2 # simple weighted sum
beta_model2_perm = np.zeros((nperms,ncomps_feat,ncomps_embed_model))
for perm in range(nperms):
    beta_model2_perm[perm] = feats_pc_model2.T@y_model_pc2[np.random.permutation(np.shape(y_model_pc2)[0])]
for ii, word in enumerate(words):
    word_embedding = embed_mat[word_to_idx[word],:glove_dim]
    word_comps = pca_embed_model.transform(np.expand_dims(word_embedding,axis=0))
    categ_feats_model[1,ii] = beta_model2 @ word_comps[0]
    for perm in range(nperms):
        categ_feats_model_perm[1,perm,ii] = beta_model2_perm[perm] @ word_comps[0]

# human vectors, fold 1
beta_human1 = feats_pc_human1.T@y_human_pc1 # simple weighted sum
beta_human1_perm = np.zeros((nperms,ncomps_feat,ncomps_embed_human))
for perm in range(nperms):
    beta_human1_perm[perm] = feats_pc_human1.T@y_human_pc1[np.random.permutation(np.shape(y_human_pc1)[0])]
for ii, word in enumerate(words):
    word_embedding = embed_mat[word_to_idx[word],:glove_dim]
    word_comps = pca_embed_human.transform(np.expand_dims(word_embedding,axis=0))
    categ_feats_human[0,ii] = beta_human1 @ word_comps[0]
    for perm in range(nperms):
        categ_feats_human_perm[0,perm,ii] = beta_human1_perm[perm] @ word_comps[0]

# human vectors, fold 2
beta_human2 = feats_pc_human2.T@y_human_pc2 # simple weighted sum
beta_human2_perm = np.zeros((nperms,ncomps_feat,ncomps_embed_human))
for perm in range(nperms):
    beta_human2_perm[perm] = feats_pc_human2.T@y_human_pc2[np.random.permutation(np.shape(y_human_pc2)[0])]
for ii, word in enumerate(words):
    word_embedding = embed_mat[word_to_idx[word],:glove_dim]
    word_comps = pca_embed_human.transform(np.expand_dims(word_embedding,axis=0))
    categ_feats_human[1,ii] = beta_human2 @ word_comps[0]
    for perm in range(nperms):
        categ_feats_human_perm[1,perm,ii] = beta_human2_perm[perm] @ word_comps[0]

# test by looking at within-group vs between-group correlations
model0 = np.ravel(pca_feats.inverse_transform(categ_feats_model[0])) # transform to DNN channels before corr
model1 = np.ravel(pca_feats.inverse_transform(categ_feats_model[1])) # transform to DNN channels before corr
human0 = np.ravel(pca_feats.inverse_transform(categ_feats_human[0]))
human1 = np.ravel(pca_feats.inverse_transform(categ_feats_human[1]))
between_total = (np.corrcoef(model0,human1)[0,1] + np.corrcoef(model1,human0)[0,1])/2
within_total = (np.corrcoef(model0,model1)[0,1] + np.corrcoef(human0,human1)[0,1])/2
between_total_perm = np.zeros(nperms)
within_total_perm = np.zeros(nperms)
for perm in range(nperms):
    model0_perm = np.ravel(pca_feats.inverse_transform(categ_feats_model_perm[0,perm]))
    model1_perm = np.ravel(pca_feats.inverse_transform(categ_feats_model_perm[1,perm]))
    human0_perm = np.ravel(pca_feats.inverse_transform(categ_feats_human_perm[0,perm]))
    human1_perm = np.ravel(pca_feats.inverse_transform(categ_feats_human_perm[1,perm]))
    between_total_perm[perm] = (np.corrcoef(model0_perm,human1_perm)[0,1] + np.corrcoef(model1_perm,human0_perm)[0,1])/2
    within_total_perm[perm] = (np.corrcoef(model0_perm,model1_perm)[0,1] + np.corrcoef(human0_perm,human1_perm)[0,1])/2

diff_total = within_total-between_total
diff_total_perm = within_total_perm-between_total_perm
np.save('model_diff.npy', diff_total)
np.save('model_diff_perm.npy', diff_total_perm)

diff_totalZ = (diff_total-np.mean(diff_total_perm))/np.std(diff_total_perm)
diff_total_permZ = (diff_total_perm-np.mean(diff_total_perm))/np.std(diff_total_perm)

q = np.quantile(diff_total_permZ, .95) # threshold for statistical significance
