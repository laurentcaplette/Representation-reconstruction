# Creation of the visual-semantic mapping (main expt) and related analyses
# Copyright (c) 2021 Laurent Caplette

import numpy as np
from sklearn.decomposition import PCA
import nltk
import csv, os, re
import torch
from torch import nn
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import h5py, json, pickle
import pkg_resources
from symspellpy import SymSpell, Verbosity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
#nltk.download()

np.random.seed() # randomly initialize the seed

homedir = '' ### path of home directory

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7) # for correction of  spelling mistakes
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

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

# list of stopwords to ignore (from NLTK)
stop_words = nltk.corpus.stopwords.words('english')

# load glove word embedding
glove_dim = 300
vico_dir = homedir+'/pretrained_vico/glove_300_vico_linear_100/' # from ViCo website and elsewhere
f = h5py.File(vico_dir+'visual_word_vecs.h5py','r') # glove = 300 first dimensions of this ViCo embedding
embed_mat = f['embeddings'][()]
f.close()
word_to_idx = json.load(open(vico_dir+'visual_word_vecs_idx.json','r')) # from ViCo website (400k vocab, not just visual)

# visual words (from Visual Genome database)
obj_synsets = json.load(open('object_synsets.json','r')) # from Visual Genome website
obj_synsets_list = list(obj_synsets.values())
osl = [name[:-5] for name in obj_synsets_list]
all_objects = list(set(osl))

os.chdir(homedir + '/DNN_noise_naming_expt/html/final_data')

# which files are results
pattern = '[0-9]{1,3}_DNN.+csv'
resfiles = []
for file in os.listdir():
    match = re.fullmatch(pattern, file)
    if match:
        resfiles.append(file)

# retrieve all data, correct mistakes, transform to vectors
stimfilenbs = np.array([])
prolificids = np.array([], 'f')
all_feats = np.empty((0,1024+2048), dtype=np.float32)
all_y = np.empty((0,glove_dim), dtype=np.float32)
all_im_words = [[] for ii in range(10000)]
all_words = []
wordcount = np.zeros(100)
badwordcount = np.zeros(100)
imcount = -1
for file_idx, filename in enumerate(resfiles): # for each participant/file

    print(file_idx)

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        headers = next(reader, None)
        column = {}
        for h in headers:
            column[h] = []
        for row in reader:
            for h, v in zip(headers, row):
                column[h].append(v)

    label0 = np.array(column['label0']) # first of the 3 potential labels
    label1 = np.array(column['label1']) # second
    label2 = np.array(column['label2']) # third
    resCont0 = np.array(column['resCont0']) # was there a word ending suggestion?...
    resCont1 = np.array(column['resCont1']) # ... if there was, it's recorded (for each label)...
    resCont2 = np.array(column['resCont2']) # ... in case subject confuses tab & enter
    stimfilenbs = np.append(stimfilenbs, int(column['participant'][1])) # participant nb
    prolificids = np.append(prolificids, column['id'][1]) # prolific ID

    # delete rows without data
    label0 = np.delete(label0, [0, 1, 2, 3, 4, 25, 46, 67, 88])
    label1 = np.delete(label1, [0, 1, 2, 3, 4, 25, 46, 67, 88])
    label2 = np.delete(label2, [0, 1, 2, 3, 4, 25, 46, 67, 88])
    resCont0 = np.delete(resCont0, [0, 1, 2, 3, 4, 25, 46, 67, 88])
    resCont1 = np.delete(resCont1, [0, 1, 2, 3, 4, 25, 46, 67, 88])
    resCont2 = np.delete(resCont2, [0, 1, 2, 3, 4, 25, 46, 67, 88])

    for im_nb in range(nIms):

        imcount += 1

        embed = np.empty((0, glove_dim))
        for lbl in range(3):

            label = eval('label' + str(lbl))
            resCont = eval('resCont' + str(lbl))

            responses = label[im_nb]

            responses2 = responses.lower()  # make lowercase
            tokens = nltk.word_tokenize(responses2)  # tokenize everything in words
            words = [w for w in tokens if not w in stop_words]  # reject if stopword
            words = [w for w in words if w not in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']] # reject if number

            # correct words if necessary
            for idx, word in enumerate(words):

                wordcount[file_idx] += 1

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

                if (word not in word_to_idx) or (len(word) < 2): # word must be recognized by embedding and min. 2 letters
                    # if last word and there is suggestion, combine them
                    if (word == words[-1]) & (word + resCont[im_nb] in word_to_idx):
                        words[-1] = word + resCont[im_nb]
                    else:
                        accepted = False
                        if len(word) > 2:  # if nonexistent word is just 1 or 2 letters, don't try to correct it
                            # check for spelling mistakes
                            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                            # put visual word suggestions before the other correction suggestions
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
                        if not accepted:  # if word too short or impossible to correct, remove it
                            words[idx] = ''
                            badwordcount[file_idx] += 1

            # if there's more than one word (after stopword & nonword removal), join them if it's recognized by embedding
            if len(words) > 1:
                if '-'.join(words) in word_to_idx:
                    words = ['-'.join(words)]
                elif ('-'.join(words)).capitalize() in word_to_idx:
                    words = [('-'.join(words)).capitalize()]
                elif '-'.join([w.capitalize() for w in words]) in word_to_idx:
                    words = ['-'.join([w.capitalize() for w in words])]

            words = [w for w in words if w != '']

            # convert responses to embedding vectors
            for word in words:
                embed = np.append(embed,np.expand_dims(embed_mat[word_to_idx[word],:glove_dim],axis=0),axis=0)
                all_im_words[imcount].append(word)
                all_words.append(word)

        all_y = np.append(all_y, np.mean(embed, axis=0, keepdims=True), axis=0)

    # collect actual features (channel activations) of all stimuli
    f = h5py.File(homedir+'/rnd_stim/rndstim_RRN50_R3dot4_'+str(stimfilenbs[-1].astype(np.int32))+'.h5','r')
    stim = f['stim'][:nIms]
    f.close()
    imgs = np.transpose(stim.astype(np.float16), (0, 3, 1, 2)) / 255 # [0,1]
    imgs = torch.tensor((imgs - imagenet_mean_batch)/imagenet_std_batch)
    feats = np.empty((nIms,0))
    feats = np.append(feats,np.mean(np.mean(fe(imgs).numpy(),axis=3),axis=2),axis=1) # sampled layer
    feats = np.append(feats,np.mean(np.mean(fe3(imgs).numpy(),axis=3),axis=2),axis=1) # higher layer
    all_feats = np.append(all_feats,feats,axis=0)

os.chdir(homedir)
np.save('all_im_words_final.npy', all_im_words)
np.save('all_feats_final.npy', all_feats)
np.save('all_y_final.npy', all_y)
unique_words = list(set(all_words))
np.save('unique_words_final.npy', unique_words)

# delete trials without an embedding vector (no valid response)
aa = np.where(np.isnan(all_y))
all_y2 = np.delete(all_y, np.unique(aa[0]), axis=0)
all_feats2 = np.delete(all_feats, np.unique(aa[0]), axis=0)

# perform PCAs on visual and semantic features
ncomps_feat = 213 # to keep 90% variance
ncomps_y = 127 # to keep 90% variance
pca_feats = PCA(whiten=True, n_components=ncomps_feat, random_state=2020)
feats_pc = pca_feats.fit_transform(all_feats2) # trials x components
pca_embed = PCA(whiten=True, n_components=ncomps_y, random_state=2020)
y_pc = pca_embed.fit_transform(all_y2) # trials x components
pickle.dump(pca_feats, open('pca_feats_object_final.obj', 'wb'))
pickle.dump(pca_embed, open('pca_embed_object_final.obj', 'wb'))

# creation of mapping (weighted sum)
beta_human = feats_pc.T @ y_pc
np.save('beta_human_final.npy', beta_human)

# null permutation distribution of mappings
nperms = 1000
beta_perm_human = np.zeros((nperms,*np.shape(beta_human)))
for perm in range(nperms):
    beta_perm_human[perm] = feats_pc.T @ y_pc[np.random.permutation(np.shape(y_pc)[0])]
np.save('beta_perm_human_final.npy', beta_perm_human)

# transform coefficients in z-scores for stats
betaZ = (beta_human-np.mean(beta_perm_human,axis=0))/np.std(beta_perm_human,axis=0)
beta_permZ = (beta_perm_human-np.mean(beta_perm_human,axis=0))/np.std(beta_perm_human,axis=0)

# bootstrap distribution to quantify uncertainty
beta_boot_human = np.zeros((nperms,*np.shape(beta_human)))
for boot in range(nperms):
    idx = np.random.choice(np.shape(feats_pc)[0],np.shape(feats_pc)[0])
    beta_boot_human[boot] = feats_pc[idx].T@y_pc[idx]
np.save('beta_boot_human_final.npy', beta_boot_human)

# statistical significance thresholds
qmax = np.quantile(np.amax(beta_permZ.reshape((nperms, ncomps_feat * ncomps_y)), axis=1), .975)
qmin = np.quantile(np.amin(beta_permZ.reshape((nperms, ncomps_feat * ncomps_y)), axis=1), .025)

# function to count frequencies of words
def CountFrequency(my_list):
    count = {}
    for i in my_list:
        count[i] = count.get(i, 0) + 1
    return count

# create feature vectors for words to reconstruct: words from experiment
word_freqs = CountFrequency(all_words)
sorted_freqs = {k: v for k, v in sorted(word_freqs.items(), key=lambda item: item[1], reverse=True)}
most_named_words = list(sorted_freqs.keys())[:369] # named >= 10x
most_named_freqs = list(sorted_freqs.values())[:369] # named >= 10x
vis_comps = np.zeros((len(most_named_words),ncomps_feat))
for idx, word in enumerate(most_named_words): # visual features for each word to reconstruct
    word_embedding = embed_mat[word_to_idx[word],:glove_dim]
    word_comps = pca_embed.transform(np.expand_dims(word_embedding,axis=0))
    vis_comps[idx] = word_comps @ beta_human.T  # words x vis PC
np.save('most_named_words_final.npy', most_named_words)
np.save('most_named_words_freqs_final.npy', most_named_freqs)
np.save('most_named_words_comps_final.npy', vis_comps)

# create feature vectors for words to reconstruct: words from Visual Genome database
obj = json.load(open('objects.json','r')) # from Visual Genome website
objlabels = [] # all instances of objects in all images
for im_nb in range(len(obj)):
    for i in range(len(obj[im_nb]['objects'])):
        objlabels.append(obj[im_nb]['objects'][i]['names'][0])
object_freqs = CountFrequency(objlabels)
sorted_freqs = {k: v for k, v in sorted(object_freqs.items(), key=lambda item: item[1], reverse=True)}
most_named_objects = list(sorted_freqs.keys())[:250] # 250 most common
most_named_freqs = list(sorted_freqs.values())[:250] # 250 most common
vis_comps = np.zeros((len(most_named_objects),ncomps_feat))
for idx, word in enumerate(most_named_objects): # visual features for each word to reconstruct
    word_embedding = embed_mat[word_to_idx[word],:glove_dim]
    word_comps = pca_embed.transform(np.expand_dims(word_embedding,axis=0))
    vis_comps[idx] = word_comps @ beta_human.T  # words x vis PC
np.save('most_common_visgen_final.npy', most_named_objects)
np.save('most_common_visgen_freqs_final.npy', most_named_freqs)
np.save('most_common_visgen_comps_final.npy', vis_comps)

# compute bootstrap and null feature vectors for specific words
words = most_named_words[:75] # do only for some words
vis_comps_boot_top = np.zeros((len(words),ncomps_feat))
vis_comps_boot_bttm = np.zeros((len(words),ncomps_feat))
vis_comps_perm = np.zeros((3*len(words),ncomps_feat))
for idx, word in enumerate(words):
    sem_embed = embed_mat[word_to_idx[word],:glove_dim]
    sem_comps = pca_embed.transform(np.expand_dims(sem_embed,axis=0))
    vis_comps_boot_top[idx] = np.quantile(beta_boot_human, .025, axis=0) @ sem_comps[0]
    vis_comps_boot_bttm[idx] = np.quantile(beta_boot_human, .975, axis=0) @ sem_comps[0]
    for perm in range(3): # do only for 3 different permutations
        vis_comps_perm[idx*3+perm] = beta_perm_human[perm] @ sem_comps[0]
np.save('vis_comps_boot_bttm_final.npy', vis_comps_boot_bttm)
np.save('vis_comps_boot_top_final.npy', vis_comps_boot_top)
np.save('vis_comps_perm_final.npy', vis_comps_perm)

# top words associated with semantic PCs
nsemcomps = 6 # nb of semantic PCs
nwords = 5 # nb of words
sem_vecs = np.empty((0,glove_dim))
for word in unique_words: # embedding vectors of all unique words
    sem_vecs = np.append(sem_vecs,np.expand_dims(embed_mat[word_to_idx[word],:glove_dim],axis=0),axis=0)
sem_comps = pca_embed.transform(sem_vecs) # transform sem embeds to sem PCs (center, whitening, PCA)
topwords = np.zeros((nsemcomps,nwords),dtype=np.object)
topvals = np.zeros((nsemcomps,nwords))
for comp_idx in range(nsemcomps):
    for word_idx in range(nwords):
        topwords[comp_idx,word_idx] = unique_words[np.argsort(sem_comps[:,comp_idx])[-word_idx-1]]
        topvals[comp_idx,word_idx] = np.sort(sem_comps[:,comp_idx])[-word_idx-1]

# top words associated with visual PCs
nviscomps = 8 # nb of visual PCs
nwords = 5 # nb of words
topwords2 = np.zeros((nviscomps,nwords),dtype=np.object)
topvals2 = np.zeros((nviscomps,nwords))
for comp_idx in range(nviscomps):
    mat = cos_sim(sem_vecs, pca_embed.inverse_transform(np.expand_dims(beta_human[comp_idx],axis=0))) # cos sim between all words and sem feats associated to vis PC
    for word_idx in range(nwords):
        topwords2[comp_idx,word_idx] = unique_words[np.argsort(mat[:,0])[-word_idx-1]]
        topvals2[comp_idx,word_idx] = np.sort(mat[:,0])[-word_idx-1]

# count number of labels before corrections and removals
os.chdir(homedir + '/DNN_noise_naming_expt/html/final_data')
pattern = '[0-9]{1,3}_DNN.+csv'
resfiles = []
for file in os.listdir(): # which files are results
    match = re.fullmatch(pattern, file)
    if match:
        resfiles.append(file)

nlabels = np.zeros(len(resfiles))
for file_idx, filename in enumerate(resfiles):  # for each participant/file

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        headers = next(reader, None)
        column = {}
        for h in headers:
            column[h] = []
        for row in reader:
            for h, v in zip(headers, row):
                column[h].append(v)

    label0 = np.array(column['label0'])  # first of the 3 potential labels
    label1 = np.array(column['label1'])  # second
    label2 = np.array(column['label2'])  # third

    # delete rows without data
    label0 = np.delete(label0, [0, 1, 2, 3, 4, 25, 46, 67, 88])
    label1 = np.delete(label1, [0, 1, 2, 3, 4, 25, 46, 67, 88])
    label2 = np.delete(label2, [0, 1, 2, 3, 4, 25, 46, 67, 88])

    nlabels[file_idx] = len(list(filter(None, label0))) + len(list(filter(None, label1))) + len(
        list(filter(None, label2))) # nb of non-empty strings

nlabels_avg = np.mean(nlabels/nIms) # mean nb of labels per trial (2.17)






