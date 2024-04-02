import numpy as np
from sklearn.decomposition import PCA
import nltk
import csv, os, re
import h5py, json, pickle
import pkg_resources
from symspellpy import SymSpell, Verbosity

np.random.seed() # randomly initialize the seed

homedir = '' ### path of home directory

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7) # for correction of  spelling mistakes
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

nIms = 100 # nb of images per participant
SIZE = 224  # image size (square)

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
#all_feats = np.empty((0,56*56*3), dtype=np.float16)  #####
all_feats = np.empty((0,224*224*3), dtype=np.float16)  #####
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

    stimfilenbs = np.append(stimfilenbs, int(column['participant'][1])) # participant nb
    prolificids = np.append(prolificids, column['id'][1]) # prolific ID

    f = h5py.File(homedir + '/rnd_stim/rndstim_RRN50_R3dot4_' + str(stimfilenbs[-1].astype(np.int32)) + '.h5', 'r')
    stim = f['stim'][:nIms]
    f.close()

    imgs = np.zeros((100, 56, 56, 3)).astype(np.float16)
    for idx in range(100):
        im = Image.fromarray(stim[idx])
        imgs[idx] = np.array(im.resize((56, 56), Image.BILINEAR)).astype(np.float16) / 255

    # feats = imgs.reshape((100, 56 * 56 * 3))

    feats = np.abs(np.fft.fft2(imgs, axes=(1, 2))).reshape((100, 56 * 56 * 3))

    all_feats = np.append(all_feats,feats,axis=0)

os.chdir(homedir)
all_y = np.load('all_y_final.npy')
np.save('all_feats_fft.npy', all_feats) ####
#np.save('all_feats_pixel.npy', all_feats) ####

# delete trials without an embedding vector (no valid response)
aa = np.where(np.isnan(all_y))
all_y2 = np.delete(all_y, np.unique(aa[0]), axis=0)
all_feats2 = np.delete(all_feats, np.unique(aa[0]), axis=0)

# perform PCAs on visual and semantic features
pca_feats = PCA(whiten=True, random_state=2020)
feats_pc = pca_feats.fit_transform(all_feats2) # trials x components
ncomps_feat = np.where(np.cumsum(pca_feats.explained_variance_ratio_)>.9)[0][0]+1
pca_feats = PCA(whiten=True, n_components=ncomps_feat, random_state=2020)
feats_pc = pca_feats.fit_transform(all_feats2) # trials x components
pickle.dump(pca_feats, open('pca_fft_object_final.obj', 'wb'))

pca_embed = PCA(whiten=True, n_components=127, random_state=2020)
y_pc = pca_embed.fit_transform(all_y2) # trials x components

# creation of mapping (weighted sum)
beta = feats_pc.T @ y_pc
np.save('beta_fft_final.npy', beta)

# null permutation distribution of mappings
nperms = 1000
beta_perm = np.zeros((nperms,*np.shape(beta)))
for perm in range(nperms):
    beta_perm[perm] = feats_pc.T @ y_pc[np.random.permutation(np.shape(y_pc)[0])]
np.save('beta_perm_fft_final.npy', beta_perm)

# transform coefficients in z-scores for stats
betaZ = (beta-np.mean(beta_perm,axis=0))/np.std(beta_perm,axis=0)
beta_permZ = (beta_perm-np.mean(beta_perm,axis=0))/np.std(beta_perm,axis=0)

# statistical significance thresholds
qmax = np.quantile(np.amax(beta_permZ.reshape((nperms, ncomps_feat * ncomps_y)), axis=1), .975)
qmin = np.quantile(np.amin(beta_permZ.reshape((nperms, ncomps_feat * ncomps_y)), axis=1), .025)

most_named_words = np.load('most_named_words_final.npy')
vis_comps = np.zeros((len(most_named_words),ncomps_feat))
for idx, word in enumerate(most_named_words): # visual features for each word to reconstruct
    word_embedding = embed_mat[word_to_idx[word],:glove_dim]
    word_comps = pca_embed.transform(np.expand_dims(word_embedding,axis=0))
    vis_comps[idx] = word_comps @ beta.T  # words x vis PC
np.save('most_named_words_fft_comps_final.npy', vis_comps)

vis_comps_perm = np.zeros((nperms,len(most_named_words),ncomps_feat))
for idx, word in enumerate(most_named_words):
    sem_embed = embed_mat[word_to_idx[word],:glove_dim]
    sem_comps = pca_embed.transform(np.expand_dims(sem_embed,axis=0))
    for perm in range(nperms): # do only for 3 different permutations
        vis_comps_perm[perm,idx] = beta_perm[perm] @ sem_comps[0]
np.save('most_named_words_fft_comps_perm_final.npy', vis_comps_perm)