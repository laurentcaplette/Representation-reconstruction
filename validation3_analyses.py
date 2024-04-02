import numpy as np
import nltk
import csv, os, re
import h5py, json, pickle
import pkg_resources
from symspellpy import SymSpell, Verbosity
import inflect
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

np.random.seed() # randomly initialize the seed

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7) # for correction of  spelling mistakes
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

eng = inflect.engine()

stop_words = nltk.corpus.stopwords.words('english')

def plural_or_same(word):
    plural_word = eng.plural_noun(word)
    if plural_word==False:
        plural_word = word
    return plural_word

def singular_or_same(word):
    singular_word = eng.singular_noun(word)
    if singular_word==False:
        singular_word = word
    return singular_word

homedir = '' ### path of home directory
os.chdir(homedir)

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

mn_words = np.load('most_named_words_final.npy')

os.chdir(homedir + '/DNN_noise_validation3_expt/html/data_final')  #######

ntrials = 180
nperms = 1000

# which files are results
pattern = '[0-9]{1,3}_DNN.+csv'
resfiles = []
for file in os.listdir():
    match = re.fullmatch(pattern, file)
    if match:
        resfiles.append(file)

# EXTRACT AND CORRECT RESPONSES
pnbs = []
all_words = np.zeros((len(resfiles), 4, int(ntrials / 4)), dtype='object')
word_list = []
for file_idx, filename in enumerate(resfiles):  # for each participant/file

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

    label = np.array(column['label2'])
    pnbs = np.append(pnbs, int(column['participant'][1]))  # participant nb
    imnbs = np.array(column['im_nb'])
    recons = np.array(column['recon_nb'])
    conds = np.array(column['cond'])
    permnbs = np.array(column['permID'])

    # delete rows without data
    label = np.delete(label, np.arange(36, 180, 37))
    imnbs = np.delete(imnbs, np.arange(36, 180, 37))
    recons = np.delete(recons, np.arange(36, 180, 37))
    conds = np.delete(conds, np.arange(36, 180, 37))
    permnbs = np.delete(permnbs, np.arange(36, 180, 37))

    imnbs = np.array([int(nb) for nb in imnbs])
    recons = np.array([int(nb) for nb in recons])
    conds = np.array([int(nb) for nb in conds])
    permnbs = np.array([int(nb) for nb in permnbs])

    all_words_temp = [[] for ii in range(ntrials)]
    for trial_nb in range(ntrials):

        response = label[trial_nb].lower()  # make lowercase
        tokens = nltk.word_tokenize(response)  # tokenize everything in words
        words = [w for w in tokens if not w in stop_words]  # reject if stopword
        words = [w for w in words if w not in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                                               'ten']]  # reject if number

        # correct words if necessary
        for idx, word in enumerate(words):

            # wordcount[file_idx] += 1

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

            if (word not in word_to_idx) or (len(word) < 2):  # word must be recognized by embedding and min. 2 letters
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
                    for suggestion in suggestion_terms:  # if first suggestion is not in vocab, try next (in decreasing frequency)
                        if suggestion in word_to_idx:
                            words[idx] = suggestion
                            accepted = True
                            break
                        else:
                            continue
                if not accepted:  # if word too short or impossible to correct, remove it
                    words[idx] = ''
                    # badwordcount[file_idx] += 1

        # if there's more than one word (after stopword & nonword removal), join them if it's recognized by embedding
        if len(words) > 1:
            if '-'.join(words) in word_to_idx:
                words = ['-'.join(words)]
            elif ('-'.join(words)).capitalize() in word_to_idx:
                words = [('-'.join(words)).capitalize()]
            elif '-'.join([w.capitalize() for w in words]) in word_to_idx:
                words = ['-'.join([w.capitalize() for w in words])]

        words = [w for w in words if w != '']

        all_words_temp[trial_nb] = words

        for word in words:
            word_list.append(word)

    all_words_temp = np.array(all_words_temp, dtype='object')
    all_words_temp = all_words_temp.flatten()
    for perm in range(4):  # 0:real, 1-3:perm
        all_words_cond = all_words_temp[permnbs == perm]
        ind = np.argsort(recons[permnbs == perm])
        all_words[file_idx, perm] = all_words_cond[ind]  # reorder in same order for all conds

recon_words = mn_words[np.unique(recons)]

word_idx = np.zeros((25,4,45),dtype='object')
for p in range(25):
    for c in range(4):
        for r in range(45):
            rword = recon_words[r]
            words = all_words[p,c,r]
            nwords = np.size(words)
            word_idx_temp = np.zeros(nwords).astype(int)
            if isinstance(words,str):
                words = [words]
            for w, word in enumerate(words):
                word = singular_or_same(word)
                word_idx_temp[w] = np.where(possible_words_sing==word)[0][0]
            word_idx[p,c,r] = word_idx_temp

## NB OF CONCEPTS WITH GOOD RESPONSES FOR REAL AND NULL RECONSTRUCTIONS
isitright = np.zeros((25,4,45))
for p in range(25):
    for c in range(4):
        for r in range(45):
            rword = recon_words[r]
            words = all_words[p,c,r]
            nwords = np.size(words)
            isitright_temp = np.zeros(nwords).astype(bool)
            if isinstance(words,str):
                words = [words]
            for w, word in enumerate(words):
                ss = singular_or_same(word)
                pp = plural_or_same(word)
                isitright_temp[w] = np.any([word==rword,ss==rword,pp==rword])
            isitright[p,c,r] = np.any(isitright_temp)

n_good1 = np.sum(np.sum(isitright[:,0],0)>np.sum(np.mean(isitright[:,1:4],1),0))

# Bootstrap for confidence interval
isitright_boot = np.zeros((nperms, 45))
for boot in range(nperms):
    temp = np.sum(isitright[:, 0], 0) > np.sum(np.mean(isitright[:, 1:4], 1), 0)
    isitright_boot[boot] = temp[np.random.choice(45, 45)]

n_good1_CI = np.quantile(np.sum(isitright_boot, 1), [0.025, 0.975]) # 95% C.I.

# Permutations for significance test
all_words_perm = np.zeros((nperms, 25, 4, 45), dtype='object')
for perm in range(nperms):
    order = np.random.permutation(45)
    all_words_perm[perm] = all_words[:, :, order]

isitright_perm = np.zeros((nperms, 25, 4, 45))
for perm in range(nperms):
    for p in range(25):
        for c in range(4):
            for r in range(45):
                rword = recon_words[r]
                words = all_words_perm[perm, p, c, r]
                nwords = np.size(words)
                isitright_temp = np.zeros(nwords).astype(bool)
                if isinstance(words, str):
                    words = [words]
                for w, word in enumerate(words):
                    ss = singular_or_same(word)
                    pp = plural_or_same(word)
                    isitright_temp[w] = np.any([word == rword, ss == rword, pp == rword])
                isitright_perm[perm, p, c, r] = np.any(isitright_temp)

n_good1_thresh = np.quantile(np.sum(np.sum(isitright_perm[:,:,0],1)>np.sum(np.mean(isitright_perm[:,:,1:4],2),1),-1),0.95) # stat threshold

## ENTROPY OF RESPONSES FOR REAL AND NULL RECONSTRUCTIONS
E = np.zeros((5, 45))
for r in range(45):
    counts_all = [[] for ii in range(4)]
    for c in range(4):
        unique, counts = np.unique(np.concatenate(word_idx[:, c, r]), return_counts=True)
        counts_all[c] = counts
        pk = counts / np.sum(counts)
        E[c, r] = -np.sum(pk * np.log2(pk))

n_good2 = np.sum((E[0] - np.mean(E[1:4], axis=0)) < 0)

# Bootstrap for confidence interval
E_boot = np.zeros((nperms, 45))
for boot in range(nperms):
    temp = (E[0] - np.mean(E[1:4], 0)) < 0
    E_boot[boot] = temp[np.random.choice(45, 45)]

n_good2_CI = np.quantile(np.sum(E_boot,1),[0.025, 0.975])

# Permutations for significance test
word_idx_r = word_idx.flatten()

E_perm = np.zeros((nperms,5,45))
for perm in range(nperms):
    #order = np.random.permutation(4*45)
    #word_idx_perm = word_idx_r[:,order].reshape((25,4,45))
    order = np.random.permutation(25*4*45)
    word_idx_perm = word_idx_r[order].reshape((25,4,45))
    for r in range(45):
        counts_all = [[] for ii in range(4)]
        for c in range(4):
            unique, counts = np.unique(np.concatenate(word_idx_perm[:,c,r]), return_counts=True)
            counts_all[c] = counts
            pk = counts/np.sum(counts)
            E_perm[perm,c,r] = -np.sum(pk * np.log2(pk))

n_good2_thresh = np.quantile(np.sum((E_perm[:, 0] - np.mean(E_perm[:, 1:4], axis=1)) < 0, axis=-1), 0.95) # stat threshold

## SEMANTIC DISTANCE TO CORRECT LABEL AND SEMANTIC CONSISTENCY FOR REAL AND NULL RECONSTRUCTIONS
all_d = np.zeros((25, 4, 45), dtype='object')
all_embed = np.zeros((25, 4, 45), dtype='object')
for p in range(25):
    for c in range(4):
        for r in range(45):
            rword = recon_words[r]
            rembed = np.expand_dims(embed_mat[word_to_idx[rword], :glove_dim], 0)
            words = all_words[p, c, r]
            if isinstance(words, str):
                words = [words]
            d = np.zeros(len(words))
            embeds = np.zeros((len(words), glove_dim))
            for w, word in enumerate(words):
                embed = np.expand_dims(embed_mat[word_to_idx[word], :glove_dim], 0)
                d[w] = 1 - cos_sim(embed, rembed)
                embeds[w] = embed
            all_d[p, c, r] = d
            all_embed[p, c, r] = embeds

## average semantic distance to correct label
dmeans = np.zeros((4, 45))
for c in range(4):
    for r in range(45):
        dmeans[c, r] = np.mean(np.concatenate(all_d[:, c, r]))

n_good3 = np.sum(dmeans[0]-np.mean(dmeans[1:],0)<0)

# bootstrap for confidence interval
dmeans_boot = np.zeros((nperms, 45))
for boot in range(nperms):
    temp = (dmeans[0] - np.mean(dmeans[1:4], 0)) < 0
    dmeans_boot[boot] = temp[np.random.choice(45, 45)]

n_good3_CI = np.quantile(np.sum(dmeans_boot,1),[0.025, 0.975])

# permutations for significance testing
all_words_perm = np.zeros((nperms, 25, 4, 45), dtype='object')
for perm in range(nperms):
    order = np.random.permutation(45)
    all_words_perm[perm] = all_words[:, :, order]

all_d_perm = np.zeros((nperms, 25, 4, 45), dtype='object')
for perm in range(nperms):
    for p in range(25):
        for c in range(4):
            for r in range(45):
                rword = recon_words[r]
                rembed = np.expand_dims(embed_mat[word_to_idx[rword], :glove_dim], 0)
                words = all_words_perm[perm, p, c, r]
                if isinstance(words, str):
                    words = [words]
                d = np.zeros(len(words))
                for w, word in enumerate(words):
                    embed = np.expand_dims(embed_mat[word_to_idx[word], :glove_dim], 0)
                    d[w] = 1 - cos_sim(embed, rembed)
                all_d_perm[perm, p, c, r] = d

dmeans_perm = np.zeros((nperms, 4, 45))
for c in range(4):
    for r in range(45):
        for perm in range(nperms):
            dmeans_perm[perm, c, r] = np.mean(np.concatenate(all_d_perm[perm, :, c, r]))

n_good3_thresh = np.quantile(np.sum(dmeans_perm[:, 0] - np.mean(dmeans_perm[:, 1:], 1) < 0, -1), 0.95)  # stat threshold

## semantic consistency of responses (trace of covariance matrix)
evar = np.zeros((4, 45))
for c in range(4):
    for r in range(45):
        evar[c, r] = np.sum(np.var(np.concatenate(all_embed[:, c, r]), 0))

n_good4 = np.sum(evar[0]-np.mean(evar[1:],0)<0)

# bootstrap for confidence interval
evar_boot = np.zeros((nperms, 45))
for boot in range(nperms):
    temp = (evar[0] - np.mean(evar[1:4], 0)) < 0
    evar_boot[boot] = temp[np.random.choice(45, 45)]

n_good4_CI = np.quantile(np.sum(evar_boot,1),[0.025, 0.975])

# permutations for significance testing
all_words_r = all_words.flatten()
all_words_perm = np.zeros((nperms, 25, 4, 45), dtype='object')
for perm in range(nperms):
    order = np.random.permutation(25 * 4 * 45)
    all_words_perm[perm] = all_words_r[order].reshape((25, 4, 45))

all_embed_perm = np.zeros((nperms, 25, 4, 45), dtype='object')
for perm in range(nperms):
    for p in range(25):
        for c in range(4):
            for r in range(45):
                rword = recon_words[r]
                rembed = np.expand_dims(embed_mat[word_to_idx[rword], :glove_dim], 0)
                words = all_words_perm[perm, p, c, r]
                if isinstance(words, str):
                    words = [words]
                embeds = np.zeros((len(words), glove_dim))
                for w, word in enumerate(words):
                    embed = np.expand_dims(embed_mat[word_to_idx[word], :glove_dim], 0)
                    embeds[w] = embed
                all_embed_perm[perm, p, c, r] = embeds

evar_perm = np.zeros((nperms, 4, 45))
for c in range(4):
    for r in range(45):
        for perm in range(nperms):
            evar_perm[perm, c, r] = np.sum(np.var(np.concatenate(all_embed_perm[perm, :, c, r]), 0))

n_good4_thresh = np.quantile(np.sum(evar_perm[:, 0] - np.mean(evar_perm[:, 1:], 1) < 0, -1), 0.95) # stat threshold

np.save('is_label_correct.npy', isitright)
np.save('is_label_correct_perm.npy', isitright_perm)
np.save('resp_entropy.npy', E)
np.save('resp_entropy_perm.npy', E_perm)
np.save('avg_dist_to_correct.npy', dmeans)
np.save('avg_dist_to_correct_perm.npy', dmeans_perm)
np.save('resp_sem_var.npy', evar)
np.save('resp_sem_var_perm.npy', evar_perm)

