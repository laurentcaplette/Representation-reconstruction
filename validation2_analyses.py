import numpy as np
import nltk
import csv, os, re
import h5py, json, pickle
import pkg_resources
from symspellpy import SymSpell, Verbosity
import inflect

np.random.seed() # randomly initialize the seed

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7) # for correction of  spelling mistakes
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

p = inflect.engine()

def plural_or_same(word):
    plural_word = p.plural_noun(word)
    if plural_word==False:
        plural_word = word
    return plural_word

def singular_or_same(word):
    singular_word = p.singular_noun(word)
    if singular_word==False:
        singular_word = word
    return singular_word

homedir = '' ### path of home directory
os.chdir(homedir)

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

mn_words = np.load('most_named_words_final.npy')

os.chdir(homedir + '/DNN_noise_validation2_expt/html/data')

ntrials = 100

# which files are results
pattern = '[0-9]{1,3}_DNN.+csv'
resfiles = []
for file in os.listdir():
    match = re.fullmatch(pattern, file)
    if match:
        resfiles.append(file)

# retrieve all data, correct mistakes, transform to vectors
pnbs = np.array([])
all_y = np.empty((0, glove_dim), dtype=np.float32)
all_imnbs = []
all_words = []
all_im_words = [[] for ii in range(5000)]
all_suj_im_words = [[] for ii in range(len(resfiles))]
all_suj_im_nbs = [[] for ii in range(len(resfiles))]
wordcount = np.zeros(100)
badwordcount = np.zeros(100)
imcount = -1
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
    pnbs = np.append(pnbs, int(column['participant'][1]))  # participant nb
    imnbs = np.array(column['im_nb'])

    # delete rows without data
    label0 = np.delete(label0, [20, 41, 62, 83])
    label1 = np.delete(label1, [20, 41, 62, 83])
    label2 = np.delete(label2, [20, 41, 62, 83])
    imnbs = np.delete(imnbs, [20, 41, 62, 83])

    imnbs = [int(nb) for nb in imnbs]

    all_suj_im_words[file_idx] = [[] for ii in range(ntrials)]
    for trial_nb in range(ntrials):

        imcount += 1

        embed = np.empty((0, glove_dim))
        for lbl in range(3):

            label = eval('label' + str(lbl))

            responses = label[trial_nb]

            responses2 = responses.lower()  # make lowercase
            tokens = nltk.word_tokenize(responses2)  # tokenize everything in words
            words = [w for w in tokens if not w in stop_words]  # reject if stopword
            words = [w for w in words if
                     w not in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                               'ten']]  # reject if number

            # correct words if necessary
            for idx, word in enumerate(words):

                wordcount[file_idx] += 1

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

                if (word not in word_to_idx) or (
                        len(word) < 2):  # word must be recognized by embedding and min. 2 letters
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

            all_suj_im_nbs[file_idx] = imnbs

            # convert responses to embedding vectors
            for word in words:
                embed = np.append(embed, np.expand_dims(embed_mat[word_to_idx[word], :glove_dim], axis=0), axis=0)
                all_im_words[imcount].append(word)
                all_suj_im_words[file_idx][trial_nb].append(word)
                all_words.append(word)

        all_y = np.append(all_y, np.mean(embed, axis=0, keepdims=True), axis=0)
    all_imnbs = np.append(all_imnbs, imnbs)

# order words for each subject
ordered_im_words = [[] for ii in range(len(resfiles))]
for suj in range(len(resfiles)):
    im_idx = [np.where([all_suj_im_nbs[suj][jj] == ii for jj in range(100)])[0][0] for ii in range(100)]
    ordered_im_words[suj] = [all_suj_im_words[suj][ii] for ii in im_idx]

np.save('ordered_im_words.npy', np.array(ordered_im_words,dtype='object'))

# shuffle across trials for stat testing
nperms = 1000
perm_words = [[] for ii in range(nperms)]
for perm in range(nperms):
    idx = np.random.permutation(100)
    perm_words[perm] = [[] for ii in range(len(resfiles))]
    for suj in range(len(resfiles)):
        perm_words[perm][suj] = [ordered_im_words[suj][ii] for ii in idx]

# check if right word is most common response for each recon, do with shuffled words as well for stats
isitmax = np.zeros(100)
isitmax_perm = np.zeros((nperms, 100))
thefreq = np.zeros(100)
thefreq_perm = np.zeros((nperms, 100))
ordered_im_words2 = [[] for ii in range(100)]
perm_words2 = [[[] for jj in range(100)] for ii in range(nperms)]
for the_im_nb in range(100):

    the_word = mn_words[the_im_nb]
    the_ss = singular_or_same(the_word)
    the_pp = plural_or_same(the_word)

    for suj in range(len(resfiles)):
        [ordered_im_words2[the_im_nb].append(word) for word in ordered_im_words[suj][the_im_nb]]
    wordlist = np.unique(ordered_im_words2[the_im_nb])
    wordfreq = np.zeros(len(wordlist))
    for widx, word in enumerate(wordlist):
        ss = singular_or_same(word)
        pp = plural_or_same(word)
        isitthere = np.zeros((len(resfiles)))
        for suj in range(len(resfiles)):
            isitthere[suj] = np.any(
                [((ii == word) or (ii == ss) or (ii == pp)) for ii in ordered_im_words[suj][the_im_nb]])
        wordfreq[widx] = np.sum(isitthere)
    whereidx = np.where(wordfreq == np.amax(wordfreq))[0]
    tv = []
    for idx in whereidx:
        ww = wordlist[idx]
        tv.append((ww == the_word) or (ww == the_ss) or (ww == the_pp))
    isitmax[the_im_nb] = np.any(np.array(tv))
    tf = wordfreq[wordlist == the_word]  # includes singular/plural
    if np.any(tf):
        thefreq[the_im_nb] = tf
    else:
        thefreq[the_im_nb] = 0

    for perm in range(nperms):
        for suj in range(len(resfiles)):
            [perm_words2[perm][the_im_nb].append(word) for word in perm_words[perm][suj][the_im_nb]]
        wordlist = np.unique(perm_words2[perm][the_im_nb])
        wordfreq = np.zeros(len(wordlist))
        for widx, word in enumerate(wordlist):
            ss = singular_or_same(word)
            pp = plural_or_same(word)
            isitthere = np.zeros((len(resfiles)))
            for suj in range(len(resfiles)):
                isitthere[suj] = np.any(
                    [((ii == word) or (ii == ss) or (ii == pp)) for ii in perm_words[perm][suj][the_im_nb]])
            wordfreq[widx] = np.sum(isitthere)
        whereidx = np.where(wordfreq == np.amax(wordfreq))[0]
        tv = []
        for idx in whereidx:
            ww = wordlist[idx]
            tv.append((ww == the_word) or (ww == the_ss) or (ww == the_pp))
        isitmax_perm[perm][the_im_nb] = np.any(np.array(tv))
        tf = wordfreq[wordlist == the_word]
        if np.any(tf):
            thefreq_perm[perm][the_im_nb] = tf
        else:
            thefreq_perm[perm][the_im_nb] = 0

# bootstrap for confidence interval
isitmax_boot = np.zeros(nperms)
for boot in range(nperms):
    order = np.random.choice(100,100)
    isitmax_boot[boot] = np.sum(isitmax[order])

np.save('thefreq.npy', thefreq)
np.save('thefreq_perm.npy', thefreq_perm)