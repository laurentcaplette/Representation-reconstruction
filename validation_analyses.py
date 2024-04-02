# Analysis of validation data
# Copyright (c) 2021 Laurent Caplette

import os, csv, re
import numpy as np

np.random.seed()

homedir = ''  ### path of home directory
os.chdir(homedir+'/final_validation_results')

pattern = '[0-9]{1,3}_DNN.+csv'
resfiles = []
for file in os.listdir():
    match = re.fullmatch(pattern,file)
    if match:
        resfiles.append(file)

correct_all = np.zeros((len(resfiles), 350))
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

    resp = np.array(column['key_resp_4.keys']) # subject responses
    rt = np.array(column['key_resp_4.rt'])
    label_left = np.array(column['label_left'])
    label_right = np.array(column['label_right'])
    im_nb = np.array(column['im_nb']) # image IDs
    ts = np.array(column['ts']) # correct responses

    resp = np.delete(resp, np.where(ts == ''))
    rt = np.delete(rt, np.where(ts == ''))
    label_left = np.delete(label_left, np.where(ts == ''))
    label_right = np.delete(label_right, np.where(ts == ''))
    im_nb = np.delete(im_nb, np.where(ts == ''))
    ts = np.delete(ts, np.where(ts == ''))

    # rt = rt.astype(np.float32)
    im_nb = im_nb.astype(np.int16)
    ts = ts.astype(np.bool)

    resp = [r[-3] for r in resp] # [-3] takes last response (important when multiple keypresses)
    resp = np.array([0 if ii is "a" else 1 for ii in resp]) # left==0, right==1

    # reorder variables in image ID order (same for all subjects)
    order = np.argsort(im_nb)
    ts = ts[order]
    label_left = label_left[order]
    label_right = label_right[order]
    resp = resp[order]
    rt = rt[order]

    correct = resp == ts # accuracy

    correct_all[file_idx] = correct

suj_means = np.mean(correct_all,axis=1) # accuracy for each subject
suj_meansZ = (suj_means-np.mean(suj_means))/np.std(suj_means)
outliers = [list(*np.where(suj_meansZ>3)), list(*np.where(suj_meansZ<-3))]
outliers = [item for sublist in outliers for item in sublist] # all outlier subjects

correct_all = np.delete(correct_all,outliers,axis=0) # remove subjects 3stds from mean

os.chdir(homedir)
labels = np.load('final_validation_labels.npy')

mn_words = np.load('most_named_words_final.npy')
visgen_words = np.load('most_common_visgen_final.npy')
ind = np.where([word in mn_words for word in visgen_words])[0] # which MostNamed words are also in VisGen
visgen_ind = [*ind,*np.arange(100)+250] # which labels are in VisGen

obj_means = np.mean(correct_all,axis=0)

barely_named_mean = np.mean(correct_all[:,250:])
most_named_mean = np.mean(correct_all[:,:250])
visgen_mean = np.mean(correct_all[:,visgen_ind])

# stats (shuffle subjects for random-effects stats, correct for objects with Sidak)
nsuj = np.shape(correct_all)[0]
nlabels = 350
nboot = 50000 # a lot, for multiple comparison correction
correct_boot_all = np.zeros((nboot,nsuj,nlabels))
for boot in range(nboot):
    correct_boot_all[boot] = correct_all[np.random.choice(nsuj,nsuj)]
CIbnd = np.quantile(np.mean(correct_boot_all,axis=1),1-(1-.05)**(1/nlabels),axis=0) # sidak correction
nboot2 = 500000 # even more, to be sure, for these objects which are close to 50% (don't save all bootstraps)
limitobj = np.where(np.logical_and(CIbnd<.6,CIbnd>.4))[0] # previously estimated CI boundary is between 40% and 60% acc
nlimitobj = np.size(limitobj)
CIbndtemp = np.zeros(nlimitobj)
for obj in range(nlimitobj):
    print(obj)
    correct_boot_temp = np.zeros((nboot2,nsuj))
    correct_all_temp = correct_all[:,limitobj[obj]]
    for boot in range(nboot2):
        correct_boot_temp[boot] = correct_all_temp[np.random.choice(nsuj,nsuj)]
    CIbndtemp[obj] = np.quantile(np.mean(correct_boot_temp,axis=1),1-(1-.05)**(1/nlabels),axis=0)
CIbnd[limitobj] = CIbndtemp # for these objects, replace bnd with one more precisely estimated
nsig_obj = np.sum(CIbnd>.5) # nb of signif objects

goodlabels = labels[np.quantile(np.mean(correct_boot_all,axis=1),1-(1-.05)**(1/nlabels),axis=0)>.5]
badlabels = labels[np.quantile(np.mean(correct_boot_all,axis=1),1-(1-.05)**(1/nlabels),axis=0)<=.5]

np.save('final_validation_acc.npy', correct_all)



