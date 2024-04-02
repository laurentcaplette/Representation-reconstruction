Code for manuscript "Computational reconstruction of mental representations using human behavior" by Caplette & Turk-Browne (psyarxiv.com/7fdvw)

This python code should work on all platforms, however it has only been tested on macOS 11, 12 and 13. 
Various Python libraries are required to run this code:
- robustness 1.2.1
- torch 1.5
- torchvision 0.6.0
- scikit-learn 0.23.2
- numpy 1.21.5
- symspellpy 6.7.0
- scipy 1.7.3
- nltk 3.7

No installation is required, one can simply run the provided Python source code.

To test the analysis pipeline, one can run the code on the provided data, after being careful to adapt the paths in the source code files. Most analyses should only take a few minutes, except the image syntheses which can take a few hours when run on a gpu (and should not be run on a cpu because it would be prohibitively computationally expensive).

Brief description of each script:
- DNN_noise_naming: JS code for main experiment
- DNN_noise_naming_ind: JS code for individual experiment
- prepare_validation: prepare files for validation #1
- prepare_validation2: prepare files for validation #2
- prepare_validation3: prepare files for validation #3
- DNN_noise_validation_final: JS code for validation #1
- DNN_noise_validation2_final: JS code for validation #2
- DNN_noise_validation3_final: JS code for validation #3
- collect_imagenet_act: collect DNN activations to ImageNet images prior to creating stimuli
- stim_creation: create stimuli
- mapping_creation: regression analyses, create the visual-semantic mapping
- mapping_noword: create visual-semantic mappings without the label of the concept being targeted for reconstruction
- mapping_noembed: create visual-semantic mappings without using a semantic embedding
- feature_vis: visualize DNN features
- model_analyses: perform analyses (visual-semantic mapping creation and representation reconstructions) on DNN
- prediction_analyses: prediction of semantic content perceived in held-out stimuli
- rsa_analyses: RSA analyses of representations using behavioral similarity judgments
- indiv_analyses: analyses of individual experiment data
- validation1_analyses: analyses of validation #1 data
- validation2_analyses: analyses of validation #2 data
- validation3_analyses: analyses of validation #3 data
