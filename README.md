# ERC-IS22
Code for paper "Distribution-based Emotion Recognition in Conversation"
Please cite: 

## Data preparation
1. Prepare the input features in the form of a dictionary. The keys are the utterance names and the values are the corresponding numpy arrays.
2. data_prep_process_label.py Generate hard label and soft label for IEMOCAP dataset. And also create a json file to store the order of utterances in each dialogue using data_prep_diag_order. An example of the order file is in 'data/order.json'.
3. data_prep_organize_in_diag.py Process input features and labels into dialogue form.
4. data_prep_split_augment.py Split data into traning, validaiton, and test set. Also perform sub-sequence randomisation for data augmentation. gives train.scp, cv.scp, test.scp

## Training and testing
1. pt_train.py is the main training file
2. can use plot_AUPR.py to plot the PR curve.

