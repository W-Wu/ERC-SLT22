# Distribution-based Emotion Recognition in Conversation
Code for paper "Distribution-based Emotion Recognition in Conversation"  
Please cite:   
> @inproceedings{wu2023distribution,  
  title={Distribution-Based Emotion Recognition in Conversation},  
  author={Wu, Wen and Zhang, Chao and Woodland, Philip C},  
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},  
  pages={860--867},  
  year={2023}, 
  organization={IEEE}  
}

## Data preparation
1. Prepare the input features in a dictionary where keys are the utterance ids and the values are the corresponding numpy array. Example code for finetuning pretrained SSL models can be found on huggingface website (e.g. https://huggingface.co/docs/transformers/training).
2. data_prep_process_label.py -- Process label for IEMOCAP dataset. Prepare majority vote label (hard label) and the sum of original one-hot labels from different evaluators for each sentence.
3. data_prep_diag_order.py -- Create a json file to store the order of utterances in each dialogue. An example of the order file is under "data/order.json".
4. data_prep_organize_in_diag.py -- Process input features and labels into dialogue form.
5. data_prep_split_augment.py -- Split data into traning, validaiton, and test set for leave-one-session-out 5 fold cross validation. Augment dialogue by subsequence randomisation.

## Training and testing
1. pt_model.py -- Model file.
2. pt_param.py -- Parameter settings.
3. pt\_utils.py -- Prepare dataset and dataloader.
4. pt_train.py -- the main training script including the test procedure which will save "AUC-score.npz" for AUPR evaluation.
5. plot_AUPR.py -- plot the PR curve.

