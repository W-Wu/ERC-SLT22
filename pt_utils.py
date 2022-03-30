import torch
import numpy as np
import os
import random
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import Namespace
import torch.optim as optim
import time
import shutil
from transformers.optimization import AdamW,get_scheduler,get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error,accuracy_score,f1_score,balanced_accuracy_score,recall_score
from sklearn.metrics import average_precision_score
import pandas as pd
import json
import math
from itertools import chain
import sys
from scipy.stats import entropy


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def sequential_collate(batches):
    seqs = []
    i=0
    for data_seq in zip(*batches):
        min_len=min([len(x) for x in data_seq])
        data_seq=[x[:min_len] for x in data_seq]
        if isinstance(data_seq[0],(torch.Tensor, np.ndarray)):
            data_seq=torch.stack(data_seq)

        seqs.append(data_seq)
    return seqs

def pad(tensorlist, batch_first=True, padding_value=0.):
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist,batch_first=batch_first,padding_value=padding_value)
    return padded_seq

def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None,longest_first=False):
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        if mega_batch_mult == 0:
            mega_batch_mult = 1
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    if longest_first:
        megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) for megabatch in megabatches]
    else:
        megabatches = [list(sorted(megabatch, key=lambda i: lengths[i])) for megabatch in megabatches]
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]
    return [i for megabatch in megabatches for i in megabatch]

class LengthGroupedSampler(data.Sampler):
    def __init__(self,batch_size: int,dataset= None,lengths= None,mega_batch_mult=None,generator=None,longest_first=False):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        self.batch_size = batch_size
        self.lengths = lengths
        self.generator = generator
        self.mega_batch_mult=mega_batch_mult
        self.longest_first=longest_first

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size,mega_batch_mult=self.mega_batch_mult,generator=self.generator,longest_first=self.longest_first)
        return iter(indices)

class DATAset(data.Dataset):
    def __init__(self,scp_file):     
        self.label_hard = np.load('./data/IEMOCAP-hardlabel-diag.npy',allow_pickle=True).item()
        self.label = np.load('./data/IEMOCAP-softlabel-sum-diag.npy',allow_pickle=True).item()
        self.w2v2 = np.load('./data/w2v2-ft-avg-diag.npy',allow_pickle=True).item()
        self.bert = np.load('./data/bert-base-diag.npy',allow_pickle=True).item()
        f=open('./data/order.json','r')
        self.paras=json.loads(f.read())
        self.scp_file = scp_file

    def __len__(self):
        return len(self.scp_file)

    def __getitem__(self, idx):
        ses_id,start,end = self.scp_file[idx]
        w2v2 = self.w2v2[ses_id][start:end,:]
        label_sum = self.label[ses_id][start:end]
        label_avg = np.array([row/sum(row) for row in label_sum])
        name =  self.paras[ses_id][start:end]
        label_hard = self.label_hard[ses_id][start:end]
        bert = self.bert[ses_id][start:end,:]
        return torch.from_numpy(w2v2).float(), torch.from_numpy(bert).float(),torch.from_numpy(label_avg).float(),torch.from_numpy(label_hard).float(),torch.from_numpy(label_sum).float(),name  

def prep_dataloader(scp_train,scp_cv,scp_test):
    train_list = pd.read_csv(scp_train,delimiter='\t',header=0).values.tolist()
    cv_list = pd.read_csv(scp_cv,delimiter='\t',header=0).values.tolist()
    test_list = pd.read_csv(scp_test,delimiter='\t',header=0).values.tolist()

    Train_dataset=DATAset(train_list)
    Val_dataset=DATAset(cv_list)
    Test_dataset=DATAset(test_list)

    lengths = [len(x[0]) for x in Train_dataset]
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    train_sampler=LengthGroupedSampler(param.train_batch_size,dataset=Train_dataset,lengths=lengths,generator=generator,longest_first=True)    #,mega_batch_mult=max(lengths)

    trainloader = DataLoader(Train_dataset,batch_size=param.train_batch_size,sampler=train_sampler,collate_fn=sequential_collate,drop_last=True)
    valloader = DataLoader(Val_dataset, batch_size=1,collate_fn=sequential_collate,shuffle=False)
    testloader = DataLoader(Test_dataset, batch_size=1,collate_fn=sequential_collate,shuffle=False)

    return trainloader,valloader,testloader


