import os
import sys
import time
import json
import math
import shutil
import random
import numpy as np
import pandas as pd
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler,get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.metrics import average_precision_score
from scipy.stats import entropy

from pt_utils import *
from pt_model import *
from pt_param import param

def remove_label_padding(o,t,padding_value):
    assert o.shape[1] == t.shape[1],(o.shape,t.shape)
    t_4 = torch.Tensor([]).to(device)
    o_4 = torch.Tensor([]).to(device)
    for b,B in enumerate(t):
        tmp_idx = [i for i,x in enumerate(B) if padding_value not in x]
        tmp_t=B[tmp_idx]
        tmp_o=o[b][tmp_idx]
        t_4=torch.cat((t_4,tmp_t),0)
        o_4=torch.cat((o_4,tmp_o),0)
    return o_4,t_4

def pick_majority_unique(o,t,n_class):
    assert o.shape[1] == t.shape[1],(o.shape,t.shape)
    t_4 = torch.Tensor([]).to(device)
    o_4 = torch.Tensor([]).to(device)
    for b,B in enumerate(t):
        tmp_idx = [i for i,x in enumerate(B) if 0<=x<n_class]
        tmp_t=B[tmp_idx]
        tmp_o=o[b][tmp_idx]
        t_4=torch.cat((t_4,tmp_t),0)
        o_4=torch.cat((o_4,tmp_o),0)
    return o_4,t_4

def train_batch(i,data,filter4=False):
    w2v2_batch,bert_batch,target,target_hard,target_sum,fname = data

    w2v2_batch = w2v2_batch.to(device)
    bert_batch = bert_batch.to(device)
    target_hard = target_hard.to(device)
    target = target.to(device)
    target_sum = target_sum.to(device)
    
    B,seq_len,num_dim = target.size()
    total_num_step = len(trainloader)*param.EPOCH
    if param.forcing_decay_type:
        if param.forcing_decay_type == 'linear':
            forcing_ratio = max(0, param.forcing_ratio - param.forcing_decay * i/total_num_step)
        elif param.forcing_decay_type == 'exp':
            forcing_decay = 0.01**(1/total_num_step)
            forcing_ratio = param.forcing_ratio * (forcing_decay ** i)
        elif param.forcing_decay_type == 'sigmoid':
            k=total_num_step/10
            forcing_ratio = param.forcing_ratio * k / (k + math.exp(i / k))
        else:
            raise ValueError('Unrecognized forcing_decay_type: ' + param.forcing_decay_type)
    else:
        forcing_ratio = param.forcing_ratio

    tf_ratio.append(forcing_ratio)
    outputs = model(w2v2_batch,bert_batch,target,forcing_ratio=forcing_ratio)
    # outputs = model(w2v2_batch,bert_batch,target_hard.unsqueeze(-1),forcing_ratio=forcing_ratio)    # for hard system
    
    op_6,tgt_soft=remove_label_padding(outputs,target,-1)
    op_6,tgt_dpn=remove_label_padding(outputs,target_sum,-1)
    fname=list(chain.from_iterable(fname))
    assert len(op_6)==len(fname),(op_6.shape,len(fname))

    op_5,tgt_5=pick_majority_unique(outputs,target_hard,5)
    if filter4:
        op_4,tgt_4=pick_majority_unique(outputs,target_hard,4)
        return op_6,tgt_soft,tgt_dpn,fname,op_5,tgt_5,op_4,tgt_4
    else:
        return op_6,tgt_soft,tgt_dpn,fname,op_5,tgt_5

def decode(data,filter4=False):
    w2v2_batch,bert_batch,target,target_hard,target_sum,fname = data

    w2v2_batch = w2v2_batch.to(device)
    bert_batch = bert_batch.to(device)
    target_hard=target_hard.to(device)
    target = target.to(device)
    target_sum = target_sum.to(device)
    outputs = model(w2v2_batch,bert_batch,target,forcing_ratio=0)    
    # outputs = model(w2v2_batch,bert_batch,target_hard.unsqueeze(-1),forcing_ratio=0)    # for hard system
    op_5,tgt_5=pick_majority_unique(outputs,target_hard,5)

    fname=list(chain.from_iterable(fname))
    assert len(target.reshape(-1,param.output_dim))==len(fname),(target.shape,len(fname))

    op_6 = outputs.reshape(-1,param.output_dim)
    tgt_soft = target.reshape(-1,param.output_dim)
    tgt_sum = target_sum.reshape(-1,param.output_dim)
    if filter4:
        op_4,tgt_4=pick_majority_unique(outputs,target_hard,4)
        return op_6,tgt_soft,tgt_sum,fname,op_5,tgt_5,op_4,tgt_4
    else:
        return op_6,tgt_soft,tgt_sum,fname,op_5,tgt_5

class Dir_loss(nn.Module):
    def __init__(self,reduction="mean",use_smooth=True):
        super(Dir_loss,self).__init__()   
        self.reduction=reduction
        self.smooth=use_smooth

    def forward(self,outputs, targets, smooth=1e-6, epsilon=1e-6):
        batch_size = targets.shape[0]
        n_class = targets.shape[1]
                           
        outputs_extend = torch.unsqueeze(outputs, 1).expand([batch_size, n_class, n_class])
        if self.smooth==True:
            pseudo_label = torch.diag_embed(torch.ones([n_class,]))*(1.0-float(n_class)*smooth) + torch.ones([n_class, n_class,])*smooth # label smooth [C, C]
        else:
            pseudo_label = torch.diag_embed(torch.ones([n_class,]))*(1.0-smooth) + torch.ones([n_class, n_class,])*smooth # if not smooth 
        diag_label = torch.unsqueeze(pseudo_label,0).expand(batch_size, n_class, n_class)
        diag_label=diag_label.to(device)
        
        alphas = torch.exp(outputs_extend)+epsilon
        alphas=alphas.to(device)
        if self.reduction == "mean":
            loss = -torch.mean( torch.sum(   
                        targets * (torch.lgamma(torch.sum(alphas, axis=2)) - torch.sum(torch.lgamma(alphas), axis=2) + torch.sum((alphas-1.0)*torch.log(diag_label.detach()),
                        axis=2)), axis=1))
        if self.reduction == "sum":
            loss = -torch.sum( torch.sum(   
                        targets * (torch.lgamma(torch.sum(alphas, axis=2)) - torch.sum(torch.lgamma(alphas), axis=2) + torch.sum((alphas-1.0)*torch.log(diag_label.detach()),
                        axis=2)), axis=1))
        return loss


if __name__ == "__main__":
    set_seed(param.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if param.save_model:
        output_dir='./exp/'
        if not os.path.exists(output_dir): os.makedirs(output_dir) 
            
    trainloader,valloader,testloader=prep_dataloader(scp_train = "./data/iemocap_diag_train.scp",
                                                        scp_cv = "./data/iemocap_diag_cv5-all.scp",
                                                        scp_test = "./data/iemocap_diag_test5-all.scp",
                                                        label_hard_path = './data/IEMOCAP-hardlabel-diag.npy',
                                                        label_path='./data/IEMOCAP-softlabel-sum-diag.npy',
                                                        w2v2_path='./data/w2v2-ft-diag.npy',
                                                        bert_path='./data/bert-base-diag.npy',
                                                        order_path='./data/order.json',)

    label_all = np.load('./data/IEMOCAP-hardlabel.npy',allow_pickle=True).item()
    name_utt1=[x for x in label_all.keys() if label_all[x] == 5]

    model = TransformerModel(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=param.learning_rate, weight_decay = param.l2_reg)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9, last_epoch=-1)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=4,num_training_steps=param.EPOCH)


    loss_fn_kl = nn.KLDivLoss(reduction='sum')
    loss_fn = nn.CrossEntropyLoss() 
    loss_fn_dir = Dir_loss(reduction = 'sum',use_smooth=True)
    logsoftmax= nn.LogSoftmax(dim=-1)
    softmax= nn.Softmax(dim=-1)


    tf_ratio=[]
    tf_idx = 0

    log_dic={}
    log_dic_test={}

    for epoch in range(param.EPOCH):
        pred_index_train=np.array([])
        target_train=np.array([])

        pred_index_val=np.array([])
        target_val=np.array([])

        start_time = time.time()
        loss_total=0.0

        if param.forcing_type =='linear':
            if epoch <param.epoch_start_tf:
                param.forcing_decay_type = None
                param.forcing_ratio = 1
            else:
                param.forcing_decay_type = param.forcing_type
                param.forcing_ratio = param.forcing_ratio_default
                param.forcing_decay=param.forcing_decay_default
        elif param.forcing_type =='exp':
            if epoch <param.epoch_start_tf:
                param.forcing_decay_type = None
                param.forcing_ratio = 1
            else:
                param.forcing_decay_type = param.forcing_type
                param.forcing_ratio = param.forcing_ratio_default
        elif param.forcing_type =='sigmoid':
                param.forcing_decay_type = param.forcing_type
                param.forcing_ratio = param.forcing_ratio_default
        else:
            param.forcing_decay_type = None
            param.forcing_ratio = param.forcing_ratio_default

        if epoch ==param.epoch_start_tf: tf_idx = 0

        model.train()
        for i, data in enumerate(trainloader, 0):
            op_6,tgt_soft,tgt_dpn,fname,op_5,tgt_5 = train_batch(tf_idx,data,filter4=False)
            tf_idx+=1

            _, indicies = torch.max(op_5,1)
            # loss = loss_fn(op_5,tgt_5.long())   # for hard system
            loss_kl = loss_fn_kl(logsoftmax(op_6),tgt_soft)
            loss_dpn = loss_fn_dir(op_6,tgt_dpn)
            loss = param.kl * loss_kl + loss_dpn

            pred_index_train=np.append(pred_index_train,indicies.cpu().numpy())
            target_train=np.append(target_train,tgt_5.cpu().numpy())
            assert pred_index_train.shape == target_train.shape,(pred_index_train.shape , target_train.shape)
            loss_total+=loss.detach().item()

            optimizer.zero_grad()
            if param.grad_accum >1 :
                loss/=param.grad_accum
                loss.backward()
                if i%param.grad_accum ==0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        model.eval()
        cv_KL=0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                op_6,tgt_soft,tgt_dpn,fname,op_5,tgt_5 = decode(data)
                KL=loss_fn_kl(logsoftmax(op_6),tgt_soft).detach().item()
                cv_KL+=KL
                _, indicies = torch.max(op_5,1)
                pred_index_val=np.append(pred_index_val,indicies.cpu().numpy())
                target_val=np.append(target_val,tgt_5.cpu().numpy())
                assert pred_index_val.shape == target_val.shape,(pred_index_val.shape , target_val.shape)


        torch.save(model.state_dict(), '{}/pytorchmodel-E{}.pt'.format(output_dir,epoch))

        # loss_batch = loss_total/len(trainloader)    #for hard system
        loss_batch = loss_total/len(target_train)
        log_dic[epoch]={'train_loss':loss_batch,
                        'train_5class_WA':accuracy_score(target_train,pred_index_train)*100.0,
                        'train_5class_UA':balanced_accuracy_score(target_train,pred_index_train)*100.0,
                        'val_5class_KL':cv_KL/len(target_val),
                        'val_5class_WA':accuracy_score(target_val,pred_index_val)*100.0,
                        'val_5class_UA':balanced_accuracy_score(target_val,pred_index_val)*100.0,
                        'elapse_time':time.time() - start_time,
                        'current_lr':optimizer.state_dict()['param_groups'][0]['lr'],
                        'tf_ratio':tf_ratio[-1]
        }
        scheduler.step()
        torch.cuda.empty_cache()
        
    with open(output_dir+'train_log.json', 'wb') as json_file:
        json_file.write(json.dumps(log_dic, indent=4, sort_keys=True).encode('utf_8'))

    # test
    pred_index_test=np.array([])
    target_test=np.array([])

    pred_index4_test=np.array([])
    target4_test=np.array([])

    model.eval()
    test_KL=0
    y_true=[]
    y_score_maxP=[]
    y_score_ent=[]
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            op_6,tgt_soft,tgt_dpn,fname,op_5,tgt_5,op_4,tgt_4 = decode(data,filter4=True)
            KL=loss_fn_kl(logsoftmax(op_6),tgt_soft).detach().item()
            test_KL+=KL
            for i,x in enumerate(fname):
                if x in name_utt1:
                    y_true.append(0)
                else:
                    y_true.append(1)
                tmp=softmax(op_6[i]).cpu().numpy()
                y_score_maxP.append(max(tmp))
                y_score_ent.append(-entropy(tmp))

            _, indicies = torch.max(op_5,1)
            pred_index_test=np.append(pred_index_test,indicies.cpu().numpy())
            target_test=np.append(target_test,tgt_5.cpu().numpy())
            assert pred_index_test.shape == target_test.shape,(pred_index_test.shape , target_test.shape)
            _, indicies = torch.max(op_4[:,:4],1)

            pred_index4_test=np.append(pred_index4_test,indicies.cpu().numpy())
            target4_test=np.append(target4_test,tgt_4.cpu().numpy())

    log_dic_test[epoch]={'test_KL':test_KL/len(target_test),
                'test_5class_WA':accuracy_score(target_test,pred_index_test)*100.0,
                'test_5class_UA':balanced_accuracy_score(target_test,pred_index_test)*100.0,
                'test_4class_WA':accuracy_score(target4_test,pred_index4_test)*100.0,
                'test_4class_UA':balanced_accuracy_score(target4_test,pred_index4_test)*100.0,
                'test_AUC_maxP':average_precision_score(y_true, y_score_maxP),
                'test_AUC_ent':average_precision_score(y_true, y_score_ent)}
    np.savez(output_dir+"/AUC-score-"+str(epoch)+".npz",y_true=y_true, maxP=y_score_maxP, Ent=y_score_ent)
    
    with open(output_dir+'test_log.json', 'wb') as json_file:
        json_file.write(json.dumps(log_dic_test, indent=4, sort_keys=True).encode('utf_8'))
    
