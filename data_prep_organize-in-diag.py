import os
import json
import numpy as np


output_dir = '../data/'
order_file_path = '../data/order.json'
data_type = 'label_hard'
f=open(order_file_path,'r')
text=f.read()
paras=json.loads(text)
if data_type == 'label':
    data_path = '../data/IEMOCAP-softlabel-sum.npy'
if data_type == 'label_hard':
    data_path = '../data/IEMOCAP-hardlabel.npy'
elif data_type == 'w2v2':
    data_path = '../data/w2v2-ft-avg.npy'
elif data_type == 'bert':
    data_path = '../data/bert-base.npy'
data=np.load(data_path,allow_pickle=True).item()

diag_dic={}
cnt=[]
for ses in paras:
    tmp=[]
    for utt_name in paras[ses]:
        tmp.append(data[utt_name])
    cnt.append(len(tmp))
    diag_dic[ses]=np.array(tmp)
    # break
assert len(diag_dic)==151,len(diag_dic)
assert sum(cnt)==10039, sum(cnt)
np.save(data_path.replace('.npy','-diag.npy'),diag_dic)
