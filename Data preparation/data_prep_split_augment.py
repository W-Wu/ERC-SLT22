import os
import numpy as np
import json
import sys

test_id=int(sys.argv[1])    # session id for testing
maxlen_set=int(sys.argv[2]) # if zero, maintain full sequence without augmentation
if maxlen_set > 0:
    num_augment=int(sys.argv[3])    
    variableL = [0.5,1.0] if sys.argv[4]=='1' else None # variable sub-seq len


f=open('../data/order.json','r')
text=f.read()
paras=json.loads(text)

test_list =[x for x in paras.keys() if int(x[4]) == test_id]
traincv_list =[x for x in paras.keys() if int(x[4]) != test_id]
cv_list =[x for i,x in enumerate(traincv_list) if i % 5 ==0]
train_list =[x for i,x in enumerate(traincv_list) if i % 5 !=0]

output_dir = '../data/'

def get_maxlen(maxlen,variableL, meetinglength):
    if variableL is not None:
        if maxlen is not None:
            maxlen = int(np.random.uniform(variableL[0], variableL[1]) * min(meetinglength, maxlen))-1
        else:
            maxlen = int(np.random.uniform(variableL[0], variableL[1]) * meetinglength)-1
    else:
        if maxlen is not None:
            maxlen = maxlen
        else:
            maxlen = meetinglength
    return maxlen

if maxlen_set >0:
    file_name=output_dir+"iemocap_diag_train.scp"
    print(file_name)
    scp = open(file_name,'w')
    scp.write('name\tstart\tend\n')
    for ses_id in train_list:
        all_utt = paras[ses_id]
        meetinglength=len(all_utt)
        for i in range(num_augment):
            maxlen=get_maxlen(maxlen= maxlen_set,variableL=variableL, meetinglength=meetinglength)
            assert maxlen>0, maxlen
            start_idx = np.random.randint(meetinglength - maxlen)
            scp.write(ses_id+'\t'+str(start_idx)+'\t'+str(start_idx+maxlen)+'\n')
    scp.close()

else:
    # full
    scp_dic={'test':test_list,'cv':cv_list,'train':train_list}
    for scp_name, scp_file in scp_dic.items():
        file_name=output_dir+"iemocap_diag_"+scp_name+"-all.scp"
        print(file_name)
        scp = open(file_name,'w')
        scp.write('name\tstart\tend\n')
        for ses_id in scp_file:
            scp.write(ses_id+'\t'+str(0)+'\t'+str(len(paras[ses_id]))+'\n')
        scp.close()
