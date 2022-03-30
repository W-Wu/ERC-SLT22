import os
import json
import numpy as np

iemo_root = "../IEMOCAP_full_release/"
output_dir = './data/'

file_roots=[iemo_root+"Session"+str(i)+"/dialog/EmoEvaluation" for i in range(1,6)]
file_paths=[]   
for file_dir in file_roots:
    for files in os.listdir(file_dir):  
        if os.path.splitext(files)[1] == '.txt':  
            file_paths.append(os.path.join(file_dir, files))  
assert len(file_paths) == 151

mapping={'Neutral':[1,0,0,0,0],'Happiness':[0,1,0,0,0],'Excited':[0,1,0,0,0],'Sadness':[0,0,1,0,0],'Anger':[0,0,0,1,0]}
emo_dic = {'neu':0,'hap':1,'sad':2,'ang':3,'oth':4,'xxx':5}

hard_label_dic={}
soft_label_dic={}
Flag=0
for label_path in file_paths:
    f = open(label_path,'r')
    line = f.readline()
    while line:
        if line.startswith('['):
            if Flag == 1:
                if emo == 'exc': emo = 'hap'
                if emo not in  ['neu','ang','hap','sad' ,'xxx']: emo ='oth' 
                hard_label_dic[name]=emo_dic[emo]
                soft_label_dic[name]=np.array(label_sum).sum(axis=0)
            tmp = line.split()
            name=tmp[3]
            emo=tmp[4]
            label_sum=[]
            Flag = 1
        elif line.startswith('C-E'):
            tmp = line.split()
            emos=[x[:-1] for x in tmp[1:-1]]
            for x in emos:
                if x in mapping:
                    label_sum.append(mapping[x])
                else:
                    label_sum.append([0,0,0,0,1])
        line=f.readline()
hard_label_dic[name]=emo_dic[emo]
soft_label_dic[name]=np.array(label_sum).sum(axis=0)
assert len(hard_label_dic)==len(soft_label_dic)==10039
np.save(output_dir+'/IEMOCAP-hardlabel.npy',hard_label_dic)
np.save(output_dir+'/IEMOCAP-softlabel-sum.npy',soft_label_dic)


trans_dir=[iemo_root+'Session'+str(x)+'/dialog/transcriptions/' for x in range(1,6)]
json_dict = {}
for path in trans_dir:
    for file in os.listdir(path):
        ses_name=file.split('.')[0]
        f = open(os.path.join(path,file),'r')
        lines=f.readlines()
        name=[line.split(' ')[0]  for line in lines if line.split(' ')[0].startswith('Ses') and 'XX' not in line.split(' ')[0]]
        json_dict[ses_name]=name

with open(output_dir+'order.json', 'wb') as json_file:
    json_file.write(json.dumps(json_dict, indent=4, sort_keys=True).encode('utf_8'))
assert len(json_dict.values())==151
cnt=0
for x in json_dict:
    cnt+=len(json_dict[x])
assert cnt == 10039