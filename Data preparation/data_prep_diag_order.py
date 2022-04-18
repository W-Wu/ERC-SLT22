import os
import json
import numpy as np

iemo_root = "./IEMOCAP_full_release/"
trans_dir=[iemo_root+'Session'+str(x)+'/dialog/transcriptions/' for x in range(1,6)]

out_dir='./'
json_dict = {}
for path in trans_dir:
    for file in os.listdir(path):
        ses_name=file.split('.')[0]
        f = open(os.path.join(path,file),'r')
        lines=f.readlines()
        name=[line.split(' ')[0]  for line in lines if line.split(' ')[0].startswith('Ses') and 'XX' not in line.split(' ')[0]]
        json_dict[ses_name]=name

with open(out_dir+'order.json', 'wb') as json_file:
    json_file.write(json.dumps(json_dict, indent=4, sort_keys=True).encode('utf_8'))
assert len(json_dict.values())==151
cnt=0
for x in json_dict:
    cnt+=len(json_dict[x])
assert cnt == 10039