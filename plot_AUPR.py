import numpy as np
from sklearn.metrics import precision_recall_curve,average_precision_score
from matplotlib import pyplot as plt 

color={'dark blue':(0,62/255,114/255),'blue':(0,115/255,207/255),'light blue':(106/255,173/255,228/255)}

def ComputePR(path, mode):
	dat=np.load(path)
	y_true=dat['y_true']
	maxP=dat['maxP']
	Ent=dat['Ent']
	if mode=='maxP':
		precision, recall, thresholds = precision_recall_curve(y_true,maxP)
		AP = average_precision_score(y_true, maxP)
	elif mode =='Ent':
		precision, recall, thresholds = precision_recall_curve(y_true,Ent)
		AP = average_precision_score(y_true, Ent)
	return precision, recall, AP

data_dir='./exp/'
P_maxP_dpnkl, R_maxP_dpnkl, AP_maxP_dpnkl =ComputePR(data_dir+"AUC-score.npz",'maxP')
P_Ent_dpnkl, R_Ent_dpnkl, AP_Ent_dpnkl =ComputePR(data_dir+"AUC-score.npz",'Ent')

plt.figure(figsize=(3.5,3))
l1,=plt.plot(R_maxP_dpnkl,P_maxP_dpnkl,color=color['light blue'], label='DPN-KL, AUPR={:.4f}'.format(AP_maxP_dpnkl))
plt.title("(a) PR curve (Max.P)")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


plt.figure(figsize=(3.5,3))
l1,=plt.plot(R_Ent_dpnkl,P_Ent_dpnkl,color=color['light blue'], label='DPN-KL, AUPR={:.4f}'.format(AP_Ent_dpnkl))
plt.title("(b) PR curve (Ent.)")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
