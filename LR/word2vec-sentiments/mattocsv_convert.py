import scipy.io
import numpy as np

data = scipy.io.loadmat("datasmall_NB_ACL12/20ng/unigram_ng20_atheisms_strip_noheader.mat")

for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("/home/dinara/java-projects/Neural-BoN-master/LR/word2vec-sentiments/datasmall_NB_ACL12/20ng/csv/"+i+".csv"),data[i],fmt='%s',delimiter=',')
