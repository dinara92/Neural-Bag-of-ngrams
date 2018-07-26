#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:51:13 2017

@author: dinara
"""
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans as kmeans

def load_word2vec(dir):
	#new: since github has a 100M limit, load from a bunch of files in 
	#a directory
	word2vec = {}
	for path in os.listdir(dir):
		iword2vec = {}
		#load the word2vec features.
		with open(os.path.join(dir,path), 'r') as fin:
			if path == 'WV500.txt':
				next(fin) #skip information on first line
			for line in fin:
				items = line.replace('\r','').replace('\n','').split(' ')
				if len(items) < 10: continue
				word = items[0]
				vect = np.array([float(i) for i in items[1:] if len(i) > 1])
				iword2vec[word] = vect
		
		word2vec.update(iword2vec)
		
	return word2vec
    
def get_furthest_word(words, word2vect):
	vectlist = []
	for word in words:
		#unknown word? 
		if word not in word2vect: return word
		#normalize.
		vectlist.append(word2vect[word]/np.linalg.norm(word2vect[word]))
	mean = np.array(vectlist).mean(axis=0)
	mean = mean / np.linalg.norm(mean)
	
	#figure out which is furthest
	dists = [np.linalg.norm(v - mean) for v in vectlist]
	return words[np.argmax(dists)]

def cluster_vects(word2vect):
    #use sklearn minibatch kmeans to cluster the vectors.
	clusters = kmeans(n_clusters= 25, max_iter=10,batch_size=200,
						n_init=1,init_size=2000)
	X = np.array([i.T for i in word2vect.itervalues()])
	y = [i for i in word2vect.iterkeys()]
	
	print ('fitting kmeans, may take some time')
	clusters.fit(X)
	print ('done.')
	
	#now we can get a mapping from word->label
	#which will let us figure out which other words are in the same cluster
	return {word:label for word,label in zip(y,clusters.labels_)}
	
def words_in_cluster(word, word_to_label):
	#sorry, this is O(n), n is pretty large
	#it could be O(k), k=cluster size, but that would cost more memory
	label = word_to_label[word]
	#get the other words with this label
	similar_words = [key for key,val in word_to_label.iteritems() if val==label]
	return similar_words

def furthest(word2vec):
    print ('Type several words separated by spaces. The more words you enter, the better I can guess.')
    while (True):
        words = input('->').lower().split(' ')
        print ('I think',get_furthest_word(words, word2vec),'doesnt belong in this list!\n')
 
        
        
        
        
def main():
	word2vec = load_word2vec('/home/dinara/java-projects/Neural-BoN-master/results/MR5/w__500_r2173/fold0/')
	index = 0
	for word in word2vec:
		if(word.count('_') < 2):
			index+=1
			print(word)

	#print(word2vec['gladiatorial_'].size)
	print(word2vec['savage_'].size)
	print(word2vec['craft_'].size)
	print(word2vec['battery_'].size)
	print(index)

if __name__ == '__main__':
    main()
