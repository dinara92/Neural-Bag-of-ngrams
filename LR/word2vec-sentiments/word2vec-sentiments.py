#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:43:13 2017

@author: dinara
"""

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np
import os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.keyedvectors import KeyedVectors
from nltk import bigrams, trigrams, ngrams

# random
import random
# classifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import Imputer
import logging
import sys

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

num_features = 500

#new: since github has a 100M limit, load from a bunch of files in
	#a directory

def load_word2vec(dir):
    word2vec = {}
    for path in os.listdir(dir):
        iword2vec = {}
        #load the word2vec features.
        with open(os.path.join(dir,path), 'r') as fin:
            if path == 'Word2Vec_AsArrays_In1Out1.txt':
                next(fin) #skip information on first line
            for line in fin:
                #print('LINE: ' + line)
                items = line.replace('\r','').replace('\n','').split(' ')
                #print(items[0])
                if len(items) < 10: continue
                word = items[0]
                vect = np.array([float(i) for i in items[1:] if len(i) > 1])
                iword2vec[word] = vect

    word2vec.update(iword2vec)
    return word2vec

def load_word2vecODP(dir):
    word2vec = {}
    for path in os.listdir(dir):
        iword2vec = {}
        #load the word2vec features.
        #with open(os.path.join(dir,path), 'rb') as fin:
        #    lines = [line.decode('utf8', 'ignore') for line in fin.readlines()]

        with open(os.path.join(dir,path), 'r',encoding='ISO-8859-1') as fin:
            if path == 'Word2Vec.txt':
                next(fin) #skip information on first line
                next(fin)
                next(fin)
            for line in fin:
                #print('LINE: ' + line)
                items = line.replace('\r','').replace('\n','').split(' ')
                #print(items[0])
                if len(items) < 10: continue
                word = items[0]
                vect = np.array([float(i) for i in items[1:] if len(i) > 1])
                iword2vec[word] = vect

    word2vec.update(iword2vec)
    return word2vec

def tfidf_vec(sentences):
    print('building tf-idf matrix ...')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x for x in sentences])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))

    return tfidf

def sent_vectorizer(sent, num_features, word2vec, tfidf):
        sent_vec = np.zeros(num_features)
        numw = 0
        # Index2word is a list that contains the names of the words in 
        # the model's vocabulary. Convert it to a set, for speed 
        #index2word_set = set(model.index2word)
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
    
        for word in sent:
            if word not in word2vec:
                #print(w + '\n')
                #f.write(w + '\n')
                continue
            try:
                sent_vec = np.add(sent_vec, word2vec[word]*tfidf[word])
                #sent_vec = np.add(sent_vec, word2vec[word])

                numw+=1
            except KeyError:
                continue
        #return sent_vec / np.sqrt(sent_vec.dot(sent_vec)) /numw
        #sent_vec = np.divide(sent_vec,np.sqrt(sent_vec.dot(sent_vec)))
        #sent_vec = np.divide(sent_vec, numw)
        #/home/dinara/java-projects/Neural-BoN-master/LR/word2vec-sentiments/word2vec-sentiments.py:78: RuntimeWarning: invalid value encountered in true_divide
        #sent_vec = np.divide(sent_vec, numw)

        return sent_vec

def sent_2gram_vectorizer(sent, num_features, word2vec, tfidf):
    sent_vec = np.zeros(num_features)
    numw = 0

    tokens = sent
    #print(tokens)
    ngrams_list = ['']
    bi_grams = bigrams(tokens)
    for bigram in bi_grams:
        bigram = ''.join(bigram)
        ngrams_list.append(str(bigram))
    #print(ngrams_list)
    for ngram in ngrams_list:
        tokens.append(ngram)
    #print(tokens)

    for word in tokens:
        if word not in word2vec:
            continue
        try:
            sent_vec = np.add(sent_vec, word2vec[word]*tfidf[word])
            numw+=1
        except KeyError:
            sent_vec = np.add(sent_vec, word2vec[word])

    return sent_vec

def sent_2and3gram_vectorizer(sent, num_features, word2vec, tfidf):
    sent_vec = np.zeros(num_features)
    numw = 0

    tokens = sent

    #print(tokens)
    n2grams_list = ['']
    bi_grams = bigrams(tokens)
    for bigram in bi_grams:
        bigram = ''.join(bigram)
        n2grams_list.append(str(bigram))
    #print(ngrams_list)
    for ngram in n2grams_list:
        tokens.append(ngram)
    #print(tokens)

    #print(tokens)
    n3grams_list = ['']
    tri_grams = trigrams(sent) #make trigrams from original sent
    for trigram in tri_grams:
        trigram = ''.join(trigram)
        n3grams_list.append(str(trigram))
    #print(ngrams_list)
    for ngram in n3grams_list:
        tokens.append(ngram)
    #print(tokens)

    for word in tokens:
        if word not in word2vec:
            continue
        try:
            sent_vec = np.add(sent_vec, word2vec[word]*tfidf[word])
            numw+=1
        except KeyError:
            sent_vec = np.add(sent_vec, word2vec[word])

    return sent_vec

def getAvgFeatureVecs(sentences, word2vec, tfidf, num_features):
    #f = open('myfile', 'w')

    # Given a set of sentences (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    featureVecs = np.zeros((len(sentences),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for sent in sentences:
       #
       # Print a status message every 1000th review
       if counter%1000 == 0:
           print ("Document %d of %d" % (counter, len(sentences)))
       featureVecs[counter] = sent_vectorizer(sent, num_features, word2vec, tfidf)
       counter = counter + 1


    return featureVecs
    

def loadDataSet(file):
    with open(file, 'r', encoding = "ISO-8859-1") as f:
        sentences = [re.sub("[^\w]", "_ ",  line).split() for line in f]
#        sentences = [re.sub("[^\w]", " ",  line).split() for line in f]

    return sentences

def odp_test():
    base_folder = '/home/dinara/java-projects/FusionODPWord2VecNew/clean1BilCorpus/'
    sents = loadDataSet(base_folder+'news.en-0000-of-00100.txt')
    tfidf_vec(sents)

    word_vectors = KeyedVectors.load_word2vec_format('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/1bil_words_news_ODP_DBpediaAnnot_10context_300f_0mincount', binary=True, encoding='utf-8')


def imdbSentiment():
    train_arrays = np.zeros((25000, num_features))
    train_labels = np.zeros(25000)

    for i in range(12500):
        train_arrays[i] = trainVecsPos[i]
        train_arrays[12500 + i] = trainVecsNeg[i]
        train_labels[i] = 1
        train_labels[12500 + i] = 0

    log.info(train_labels)

    test_arrays = np.zeros((25000, num_features))
    test_labels = np.zeros(25000)

    for i in range(12500):
        test_arrays[i] = testVecsPos[i]
        test_arrays[12500 + i] = testVecsNeg[i]
        test_labels[i] = 1
        test_labels[12500 + i] = 0

    arrays_and_labels = {0: train_arrays, 1: train_labels, 2: test_arrays, 3: test_labels}
    return arrays_and_labels

def MRSentiment():
    train_arrays = np.zeros((5330, num_features))
    train_labels = np.zeros(5330)

    for i in range(2665):
        train_arrays[i] = trainVecsPos[i]
        train_arrays[2665 + i] = trainVecsNeg[i]
        train_labels[i] = 1
        train_labels[2665 + i] = 0

    log.info(train_labels)

    test_arrays = np.zeros((5330, num_features))
    test_labels = np.zeros(5330)

    for i in range(2665):
        test_arrays[i] = testVecsPos[i]
        test_arrays[2665 + i] = testVecsNeg[i]
        test_labels[i] = 1
        test_labels[2665 + i] = 0

    arrays_and_labels = {0: train_arrays, 1: train_labels, 2: test_arrays, 3: test_labels}
    return arrays_and_labels

def CRSentiment():
    train_arrays = np.zeros((1887, num_features))
    train_labels = np.zeros(1887)

    for i in range(1203):
        train_arrays[i] = trainVecsPos[i]
        train_labels[i] = 1

    for i in range(683):
        train_arrays[1203 + i] = trainVecsNeg[i]
        train_labels[1203 + i] = 0

    log.info(train_labels)

    test_arrays = np.zeros((1887, num_features))
    test_labels = np.zeros(1887)

    for i in range(1203):
        test_arrays[i] = testVecsPos[i]
        test_labels[i] = 1

    for i in range(683):
        test_arrays[1203 + i] = testVecsNeg[i]
        test_labels[1203 + i] = 0

    arrays_and_labels = {0: train_arrays, 1: train_labels, 2: test_arrays, 3: test_labels}
    return arrays_and_labels

def SubjSentiment():
    train_arrays = np.zeros((5001, num_features))
    train_labels = np.zeros(5001)

    for i in range(2500):
        train_arrays[i] = trainVecsPos[i]
        train_arrays[2500 + i] = trainVecsNeg[i]
        train_labels[i] = 1
        train_labels[2500 + i] = 0

    log.info(train_labels)

    test_arrays = np.zeros((5001, num_features))
    test_labels = np.zeros(5001)

    for i in range(2500):
        test_arrays[i] = testVecsPos[i]
        test_arrays[2500 + i] = testVecsNeg[i]
        test_labels[i] = 1
        test_labels[2500 + i] = 0

    arrays_and_labels = {0: train_arrays, 1: train_labels, 2: test_arrays, 3: test_labels}
    return arrays_and_labels

def read_file(filename, stream=False):
    """
    Reads the content of the given file in binary mode
    :param filename: Name of the file to read
    :param stream: If True, return the file stream
    :return: File content or stream
    """
    for subfolder in ('java', ''):
        found_file = os.path.join(
            os.path.dirname(__file__), subfolder, filename)
        if os.path.exists(found_file):
            break
    else:
        raise IOError("File not found: {0}".format(filename))

    if stream:
        return open(found_file, "rb")
    else:
        with open(found_file, 'rb') as filep:
            return filep.read()



log.info('source load')
base_folder = '/home/dinara/java-projects/Neural-BoN-master/data/imdb_train_test/'

log.info('Load Sentences')
'''train_pos_sentences = loadDataSet(base_folder+'train-rt-polarity.pos')
train_neg_sentences = loadDataSet(base_folder+'train-rt-polarity.neg')
test_pos_sentences = loadDataSet(base_folder+'test-rt-polarity.pos')
test_neg_sentences = loadDataSet(base_folder+'test-rt-polarity.neg')'''

train_pos_sentences = loadDataSet(base_folder+'train-pos.txt')
train_neg_sentences = loadDataSet(base_folder+'train-neg.txt')
test_pos_sentences = loadDataSet(base_folder+'test-pos.txt')
test_neg_sentences = loadDataSet(base_folder+'test-neg.txt')

'''train_pos_sentences = loadDataSet(base_folder+'train-custrev.pos')
train_neg_sentences = loadDataSet(base_folder+'train-custrev.neg')
test_pos_sentences = loadDataSet(base_folder+'test-custrev.pos')
test_neg_sentences = loadDataSet(base_folder+'test-custrev.neg')'''

'''train_pos_sentences = loadDataSet(base_folder+'train-subj.objective')
train_neg_sentences = loadDataSet(base_folder+'train-subj.subjective')
test_pos_sentences = loadDataSet(base_folder+'test-subj.objective')
test_neg_sentences = loadDataSet(base_folder+'test-subj.subjective')'''

#odp_test()

tfidf = tfidf_vec(train_pos_sentences+train_neg_sentences)

log.info('word2vec')
#word2vec = load_word2vec('/home/dinara/java-projects/Neural-BoN-master/results/imdb_train_test2/l__500_r3084/fold0/')
#word2vec = load_word2vec('/home/dinara/java-projects/FusionODPWord2VecNew/ODP_DBPedia_SkipGram/word2vec_original/')

#word2vec = load_word2vec('/home/dinara/java-projects/FusionODPWord2VecNew/word2vec_models_asVectorArrays/word2vec_original/')
word2vec = KeyedVectors.load_word2vec_format('/home/dinara/java-projects/FusionODPWord2VecNew/word2vec_models_asVectorArrays/word2vec_original/Word2Vec_AsArrays-copy.txt', binary=False)
#print(train_pos_sentences[55])
#sent_ngrams = sent_2and3gram_vectorizer(train_pos_sentences[55], num_features, word2vec, tfidf)


log.info('FeatureVecs')
trainVecsPos = Imputer().fit_transform(getAvgFeatureVecs(train_pos_sentences, word2vec, tfidf, num_features))
trainVecsNeg = Imputer().fit_transform(getAvgFeatureVecs(train_neg_sentences, word2vec, tfidf, num_features))
testVecsPos = Imputer().fit_transform(getAvgFeatureVecs(test_pos_sentences, word2vec, tfidf, num_features))
testVecsNeg = Imputer().fit_transform(getAvgFeatureVecs(test_neg_sentences, word2vec, tfidf, num_features))


log.info('Sentiment')

log.info('Fitting')

classifier = LogisticRegression()
classifier.fit(imdbSentiment()[0], imdbSentiment()[1])

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

log.info(classifier.score(imdbSentiment()[2], imdbSentiment()[3]))


'''classifier = LogisticRegressionCV()
classifier.fit(SubjSentiment()[0], SubjSentiment()[1])

LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001,
                     max_iter=100, class_weight=None, n_jobs=1, verbose=0, refit=True, intercept_scaling=1.0, multi_class='ovr', random_state=None)

log.info(classifier.score(SubjSentiment()[2], SubjSentiment()[3]))'''

