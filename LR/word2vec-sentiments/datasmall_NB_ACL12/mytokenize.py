import re
import nltk.tokenize as tk
import numpy as np
import nltk
import os
import sys

# TreebankWordTokenizer
# WordPunctTokenizer
# PunctWordTokenizer
def tokenizefile(fname):
	tbt = tk.TreebankWordTokenizer()
	f = open(fname)
	f.readline(); f.readline()
	raw = f.read(500000)
	raw = re.sub('[0-9]', '#', raw)
	raw = re.sub('#+', '#', raw)
	raw = re.sub('[~%<>^&*()+={}/\\\]', ' ', raw)
	raw = re.sub('-+', '-', raw)
	f.close()
	return tbt.tokenize(raw)

def writeFile(tokens, fname):
	allstr = ' '.join(tokens)
	f = open(fname, 'w')
	f.write(allstr.lower())
	f.close();

def tokenizefolder(pathname):
	dirList=os.listdir(pathname)
	for fname in dirList:
		if not str.isdigit(fname): continue
		inputname = os.path.join(pathname, fname)
		tokens = tokenizefile(inputname)
		outputname = inputname+'.tod3'
		writeFile(tokens, outputname)

if __name__ == "__main__":
	ngs = ['alt.atheism', 'comp.graphics', 'comp.windows.x', 'rec.sport.baseball', 'sci.crypt', 'talk.religion.misc'];
	for ngname in ngs:
		fulldir = os.path.join('./20ng/', ngname)
		print 'tokenizing ' + fulldir
		tokenizefolder(fulldir)

#	ngs = ['train/pos', 'train/neg', 'test/pos', 'test/neg'];
#	for ngname in ngs:
#		fulldir = os.path.join('./mrl/', ngname)
#		print 'tokenizing ' + fulldir
#		tokenizefolder(fulldir)


