from sklearn.datasets import load_files
from sklearn.feature_extraction.text import WordNGramAnalyzer
import nltk

twenty_train = load_files('./20news-bydate-train', categories=['alt.atheism'], shuffle=False)
WordNGramAnalyzer().analyze(twenty_train.data[0])
mytok = nltk.tokenize.TreebankWordTokenizer();
mytok.tokenize(twenty_train.data[4])
