#Modifications of this repo
Added Logistic Regression (LR/ folder) on top of the produced ngram embeddings, according to author's code from this repository --> https://github.com/libofang/Neural-BoN .
Modifications to code and additional datasets.


# Neural-BoN
This code implements the Neural-BoN model proposed in AAAI-17 paper: [Bofang Li, Tao Liu, Zhe Zhao, Puwei Wang and Xiaoyong Du - **Neural Bag-of-Ngrams**].

## Code
This code has been tested on Windows, but it should work on Linux, OSX or any other operation system without any changes (Thanks to Java). 

All parameters your may need to change are in the top lines of src/main/Main.java.

To train Neural-BoN on an unlabeled corpus, you can just specify the corpus in the Main function of src/main/Main.java and run it. This will generate the learned n-gram representations in the results folder.

To train Neural-BoN on a labeled corpus, you can specify the corpus and implement an getXXXDataset function in src/myUtils/Dataset.java or change the getIMDBDataset function in src/myUtils/Dataset.java for the specific format.

## More Information
word2vec-sentiments.py to run logistic regression on top of embeddings produced in .txt format in previous step.
