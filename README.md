## Senti-Analysis-Irony

Sarcasm and irony recognition using machine learning and text mining techniques.

### Description

Sentiment Analysis with relation to sarcasm and irony is a complex issue, given the nuances of spoken and written language. There are several aims that this dataset and classifier aim to do:
- Provide a strong ensemble classifier to classify whether an input 'document' (a document being a given segment of text) is ironic or not.
- Utilise existing datasets to classify both sarcasm and irony, analysing the effect of context on recognition of these linguistic features.

### Datasets
- [ACL-Irony](https://github.com/bwallace/ACL-2014-irony), [Wallace, Kertz & Cherniak](https://www.aclweb.org/anthology/P14-2084) (2014).
- [News Headlines](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection).

### Results

- ACL-Irony: bag-of-words and doc2vec concatenation with Multinomial Naive Bayes -- 0.440 f1-score.
- News Headlines: bag-of-words and Multinomial Naive Bayes -- 0.770 f1-score. UPDATE

### Requirements

These scripts are built using Python 3.7, and require the following modules:
- [Scikit-Learn](http://scikit-learn.org/stable/install.html).
- [Gensim](https://radimrehurek.com/gensim/install.html).

### Scripts

There are several scripts that have been used for this work, including:
- nlp_utils.py: a utility script providing functionality for vectorisation of documents, as well as pre-processing and classification.
- vectorisers.py: the main script to run for testing the news headlines dataset.
- irony_stats.py: an edited version of the original provided with the ACL-Irony dataset. Run this to test the ACL-Irony dataset.

To run the scripts, type (update with sys.argv):
- News Headlines: python vectorisers.py
- ACL-Irony: python irony_stats.py

### Author

The author of this repository is [Erik Thomas](https://github.com/EThomas16), with the modules used having their respective authors available through the links provided.