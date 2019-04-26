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

### Author

The author of this repositary is [Erik Thomas](https://github.com/EThomas16), with the modules used having their respective authors available through the links provided.