"""
IAC paper: http://nldslab.soe.ucsc.edu/iac/iac_lrec.pdf

Ideas for improving on baseline:
- Use Word2Vec or Doc2Vec to extract vectors of words to glean contextual information

Index 0 of each JSON object is the contents of the posts
The remaining indexes are:
    - Post title
    - URL
    - Breadcrumbs
    - Discussion files

QR pairs have sarcasm as a metric as well as sarcasm unsure
this data is continuous and has other classes such as nicenasty,
questioning asserting etc.

No clear label for sarcasm, unlike sarcasm headlines dataset

For the first set of instances of data the indexes are as follows:
0: post ID
1: ?
2: Username of poster
3: Quote
4: post_info (including post title)
"""
import os
import re
import csv
import json
import numpy as np
import pandas as pd
from sklearn import metrics, svm, ensemble, naive_bayes, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, StratifiedShuffleSplit

import nlp_utils
from decorators import timer

np.set_printoptions(threshold=np.nan)
        
@timer  
def classify(clf, sent_data: object) -> float:
    """"
    Fits the given classifier to the training data and predicts the test data

    Keyword arguments:
    clf -- the classifier to be used to predict the data
    sent_data -- the instance of the SentimentData class containing the correct dataset to be tested

    Returns:
    score_list -- a list of all of the metrics for the classifier of the given dataset
    """
    clf.fit(sent_data.X_train, sent_data.y_train)
    predicted = clf.predict(sent_data.X_test)
    print(metrics.classification_report(sent_data.y_test, predicted, digits=3))
    print(f"Accuracy: {metrics.accuracy_score(sent_data.y_test, predicted)}")

"""
TEMPORARY METHODS
"""

def validate_kfold(train_path: str, test_path: str):
    train = np.load(train_path)
    test = np.load(test_path)

    for train_idx in train:
        for test_idx in test:
            if test_idx == train_idx:
                print("same value")

if __name__ == "__main__":
    data_root = "Data/ACL-2014-irony-master"
    lbl = "label"
    # sent_data = SentimentData(
    #     "Data/ACL-2014-irony-master/irony-bow_train_fixed.csv",
    #     "Data/ACL-2014-irony-master/irony-bow_test_fixed.csv",
    #     lbl)

    # SentimentData.sklearn_bow(
    #     f"{data_root}/irony-labeled.csv",
    #     f"{data_root}/irony-bow_stop-words.csv",
    #     lbl)

    # sent_data = SentimentData(
    #     single_file_path=f"{data_root}/csv/irony-bow.csv",
    #     lbl=lbl)

    # sent_data = SentimentData(
    #     single_file_path="F:/Documents/Programming/LocalSandbox/Sentiment-Analysis/Review_Dataset/reviews_Video_Games_training.csv",
    #     lbl="review_score"
    # )

    features, labels = nlp_utils.read_json(
        "Data/Sarcasm_Headlines_Dataset.json", "headline", "is_sarcastic")

    sarc = []
    non_sarc = []
    for label in labels:
        if int(label) == 1:
            sarc.append(label)
        else:
            non_sarc.append(label)

    print(f"Sarcastic instances: {len(sarc)}")
    print(f"Non-Sarcastic instances: {len(non_sarc)}")

    y = np.array(labels)
    X = nlp_utils.vectorise_feature_list(features, vectoriser="tf-idf")

    sent_data = nlp_utils.TextData()
    sent_data.X = X
    sent_data.y = y

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.3
    # )

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    shuffle_split = StratifiedShuffleSplit(n_splits=1)

<<<<<<< HEAD
    # clf = naive_bayes.BernoulliNB()
    clf = svm.LinearSVC()
    sent_data.stratify(shuffle_split, clf)
=======
    clf = naive_bayes.MultinomialNB()
    scores = sent_data.stratify(skf, clf, eval_metric="accuracy")
    for idx, score in enumerate(scores):
        print(f"Iteration {idx} accuracy: {score}")
>>>>>>> d02f2e767235f7f9f1feb986b6f8660a2503b7d9

    # sent_data.stratified_kfold(clf, splits=2, rand_state=50)
    # validate_kfold(
    #     "Data/ACL-2014-irony-master/k-fold_splits_train.npy", 
    #     "Data/ACL-2014-irony-master/k-fold_splits_test.npy")
    # sent_data.X_train, sent_data.X_test, sent_data.y_train, sent_data.y_test = train_test_split(sent_data.X, sent_data.y, test_size=0.01, random_state=42)
    # classify(clf, sent_data)