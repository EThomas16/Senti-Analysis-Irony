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
import numpy as np
import pandas as pd
from sklearn import metrics, svm, ensemble
from sklearn.feature_extraction.text import CountVectorizer

from decorators import timer

class SentimentData():
    def __init__(self, train_path: str, test_path: str, lbl: str):
        df_train = pd.read_csv(train_path, encoding='utf-8')
        df_test = pd.read_csv(test_path, encoding='utf-8')
        self.X_train = np.array(df_train.drop([lbl], 1))
        self.y_train = np.array(df_train[lbl])
        self.X_test = np.array(df_test.drop([lbl], 1))
        self.y_test = np.array(df_test[lbl])

        self.stopwords = ['a', "the", "is"]

    def clean_data(self, sentence: str):
        words = re.sub(r"[^\w]", " ", sentence).split()
        cleaned_text = [word.lower() for word in words if word not in self.stopwords]
        return cleaned_text

    def tokenise(self, features):
        words = []
        for sentence in features:
            cleaned_text = self.clean_data(sentence)
            words.extend(cleaned_text)

        words = sorted(list(set(words)))
        return words

    def bag_of_words(self, words: list, test_or_train: str):
        if test_or_train == "train":
            features = self.X_train
            labels = self.y_train
        elif test_or_train == "test":
            features = self.X_test
            labels = self.y_test

        vocab = self.tokenise(features)

    @staticmethod
    def sklearn_bow(input_path: str, output_path: str, lbl: str):
        df = pd.read_csv(input_path, encoding='utf-8')
        features = np.array(df.drop([lbl], 1))
        labels = np.array(df[lbl])
        vectoriser = CountVectorizer()
        fts = vectoriser.fit_transform(features.flatten())
        with open(output_path, 'a', encoding='utf-8') as test:
            for feature_name in vectoriser.get_feature_names():
                test.write(f"{feature_name},")
            test.write("label\n")
            for ft, label in zip(fts.toarray(), labels):
                for instance in ft:
                    test.write(f"{instance},")
                test.write(f"{label}\n")            

    @staticmethod
    def scan_data_dir(data_dir: str, ext: str = '.csv') -> list:
        """
        Scans a given directory to find all dataset files (isolates them from other file types)

        Keyword arguments:
        data_dir -- the directory containing the data in question
        ext -- the file extension of the files to be extracted

        Returns:
        data_files -- a list of all files of a given extension in the data_dir directory
        """
        data_files = []

        for path, subdir, files in os.walk(data_dir):
            for f_name in files:
                if ext in f_name:
                    f_path = os.path.join(path, f_name)
                    data_files.append(f_path)

        return data_files

    @staticmethod
    def train_test_split(original_path: str, train_path: str, test_path: str, num_instances: int, enc: str = 'utf-8'):
        train_data = []
        test_data = []
        with open(original_path, 'r', encoding='utf-8') as csv:
            for idx, line in enumerate(csv.readlines()):
                if "[deleted]" in line:
                    continue
                if idx <= (round(num_instances * 0.8)):
                    train_data.append(line)
                else:
                    test_data.append(line)

        with open(train_path, 'a', encoding='utf-8') as train:
            for line in train_data:
                train.write(line)

        with open(test_path, 'a', encoding='utf-8') as test:
            for line in test_data:
                test.write(line)

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

if __name__ == "__main__":
    sent_data = SentimentData(
        "Data/ACL-2014-irony-master/irony-bow_train.csv",
        "Data/ACL-2014-irony-master/irony-bow_test.csv",
        "label")

    # SentimentData.sklearn_bow(
    #     "Data/ACL-2014-irony-master/irony-labeled.csv",
    #     "Data/ACL-2014-irony-master/irony-bow.csv",
    #     "label")

    # SentimentData.train_test_split(
    #     "Data/ACL-2014-irony-master/irony-bow.csv", 
    #     "Data/ACL-2014-irony-master/irony-bow_train.csv", 
    #     "Data/ACL-2014-irony-master/irony-bow_test.csv",
    #     1949)
    clf = svm.SVC(kernel='linear')
    classify(clf, sent_data)