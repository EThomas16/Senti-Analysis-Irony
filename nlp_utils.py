import os
import re
import csv
import json
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, StratifiedShuffleSplit

class TextData():
    def __init__(
        self, train_path: str = "", test_path: str = "", 
        lbl: str = "label", single_file_path: str = "",
        stopwords = ["a", "the", "is"]):

        if single_file_path:
            df = pd.read_csv(single_file_path)
            self.X = np.array(df.drop([lbl], 1))
            # FIXME: work with any labelled column (strip string?)
            self.y = np.array(df.iloc[:, -1])
            self.X_train = []; self.y_train = []
            self.X_test = []; self.y_test = []
            return

        if not train_path and not test_path:
            return

        self.read_train_test_csvs(train_path, test_path, lbl)

        self.stopwords = stopwords

    def read_train_test_csvs(self, train_path: str, test_path: str, lbl: str):
        """
        Initialises split training and test CSV files

        Keyword arguments:
        train_path -- the path to the training data CSV
        test_path -- the path to the testing data CSV
        lbl -- the heading used for the labels in the CSV
        """
        df_train = pd.read_csv(train_path, encoding='utf-8')
        df_test = pd.read_csv(test_path, encoding='utf-8')
        self.X_train = np.array(df_train.drop([lbl], 1))
        self.y_train = np.array(df_train[lbl])
        self.X_test = np.array(df_test.drop([lbl], 1))
        self.y_test = np.array(df_test[lbl])

    def clean_data(self, sentence: str):
        """
        Removes unwanted characters from a sentence using regular expressions

        Keyword arguments:
        sentence -- 
        """
        words = re.sub(r"[^\w]", " ", sentence).split()
        cleaned_text = [word.lower() for word in words if word not in self.stopwords]
        return cleaned_text

    def tokenise(self, features: list):
        """
        LEGACY: tokenises a set of sentences into individual words, whilst sorting them

        Keyword arguments:
        features -- the feature set to be converted into sorted words

        Returns:
        words -- the sorted words for use in bag-of-words
        """
        words = []
        for sentence in features:
            cleaned_text = self.clean_data(sentence)
            words.extend(cleaned_text)

        words = sorted(list(set(words)))
        return words

    def stratify(
        self, split_algorithm: object, clf: object):
        """
        Stratifies the data using the provided splitting algorithm. This is used to split datasets
        into training and test, as well as in cross-validation of datasets

        Keyword arguments:
        split_algorithm -- the stratification algorithm to use to split the data (scikit-learn tested)
        clf -- the classifier to use to classify the splits
        """
        for train_idx, test_idx in split_algorithm.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx] 
            clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)
            # TODO: return the performance scores instead of just printing them
            print(metrics.classification_report(y_test, predicted))

def sklearn_bow(
    input_path: str, output_path: str, lbl: str,
    max_feats: int = 50000, stop_word_lang: str = "english"):
    """
    Generates bag-of-words features from a given CSV file of features

    Keyword arguments:
    input_path -- the path to the CSV file containing the features to read
    output_path -- the path to the new CSV file be generated
    lbl -- the heading of the label for the dataset
    max_feats -- the maximum number of words that can be used from all sentences in the bag of words
    stop_word_lang -- the language of the text
    """
    df = pd.read_csv(input_path, encoding='utf-8')
    features = np.array(df.drop([lbl], 1))
    labels = np.array(df[lbl])

    vectoriser = CountVectorizer(
        max_features=max_feats, 
        stop_words=stop_word_lang)

    fts = vectoriser.fit_transform(features.flatten())
    with open(output_path, 'a', encoding='utf-8') as test:
        for feature_name in vectoriser.get_feature_names():
            test.write(f"{feature_name},")
        test.write("label\n")
        for ft, label in zip(fts.toarray(), labels):
            for instance in ft:
                test.write(f"{instance},")
            test.write(f"{label}\n") 

def sklearn_bow_list(
    features: list, max_feats: int = 50000, stop_word_lang: str = "english") -> object:
    """
    Generates bag-of-words features from a list of sentences

    Keyword arguments:
    features -- list of all the sentences to be converted
    max_feats -- the maximum number of words that can be used from all sentences in the bag of words
    stop_word_lang -- the language of the text

    Returns:
    A vectoriser that has been fitted to the provided sentences
    """
    vectoriser = CountVectorizer(
        max_features=max_feats,
        stop_words=stop_word_lang
    )

    return vectoriser.fit_transform(features)

def read_json(json_path: str, feature_heading: str, label_heading: str) -> (list, list):
    """
    Reads a JSON file with the given feature and label headings

    Keyword Arguments:
    json_path -- the path to the JSON file to be read
    feature_heading -- the title of the object attribute that will be used as a feature
    label_heading -- the title of the object attribute that will be used as the label

    Returns:
    features -- the entire list of features from the JSON file
    labels -- all the labels for the feature instances from the JSON file
    """
    features = []
    labels = []
    with open(json_path, 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            features.append(data[feature_heading])
            labels.append(data[label_heading])

    return features, labels          

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

def split_train_test(original_path: str, train_path: str, test_path: str, num_instances: int, enc: str = 'utf-8'):
    """
    Manual splitting of training and test data. This method is deprecated due to the
    introduction of scikit-learn's methods such as train test split and stratified split validation

    Keyword arguments:
    original_path -- path to full dataset to be split
    train_path -- the path to the new training data to be formed
    test_path -- the path to the new testing data to be formed
    num_instances -- total number of data instances in the dataset
    enc -- the encoding to be used for reading and writing the files
    """
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

def overwrite_csv_column(csv_in_path: str, csv_out_path: str, column_idx: int, values_to_switch: tuple):
    """
    Takes a given csv and overwrites all the values within that column

    Keyword arguments:
    csv_in_path -- original csv whose column will be overwritten
    csv_out_path -- path to write the altered csv to
    column_idx -- index of the column to change, treated as a list element
    values_to_switch -- the value to find in the column and the value to replace it with
    """
    csv_in = open(csv_in_path, 'r')
    csv_out = open(csv_out_path, 'w', newline='')
    writer = csv.writer(csv_out)

    for row in csv.reader(csv_in):
        if row[column_idx] == values_to_switch[0]:
            row[column_idx] = values_to_switch[1]
        writer.writerow(row)
    
    csv_in.close()
    csv_out.close()