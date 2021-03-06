import os
import re
import csv
import json
import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class TextData():
    def __init__(
        self, train_path: str = "", test_path: str = "", 
        lbl: str = "label", single_file_path: str = "",
        stopwords = ["a", "the", "is"]):

        self.stopwords = stopwords
        self.eval_metric_methods = {
            "accuracy" : metrics.accuracy_score,
            "f1-score" : metrics.f1_score,
            "conf matrix" : metrics.confusion_matrix
        }

        if single_file_path:
            df = pd.read_csv(single_file_path)
            self.X = np.array(df.drop([lbl], 1))
            self.y = np.array(df.iloc[:, -1])
            self.X_train = []; self.y_train = []
            self.X_test = []; self.y_test = []
            return

        if not train_path and not test_path:
            return

        self.read_train_test_csvs(train_path, test_path, lbl)

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
        sentence -- the sentence to be cleaned

        Returns:
        The sentence that has been changed to lowercase, as well as cleaned of any unnecessary characters
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
        self, split_algorithm: object, clf: object, eval_metric: str = "accuracy"):
        """
        Stratifies the data using the provided splitting algorithm. This is used to split datasets
        into training and test, as well as in cross-validation of datasets

        Keyword arguments:
        split_algorithm -- the stratification algorithm to use to split the data (scikit-learn tested)
        clf -- the classifier to use to classify the splits
        """
        score_list = []
        execution_times = []
        for train_idx, test_idx in split_algorithm.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx] 
            start_time = time.time()
            clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)
            execution_times.append(time.time() - start_time)
            score_list.append(self.eval_metric_methods[eval_metric](y_test, predicted))

        return score_list, execution_times

class DocumentEmbeddings():
    def __init__(self, features: list, vec_size: int):
        self.create_model(features, vec_size)

    def create_model(self, features: list, vec_size: int):
        """
        Generates a doc2vec model, storing the model as an attribute of the class

        Keyword Arguments:
        features -- all features for a corpus
        vec_size -- length of each document vector
        """
        documents = []
        for idx, doc in enumerate(features):
            documents.append(TaggedDocument(doc, [idx]))
        
        self.model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=1, workers=4)

    def vectorise(self, normalise_range: tuple = None):
        """
        Vectorises document vectors from the trained model and normalises them

        Keyword arguments:
        normalise_range -- values within which to normalise each vector

        Returns:
        An array of the trained document vectors
        """
        doc_vectors = []
        for vector in self.model.docvecs.vectors_docs:
            doc_vectors.append(vector)

        if type(normalise_range) is tuple:
            scaler = MinMaxScaler(feature_range=normalise_range)
            doc_vectors = scaler.fit_transform(np.array(doc_vectors))

        return doc_vectors

    def infer_vector_doc2vec(self, document: list, min_alpha: float) -> list:
        """
        Used to infer a vector from a single document, for testing and domain transferral purposes

        Keyword arguments:
        document -- the document to be vectorised
        min_alpha -- the minimum learning rate during training

        Returns:
        The inferred vector for the document
        """
        return self.model.infer_vector(document.split(' '), min_alpha=min_alpha)

def __load_vectoriser(max_feats: int, ngrams: tuple, stop_word_lang: str, vectoriser: str) -> object:
    """
    Loads the specified vectoriser using the provided arguments

    Keyword arguments:
    max_feats -- the maximum number of features for a data instance
    stop_word_lang -- the language to use as the stop word dictionray
    vectoriser -- the vectoriser type to use, either bag-of-words of tf-idf

    Returns:
    A vectoriser object of the specified type
    """
    if vectoriser == "bag-of-words":
        vectoriser = CountVectorizer(
            max_features=max_feats, 
            ngram_range=ngrams,
            stop_words=stop_word_lang)
    elif vectoriser == "tf-idf":
        vectoriser = TfidfVectorizer(
            max_features=max_feats, 
            ngram_range=ngrams,
            stop_words=stop_word_lang)
    else:
        print("Invalid vectoriser given, assigning bag-of-words as default")
        vectoriser = CountVectorizer(
            max_features=max_feats,
            ngram_range=ngrams, 
            stop_words=stop_word_lang)

    return vectoriser

def __load_stemmer(stemmer_type: str) -> object:
    """
    Loads a stemmer based on the provided argument

    Keyword arguments:
    stemmer_type -- the type of stemmer to be used, either Porter or Lancaster

    Returns:
    A stemmer object of the specified type
    """
    if stemmer_type == "Porter":
        stemmer = PorterStemmer()
    elif stemmer_type == "Lancaster":
        stemmer = LancasterStemmer()

    return stemmer

def __load_lemmatiser(lemmatiser_type: str) -> object:
    """
    Loads a lemmatiser based on the provided argument

    Keyword arguments:
    lemmatiser_type -- the type of lemmatiser to be used, currently only WordNet

    Returns:
    A lemmatiser object of the specified type
    """
    if lemmatiser_type == "WordNet":
        lemmatiser = WordNetLemmatizer()

    return lemmatiser

def concatenate_features(feature_sets: tuple, axis: int = 1):
    """
    Concatenates two feature sets along a specified axis

    Keyword arguments:
    feature_sets -- the two feature arrays to be merged
    axis -- the axis which they will be merged along
    """
    return np.concatenate(feature_sets, axis=axis)

def stem_words(features: list, stemmer_type: str = "Porter") -> list:
    """
    Stems words using the specified stemmer type, changing them to their root
    i.e. gaming -> game
    
    Keyword arguments:
    features -- 2D array of all documents in a dataset
    stemmer_type -- the stemmer to be used i.e. Porter, Lancaster etc. Default: Porter
    
    Returns:
    The stemmed features that have been stemmed using the specified stemmer
    """
    stemmer = __load_stemmer(stemmer_type)
    new_features = []
    for feature_set in features:
        new_features.append(stemmer.stem(feature_set))

    return new_features

def lemmatise_words(features: list, lemmatiser_type: str = "WordNet") -> list:
    """
    Lemmatises words through removing suffixes and prefixes whilst maintaining
    the meaning of the word

   Keyword arguments:
   features -- 2D array of all documents in a dataset
   lemmatiser_type -- the lemmatiser to be used i.e. WordNet, which is the default

   Returns:
   Array of the lemmatised features to be used in feature extraction
    """
    lemmatiser = __load_lemmatiser(lemmatiser_type)
    new_features = []
    for feature_set in features:
        new_features.append(lemmatiser.lemmatize(feature_set))
        
    return new_features

def vectorise_feature_file(
    input_path: str, output_path: str, lbl: str,
    max_feats: int = 50000, ngrams: tuple = (1, 2), stop_word_lang: str = "english", vectoriser: str = "bag-of-words"):
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

    vectoriser = __load_vectoriser(max_feats, ngrams, stop_word_lang, vectoriser)

    fts = vectoriser.fit_transform(features.flatten())
    with open(output_path, 'a', encoding='utf-8') as test:
        for feature_name in vectoriser.get_feature_names():
            test.write(f"{feature_name},")
        test.write("label\n")
        for ft, label in zip(fts.toarray(), labels):
            for instance in ft:
                test.write(f"{instance},")
            test.write(f"{label}\n") 

def vectorise_feature_list(
    features: list, max_feats: int = 50000, ngrams: tuple = (1, 2), stop_word_lang: str = "english", vectoriser: str = "bag-of-words") -> object:
    """
    Generates bag-of-words features from a list of sentences

    Keyword arguments:
    features -- list of all the sentences to be converted
    max_feats -- the maximum number of words that can be used from all sentences in the bag of words
    stop_word_lang -- the language of the text

    Returns:
    A vectoriser that has been fitted to the provided sentences
    """
    vectoriser = __load_vectoriser(max_feats, ngrams, stop_word_lang, vectoriser)

    return vectoriser.fit_transform(features)

def vectorise_lda(features: list, components: int = 10, learn_decay: float = 0.7, rand_state: int = 40) -> list:
    """
    Uses the probabilstic approach of LDA to transform pre-vectorised features

    Keyword arguments:
    features -- list of all the sentences to be converted
    components -- number of topics to be used in generating LDA features
    learn_decay -- the speed at which the learning rate decreases
    rand_state -- the random state used in initialising the LDA object

    Returns:
    The transformed features using the parameters specified for the LDA object
    """
    lda = LatentDirichletAllocation(n_components=components, learning_decay=learn_decay, random_state=rand_state)
    return lda.fit_transform(features)

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

def write_results_stratification(result_file: str, clf: str, score: float, changed_parameter: int, execution_time: float):
    """
    Used to format the stratification results in a specific manner

    Keyword arguments:
    result_file -- the file to write the results to
    clf -- the classifier used to obtain the results
    score -- the score obtained from classification
    changed_parameter -- the value that has been changed throughout testing i.e. Alpha for Naive Bayes
    execution_time -- time taken for fitting and predicting using the given classifier
    """
    with open(result_file, 'a') as results:
        results.write(f"{clf},{score},{changed_parameter},{execution_time}\n")