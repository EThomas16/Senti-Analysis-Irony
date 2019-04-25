"""
Experiments TODO:
    - Feature extraction
    - Pre-processing
    - Classifiers

Datasets:
    - News Headlines
    - IAC
    - ACL-Irony

Feature extraction:
    - Bag-of-words/TF-IDF
    - doc2vec
    - LDA: makes tf-idf worse???
    - Text blobs
    + Concatenation

Pre-processing:
    - Stemming
    - Stop word removal
    - Punctuation removal
    - Lemmatisation
    + Combinations of each

Classifiers:
    - Naive Bayes
        - Multinomial
        - Bernoulli
    - SVM
        - Linear
        - Liblinear
        - RBF (+ other non-linear)
"""

import time
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.ldamodel import LdaModel

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import naive_bayes, svm, ensemble

import nlp_utils
from decorators import reset_file

def write_results(output_file: str, features: list, labels: list):
    # features = nlp_utils.stem_words(features)
    # doc_embeddings = nlp_utils.DocumentEmbeddings(features, vec_size=300)
    # doc_vectors = doc_embeddings.vectorise(normalise_range=(0, 1))
    # tfidf_features = nlp_utils.vectorise_feature_list(features, max_feats=8100, ngrams=(1, 3), vectoriser="tf-idf")
    for n_components in range(2, 11):
        vectorised_features = nlp_utils.vectorise_feature_list(features, max_feats=8100, ngrams=(1, n_components), vectoriser="tf-idf")
        # lda_features = nlp_utils.vectorise_lda(doc_vectors, components=n_components)
        # bow_features = np.array(bow_features.toarray())
        # tfidf_features = np.array(tfidf_features.toarray())
        # concatenated_features = nlp_utils.concatenate_features((doc_vectors, tfidf_features))
        data = nlp_utils.TextData()
        data.X = vectorised_features
        data.y = np.array(labels)
        
        # shuffle_split = StratifiedShuffleSplit(n_splits=1)
        
        all_scores = []; all_times = []; hyper_parameters = []

        for val in range(100, 101):
            val /= 100
            print(f"Current hyper-parameter: {val}")
            avg_scores = []; avg_times = []
            # clf = svm.SVC(kernel='rbf', C=5000, gamma='auto')
            # clf = svm.LinearSVC(C=val)
            clf = naive_bayes.MultinomialNB(alpha=val)

            seeds = [6, 40, 32, 17, 19]
            for seed in seeds:
                skf = StratifiedKFold(n_splits=2, random_state=seed)
                scores, times = data.stratify(skf, clf, eval_metric="f1-score")

                avg_scores.extend([scores[0], scores[1]])
                avg_times.extend([times[0], times[1]])

            all_scores.append(sum(avg_scores)/len(avg_scores))
            all_times.append(sum(avg_times)/len(avg_times))
            hyper_parameters.append(n_components)

        # TODO: delete this file after using
        # reset_file(output_file, "Classifier,Score,upper bound ngrams,Execution Time")
        for (score, _time, val) in zip(all_scores, all_times, hyper_parameters):
            nlp_utils.write_results_stratification(output_file, "Multinomial NB", score, val, _time)

if __name__ == "__main__":
    features, labels = nlp_utils.read_json(
        "Data/Sarcasm_Headlines_Dataset.json", "headline", "is_sarcastic")
    
    write_results("Results/News-Headlines/TF-IDF/tfidf_ngrams.csv", features, labels)
    
    
