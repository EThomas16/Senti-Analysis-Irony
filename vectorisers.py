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
    - LDA
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

import nlp_utils
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.ldamodel import LdaModel

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import naive_bayes, svm, ensemble

from decorators import reset_file

#np.set_printoptions(threshold=np.nan)

def generate_doc2vec_model(features: list, vec_size: int = 200) -> object:
    # documents = [TaggedDocument(doc, [idx]) for idx, doc in enumerate(features)]
    documents = []
    for idx, doc in enumerate(features):
        documents.append(TaggedDocument(doc, [idx]))
    # default arguments from: https://radimrehurek.com/gensim/models/doc2vec.html
    model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=1, workers=4)

    return model

def infer_vector_doc2vec(model: object, document: list, min_alpha: float) -> list:
    return model.infer_vector(document.split(' '), min_alpha=min_alpha)

def generate_doc2vec_vectors(model: object) -> list:
    doc_vectors = []
    for vector in model.docvecs.vectors_docs:
        doc_vectors.append(vector)

    return doc_vectors

if __name__ == "__main__":
    features, labels = nlp_utils.read_json(
        "Data/Sarcasm_Headlines_Dataset.json", "headline", "is_sarcastic")
    
    features = nlp_utils.stem_words(features)
    # model = generate_doc2vec_model(features, vec_size=300)
    # model.save("Data/models/headlines_test_model.d2v")
    # model = Doc2Vec.load("Data/models/headlines_test_model.d2v")    
    # doc_vectors = generate_doc2vec_vectors(model)
    bow_features = nlp_utils.vectorise_feature_list(features, vectoriser="bag-of-words")
    # tfidf_features = nlp_utils.vectorise_feature_list(features, vectoriser="tf-idf")

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # doc_vectors = scaler.fit_transform(np.array(doc_vectors))
    bow_features = np.array(bow_features.toarray())
    # tfidf_features = np.array(tfidf_features.toarray())
    # concatenated_features = nlp_utils.concatenate_features((doc_vectors, tfidf_features))
    # clf = naive_bayes.MultinomialNB()
    clf = svm.LinearSVC()
    # clf = LinearDiscriminantAnalysis()
    data = nlp_utils.TextData()
    data.X = bow_features
    data.y = np.array(labels)
    
    # shuffle_split = StratifiedShuffleSplit(n_splits=1)
    seeds = [6, 40, 32, 17, 19]
    all_scores = []; all_times = []
    upper_bound = 11
    for c_val in range(10, upper_bound):
        print(f"Current C: {c_val}")
        avg_scores = []; avg_times = []
        clf = svm.LinearSVC(C=c_val)

        for idx, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=2, random_state=seed)
            scores, times = data.stratify(skf, clf, eval_metric="f-score")
            avg_scores.extend([scores[0], scores[1]])
            avg_times.extend([times[0], times[1]])
 
        all_scores.append(sum(avg_scores)/upper_bound)
        all_times.append(sum(avg_times)/upper_bound)

    results_file_path = "Results/News-Headlines/Bag-of-Words/bow_svm_stemming.csv"
    # reset_file(results_file_path, "Classifier,Score,C-value,Execution Time")
    for idx, (score, time) in enumerate(zip(all_scores, all_times)):
        nlp_utils.write_results_stratification(results_file_path, "Liblinear SVM", score, idx+1, time)