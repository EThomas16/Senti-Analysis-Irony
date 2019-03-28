import nlp_utils
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import normalize, Normalizer, MinMaxScaler
from sklearn import naive_bayes, svm, ensemble

from decorators import reset_file

#np.set_printoptions(threshold=np.nan)

def generate_doc2vec_model(features: list, vec_size: int = 5) -> object:
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
    model = generate_doc2vec_model(features)
    # model.save("Data/models/headlines_test_model.d2v")
    # model = Doc2Vec.load("Data/models/headlines_test_model.d2v")    
    doc_vectors = generate_doc2vec_vectors(model)
    # bow_features = nlp_utils.vectorise_feature_list(features, vectoriser="bag-of-words")
    tfidf_features = nlp_utils.vectorise_feature_list(features, vectoriser="tf-idf")

    doc_vectors = np.array(doc_vectors)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(doc_vectors)
    # doc_vectors = normalize(doc_vectors, axis=1)
    # doc_vectors = Normalizer(copy=False).fit_transform(doc_vectors)
    # bow_features = np.array(bow_features.toarray())
    tfidf_features = np.array(tfidf_features.toarray())
    concatenated_features = nlp_utils.concatenate_features((doc_vectors, tfidf_features))

    clf = naive_bayes.BernoulliNB()
    # clf = svm.LinearSVC()
    data = nlp_utils.TextData()
    data.X = concatenated_features
    data.y = np.array(labels)
    
    shuffle_split = StratifiedShuffleSplit(n_splits=1)
    skf = StratifiedKFold(n_splits=10)
    scores, times = data.stratify(skf, clf, eval_metric="f-score")
    
    results_file_path = "Results/concatenated_results_tfidf_d2v_skf_stem.csv"
    # reset_file(results_file_path, "Classifier,Score,Fold,Execution Time")
    for idx, (score, time) in enumerate(zip(scores, times)):
        nlp_utils.write_results_stratification(results_file_path, "Bernoulli NB", score, idx+1, time)