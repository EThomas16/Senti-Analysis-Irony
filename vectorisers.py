import nlp_utils
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import normalize, Normalizer
from sklearn import naive_bayes, svm, ensemble

np.set_printoptions(threshold=np.nan)

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

if __name__ == "__main__":
    features, labels = nlp_utils.read_json(
        "Data/Sarcasm_Headlines_Dataset.json", "headline", "is_sarcastic")
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = generate_doc2vec_model(features)
    # model.save("Data/models/headlines_test_model.d2v")
    # model = Doc2Vec.load("Data/models/headlines_test_model.d2v")

    doc_vectors = []
    for vector in model.docvecs.vectors_docs:
        doc_vectors.append(vector)

    bow_features = nlp_utils.sklearn_bow_list(features)

    doc_vectors = np.array(doc_vectors)
    # doc_vectors = normalize(doc_vectors, axis=1)
    # doc_vectors = Normalizer(copy=False).fit_transform(doc_vectors)
    bow_features = np.array(bow_features.toarray())
    print(doc_vectors.shape)
    print(bow_features.shape)
    concatenated_features = []
    concatenated_features = np.concatenate((doc_vectors, bow_features), axis=1)

    concatenated_features = np.array(concatenated_features)

    clf = naive_bayes.BernoulliNB()
    # clf = svm.LinearSVC()
    data = nlp_utils.TextData()
    data.X = concatenated_features
    data.y = np.array(labels)
    
    shuffle_split = StratifiedShuffleSplit(n_splits=1)
    skf = StratifiedKFold(n_splits=10)
    score = data.stratify(shuffle_split, clf)
    print(score)