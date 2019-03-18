import nlp_utils
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import naive_bayes, svm

def generate_doc2vec_model(features: list, vec_size: int = 5) -> object:
    documents = [TaggedDocument(doc, [idx]) for idx, doc in enumerate(features)]
    # default arguments from: https://radimrehurek.com/gensim/models/doc2vec.html
    model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=1, workers=4)

    return model

def infer_vector_doc2vec(model: object, document: list, min_alpha: float) -> list:
    return model.infer_vector(document.split(' '), min_alpha=min_alpha)

if __name__ == "__main__":
    features, labels = nlp_utils.read_json(
        "Data/Sarcasm_Headlines_Dataset.json", "headline", "is_sarcastic")
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    # model = generate_doc2vec_model(features)

    # model.save("Data/models/headlines_test_model.d2v")
    model = Doc2Vec.load("Data/models/headlines_test_model.d2v")

    doc_vectors = []
    for instance in features:
        doc_vectors.append(infer_vector_doc2vec(model, instance, model.min_alpha))

    # clf = naive_bayes.MultinomialNB()
    clf = svm.LinearSVC()
    data = nlp_utils.TextData()
    data.X = np.array(doc_vectors)
    data.y = np.array(labels)
    
    shuffle_split = StratifiedShuffleSplit(n_splits=1)
    skf = StratifiedKFold(n_splits=10)
    data.stratify(shuffle_split, clf)