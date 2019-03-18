import nlp_utils
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.model_selection import cross_val_score, train_test_split
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
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = generate_doc2vec_model(X_train)

    doc_vectors = []
    for instance in X_test:
        doc_vectors.append(infer_vector_doc2vec(model, instance, model.min_alpha))

    clf = naive_bayes.MultinomialNB()
    score = cross_val_score(clf, doc_vectors, y_test)
    print(str(np.mean(score)) + str(np.std(score)))