import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

def read_json(json_path: str):
    features = []
    labels = []
    with open(json_path, 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            # print(f"{data['is_sarcastic']}\t{data['headline']}")
            features.append(data['headline'])
            labels.append(data['is_sarcastic'])

    return features, labels

def sklearn_bow(
    features: list, max_feats: int = 50000, stop_word_lang: str = "english"):

    vectoriser = CountVectorizer(
        max_features=max_feats,
        stop_words=stop_word_lang
    )

    return vectoriser.fit_transform(features)

features, labels = read_json("Data/Sarcasm_Headlines_Dataset.json")

X = features
y = np.array(labels)
# print(f"X: {X.shape}\tY: {y.shape}")

X = sklearn_bow(features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)
# print(f"{X_train.shape} {y_train.shape}\n{X_test.shape} {y_test.shape}")

clf = LinearSVC()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, predicted)}")
print(metrics.classification_report(y_test, predicted))