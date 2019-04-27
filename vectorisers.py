import time
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import StratifiedKFold
from sklearn import naive_bayes, svm, ensemble

import nlp_utils
from decorators import reset_file

def write_results(output_file: str, features: list, labels: list):
    """
    This script has been used for testing with the news headlines dataset

    Keyword arguments:
    output_file -- the file to write the results of the experiments to
    features -- the 2D array of documents
    labels -- the subsequent labels for each document, whether it is sarcastic or not
    """
    features = nlp_utils.stem_words(features)
    features = nlp_utils.lemmatise_words(features)
    # doc_embeddings = nlp_utils.DocumentEmbeddings(features, vec_size=1680)
    # doc_vectors = doc_embeddings.vectorise(normalise_range=(0, 1))
    vectorised_features = nlp_utils.vectorise_feature_list(features, max_feats=11000, ngrams=(1, 3), vectoriser="bag-of-words")

    # lda_features = nlp_utils.vectorise_lda(vectorised_features, components=n_components)
    # concatenated_features = np.concatenate((
    #     doc_vectors, np.array(vectorised_features.toarray())), axis=1)

    data = nlp_utils.TextData()
    data.X = vectorised_features
    data.y = np.array(labels)
            
    all_scores = []; all_times = []; hyper_parameters = []

    for val in range(1, 21):
        val /= 20

        avg_scores = []; avg_times = []

        clf = naive_bayes.MultinomialNB(alpha=val)

        seeds = [6, 40, 32, 17, 19]
        for seed in seeds:
            skf = StratifiedKFold(n_splits=2, random_state=seed)
            scores, times = data.stratify(skf, clf, eval_metric="f1-score")

            avg_scores.extend([scores[0], scores[1]])
            avg_times.extend([times[0], times[1]])

        all_scores.append(sum(avg_scores)/len(avg_scores))
        all_times.append(sum(avg_times)/len(avg_times))
        hyper_parameters.append(val)

    # uncomment this to format a file in the desired manner
    # reset_file(output_file, "Classifier,Score,Alpha,Execution Time")
    for (score, _time, val) in zip(all_scores, all_times, hyper_parameters):
        nlp_utils.write_results_stratification(output_file, "Multinomial NB", score, val, _time)

if __name__ == "__main__":
    features, labels = nlp_utils.read_json(
        "Data/Sarcasm_Headlines_Dataset.json", "headline", "is_sarcastic")
    
    write_results("Results/News-Headlines/Bag-of-Words/bow_pre-processing.csv", features, labels)
    
    
