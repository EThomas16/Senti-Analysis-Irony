'''
Code to reproduce the analyses in our ACL 2014 paper: 

    Humans Require Context to Infer Ironic Intent (so Computers Probably do, too)
        Byron C Wallace, Do Kook Choe, Laura Kertz, and Eugene Charniak

Made possible by support from the Army Research Office (ARO), grant# 528674 
"Sociolinguistically Informed Natural Lanuage Processing: Automating Irony Detection"

Contact: Byron Wallace (byron.wallace@gmail.com)

The main methods of interest are context_stats and ml_bow. 
'''

''' built-ins. '''
import os
import pdb
import sys
import collections
from collections import defaultdict
import re
import itertools
import sqlite3

''' dependencies: sklearn, numpy, statsmodels '''
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics, ensemble, tree, svm

import statsmodels.api as sm
import nlp_utils

class DBConnection():
    def __init__(self, path_to_db):
        self.conn = sqlite3.connect(path_to_db)
        self.cursor = self.conn.cursor()
        
        self.labelers_of_interest = [2,4,5,6]
        self.labeler_id_str = self.__make_sql_list_str(self.labelers_of_interest)

    def __make_sql_list_str(self, ls):
        return "(" + ",".join([str(x_i) for x_i in ls]) + ")"

    def __grab_single_element(self, result_set, COL=0):
        return [x[COL] for x in result_set]
    
    def get_all_comment_ids(self):
        return self.__grab_single_element(self.cursor.execute(
                    '''select distinct comment_id from irony_label where labeler_id in %s;''' % 
                        self.labeler_id_str)) 
    
    def get_ironic_comment_ids(self):       
        self.cursor.execute(
            '''select distinct comment_id from irony_label 
                where forced_decision=0 and label=1 and labeler_id in %s;''' % 
                self.labeler_id_str)
    
        ironic_comments = self.__grab_single_element(self.cursor.fetchall())
        return ironic_comments
    
    def show_tables(self):
        """
        Returns all tables from the database file

        Returns:
        All tables from the loaded database
        """
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        return self.__grab_single_element(self.cursor.fetchall())  
    
    def custom_query(self, query):
        """
        Allows for custom queries to be run, providing an SQL query as a string

        Keyword arguments:
        query -- the query to be run on the database

        Returns:
        All results from the provided query on the loaded database
        """
        self.cursor.execute(
            query
        )
        return self.__grab_single_element(self.cursor.fetchall())
    
    def context_stats(self):
        """
        Section 4, Eq (1) in the paper.
    
        > irony_stats.context_stats()
        ==============================================================================
        Dep. Variable:                      y   No. Observations:                 3550
        Model:                          Logit   Df Residuals:                     3548
        Method:                           MLE   Df Model:                            1
        Date:                Sat, 26 Apr 2014   Pseudo R-squ.:                 0.06012
        Time:                        05:39:34   Log-Likelihood:                -2240.5
        converged:                       True   LL-Null:                       -2383.8
                                                LLR p-value:                 2.670e-64
        ==============================================================================
                         coef    std err          z      P>|z|      [95.0% Conf. Int.]
        ------------------------------------------------------------------------------
        const         -0.7108      0.040    -17.961      0.000        -0.788    -0.633
        x1             1.5081      0.093     16.223      0.000         1.326     1.690
        ==============================================================================
        """
        all_comment_ids = self.get_all_comment_ids()
    
        # pre-context / forced decisions
        forced_decisions = self.__grab_single_element(self.cursor.execute(
                    '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' % 
                        self.labeler_id_str)) 
    
        for labeler in self.labelers_of_interest:
            labeler_forced_decisions = self.__grab_single_element(self.cursor.execute(
                    '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id = %s;''' % 
                        labeler))
    
            all_labeler_decisions = self.__grab_single_element(self.cursor.execute(
                    '''select distinct comment_id from irony_label where forced_decision=0 and labeler_id = %s;''' % 
                        labeler))
    
            p_labeler_forced = float(len(labeler_forced_decisions))/float(len(all_labeler_decisions))
            print("labeler %s: %s" % (labeler, p_labeler_forced))
    
        p_forced = float(len(forced_decisions)) / float(len(all_comment_ids))
    
        # now look at the proportion forced for the ironic comments
        ironic_comments = self.get_ironic_comment_ids()
        ironic_ids_str = self.__make_sql_list_str(ironic_comments)
        forced_ironic_ids =  self.__grab_single_element(self.cursor.execute(
                    '''select distinct comment_id from irony_label where 
                            forced_decision=1 and comment_id in %s and labeler_id in %s;''' % 
                                    (ironic_ids_str, self.labeler_id_str))) 
    
        ''' regression bit: construct target vector + design matrix  '''
        X,y = [],[]
    
        for c_id in all_comment_ids:
            if c_id in forced_decisions:
                y.append(1.0)
            else:
                y.append(0.0)
    
            if c_id in ironic_comments:
                X.append([1.0])
            else:
                X.append([0.0])
    
        X = sm.add_constant(X, prepend=True)
        logit_mod = sm.Logit(y, X)
        logit_res = logit_mod.fit()
        
        print(logit_res.summary())
        return logit_res
    
    def ml_bow(self, show_features=False):
        """
        Section 5, Eq (2) in the paper. 
    
        > irony_stats.ml_bow()
        Optimization terminated successfully.
                 Current function value: 0.611578
                 Iterations 5
                                   Logit Regression Results                           
        ==============================================================================
        Dep. Variable:                      y   No. Observations:                 1949
        Model:                          Logit   Df Residuals:                     1946
        Method:                           MLE   Df Model:                            2
        Date:                Sun, 04 May 2014   Pseudo R-squ.:                 0.06502
        Time:                        08:24:43   Log-Likelihood:                -1192.0
        converged:                       True   LL-Null:                       -1274.9
                                                LLR p-value:                 9.956e-37
        ==============================================================================
                         coef    std err          z      P>|z|      [95.0% Conf. Int.]
        ------------------------------------------------------------------------------
        const         -1.3284      0.088    -15.170      0.000        -1.500    -1.157
        x1             0.9404      0.108      8.723      0.000         0.729     1.152
        x2             0.7573      0.106      7.149      0.000         0.550     0.965
        ==============================================================================

        TWO NOTES:
        1 A small bug in the original SQL code here resulted in a slightly different value for 
        x2; however the resutls are qualitatively the same as in the paper.
        2 In any case, this result will vary slightly because we are using stochastic gradient 
        descent! Still, the x2 estimate and CI (which is of interest) should be quite close.
        """
        X, all_comment_ids, ironic_comment_ids, forced_decision_ids = self.get_bow_feature_labels()
        kf = KFold(random_state=len(y), n_splits=5, shuffle=True)
        X_context, y_mistakes = [], []
        recalls, precisions = [], []
        Fs = []
        top_features = []
        for train, test in kf.split(X, y):
            train_ids = self.__get_entries(all_comment_ids, train)
            test_ids = self.__get_entries(all_comment_ids, test)
            y_train = self.__get_entries(y, train)
            y_test = self.__get_entries(y, test)
    
            X_train, X_test = X[train], X[test]
            svm = SGDClassifier(loss="hinge", penalty="l2", class_weight="balanced", alpha=.01)
            #pdb.set_trace()
            parameters = {'alpha':[.001, .01,  .1]}
            clf = GridSearchCV(svm, parameters, scoring='f1')
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            #precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_test, preds)
            tp, fp, tn, fn = 0,0,0,0
            N = len(preds)
    
            for i in range(N):
                cur_id = test_ids[i]
                irony_indicator = 1 if cur_id in ironic_comment_ids else 0
                forced_decision_indicator = 1 if cur_id in forced_decision_ids else 0
                # so x1 is the coefficient for forced decisions (i.e., context); 
                # x2 is the coeffecient for irony (overall)
                X_context.append([irony_indicator, forced_decision_indicator])
    
                y_i = y_test[i]
                pred_y_i = preds[i]
    
                if y_i == 1:
                    # ironic
                    if pred_y_i == 1:
                        # true positive
                        tp += 1 
                        y_mistakes.append(0)
                    else:
                        # false negative
                        fn += 1
                        y_mistakes.append(1)
                else:
                    # unironic
                    if pred_y_i == -1:
                        # true negative
                        tn += 1
                        y_mistakes.append(0)
                    else:
                        # false positive
                        fp += 1
                        y_mistakes.append(1)
    
            recall = tp/float(tp + fn)
            precision = tp/float(tp + fp)
            recalls.append(recall)
            precisions.append(precision)
            f1 = 2* (precision * recall) / (precision + recall)
            Fs.append(f1)
    
        X_context = sm.add_constant(X_context, prepend=True)
        logit_mod = sm.Logit(y_mistakes, X_context)
        logit_res = logit_mod.fit()
    
        print(logit_res.summary())
    
    def grab_comments(self, comment_id_list, verbose=False):
        comments_list = []
        for comment_id in comment_id_list:
            self.cursor.execute("select text from irony_commentsegment where comment_id='%s' order by segment_index" % comment_id)
            segments = self.__grab_single_element(self.cursor.fetchall())
            comment = " ".join(segments)
            if verbose:
                print(comment)
            comments_list.append(comment.encode('utf-8').strip())
        return comments_list
    
    def __get_entries(self, a_list, indices):
        return [a_list[i] for i in indices]
    
    def get_labeled_thrice_comments(self):
        """ 
        get all ids for comments labeled >= 3 times 
        """
        self.cursor.execute(
            '''select comment_id from irony_label group by comment_id having count(distinct labeler_id) >= 3;'''
        )
        thricely_labeled_comment_ids = self.__grab_single_element(self.cursor.fetchall())
        return thricely_labeled_comment_ids
    
    def get_feature_labels(self, vectoriser="bag-of-words", max_feats=50000, ngrams=(1, 2)):
        """
        Returns vectorised documents, vectorising using the provided parameters

        Keyword arguments:
        vectoriser -- the feature extraction method to be used, either TF-IDF or bag-of-words
        max_feats -- the maximum feature length for each vectorised document
        ngrams -- the range of ngrams formed from the data

        Returns:
        The features, labels and non-processed documents for use in other vectorisers
        """
        all_comment_ids = self.get_labeled_thrice_comments()
    
        ironic_comment_ids = self.get_ironic_comment_ids()
        #ironic_ids_str = _make_sql_list_str(ironic_comments)
    
        forced_decision_ids = self.__grab_single_element(self.cursor.execute(
                    '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' % 
                        self.labeler_id_str)) 
    
        comment_texts, y = [], []
        for id_ in all_comment_ids:
            comment_texts.append(self.grab_comments([id_])[0])
            if id_ in ironic_comment_ids:
                y.append(1)
            else:
                y.append(-1)
    
        # adding some features here; just adding them as tokens,
        # which is admittedly kind of hacky.
        emoticon_RE_str = '(?::|;|=)(?:-)?(?:\)|\(|D|P)'
        question_mark_RE_str = '\?'
        exclamation_point_RE_str = '\!'
        # any combination of multiple exclamation points and question marks
        interrobang_RE_str = '[\?\!]{2,}'
    
        for i, comment in enumerate(comment_texts):
            #pdb.set_trace()
            comment = str(comment)
            if len(re.findall(r'%s' % emoticon_RE_str, comment)) > 0:
                comment += " PUNCxEMOTICON"
            if len(re.findall(r'%s' % exclamation_point_RE_str, comment)) > 0:
                comment += " PUNCxEXCLAMATION_POINT"
            if len(re.findall(r'%s' % question_mark_RE_str, comment)) > 0:
                comment += " PUNCxQUESTION_MARK"
            if len(re.findall(r'%s' % interrobang_RE_str, comment)) > 0:
                comment += " PUNCxINTERROBANG"
            
            if any([len(s) > 2 and str.isupper(s) for s in comment.split(" ")]):
                comment = comment + " PUNCxUPPERCASE" 
            
            comment_texts[i] = comment

        X = nlp_utils.vectorise_feature_list(comment_texts, vectoriser=vectoriser, max_feats=max_feats, ngrams=ngrams)
        
        return X, y, all_comment_ids, ironic_comment_ids, forced_decision_ids, comment_texts
        
### assumes the database file is local!
# download this from: 
# email me (byron.wallace@gmail.com) if this url
# fails.

if __name__ == "__main__":
    db_handler = DBConnection("ironate.db")
    print("{} Non-Ironic Instances".format(len(db_handler.get_all_comment_ids()) - len(db_handler.get_ironic_comment_ids())))
    print("{} Total Instances".format(len(db_handler.get_all_comment_ids())))
    X, y, _, _, _, raw_features = db_handler.get_feature_labels(vectoriser="bag-of-words", max_feats=2900, ngrams=(1, 3))
    
    doc_embeddings = nlp_utils.DocumentEmbeddings(raw_features, vec_size=980)
    doc_vectors = doc_embeddings.vectorise(normalise_range=(0, 1))
    
    scaler = MinMaxScaler(feature_range=(0,1))

    data = nlp_utils.TextData()
    data.X = np.concatenate((doc_vectors, np.array(X.toarray())), axis=1)
    data.X = scaler.fit_transform(data.X)
    data.y = np.array(y)
    
    seeds = [6, 40, 32, 17, 19]
    all_scores = []; all_times = []; params = []
    for c_val in range(21, 51):
        c_val *= 100
        print(f"Current hyper-parameter: {c_val}")
        avg_scores = []; avg_times = []
        clf = svm.SVC(kernel='rbf', C=c_val, gamma='auto')
        # clf = MultinomialNB(alpha=c_val)

        for idx, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=2, random_state=seed)
            scores, times = data.stratify(skf, clf, eval_metric="f1-score")
            avg_scores.extend([scores[0], scores[1]])
            avg_times.extend([times[0], times[1]])
    
        all_scores.append(sum(avg_scores)/len(avg_scores))
        all_times.append(sum(avg_times)/len(avg_times))
        params.append(c_val)
        
    results_file_path = "../../Results/ACL-Irony/Concatenated/bow_d2v_svm.csv"
    # reset_file(results_file_path, "Classifier,Score,C-Value,Execution Time")
    for (score, time, param) in zip(all_scores, all_times, params):
        nlp_utils.write_results_stratification(results_file_path, "RBF SVM", score, param, time)
