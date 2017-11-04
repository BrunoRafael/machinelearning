import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class LogisticRegression :
    def __init__ (self, n, main_class, prob_limit, coverage_limit):
        self._n = n
        self._main_class = main_class
        self._prob_limit = prob_limit
        self._coverage_limit = coverage_limit

    def fit (self, H, Y):
        d = H.shape[1] + 1
        self.coef_ = np.matrix([0 for i in range(1, d)]).T
        self._gradient_ascendent_runner(H, Y, self.coef_, self._n)
        
    def _gradient_ascendent_runner (self, H, Y, W, n):
        G = self._RSS_GRADIENT(H, Y, W)
        RSS_NORM = np.linalg.norm(G)
        while (RSS_NORM >= self._coverage_limit) :
            G = self._RSS_GRADIENT(H, Y, W)
            W = W + n * G
            RSS_NORM = np.linalg.norm(G)
        
        self._RSS_NORM = RSS_NORM
        self.coef_ = W
        
    
    def _RSS_GRADIENT (self, H, Y, W):
        G = []
        sig = 1 / (1 + np.exp(-np.dot(H, W)))
        for i in range(0, len(H.columns)):
            x = H.ix[:, i]
            gradient = np.dot(x.T, (Y - sig))
            G.append(gradient[0])
        
        return pd.DataFrame({'G' : G})
    
    def metrics(self, predictions, truth_values):
        error_count = 0
        correct = 0
        
        metrics = {}
        true_p, true_n, false_p, false_n = 0, 0, 0, 0
        for i in range(0, len(predictions)):
            if(predictions[i] == 1 and truth_values[i] == 0):
                false_p += 1
            elif(predictions[i] == 0 and truth_values[i] == 1):
                false_n += 1
            elif(predictions[i] == 1 and truth_values[i] == 1):
                true_p += 1
            else:
                true_n += 1
        
        total_points = true_p + true_n + false_p + false_n
        metrics['true_p'] = true_p
        metrics['true_n'] = true_n
        metrics['false_p'] = false_p
        metrics['false_n'] = false_n
        metrics['error'] = (false_n + false_p)/total_points
        metrics['accuracy'] = (true_p + true_n)/total_points
        metrics['precision'] = true_p / (true_p + false_p)
        metrics['recall'] = true_p / (true_p + false_n)
        metrics['f-measure'] = 2 * ((metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']))
        metrics['total_of_points'] = total_points
        
        return pd.DataFrame({"metrics" : metrics.keys(), "values": metrics.values()})

    def predict (self, X):
        result = {}
        
        score = np.dot(X, self.coef_)
        result = 1 / (1 + np.exp(-score))
        
        return np.greater_equal(pd.DataFrame(result), 0.5).astype(int)
    