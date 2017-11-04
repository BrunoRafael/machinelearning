import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class LinearRegression:
    def __init__(self, H, Y, alpha):
        self.H_ = H
        self.Y_ = Y
        self.alpha_ = alpha
        self.coef_ = self._generate_model(H, Y, alpha);
    
    def _generate_model(self, H, Y, alpha):
        #y = HW + e; This array is the vector parameters W
        d = H.shape[1] + 1
        W = np.matrix([0 for i in range(1, d)]).T
        return self._gradient_descendent_runner(H, Y, W, alpha)
    
    def _gradient_descendent_runner(self, H, Y, W, alpha):
        RSS_NORM = np.linalg.norm(self._RSS_GRADIENT(H, Y, W, alpha))
        #for i in range(0, num_interactions):
        while (RSS_NORM >= 1e-15) :
            W = self._step_gradient(H, Y, W, alpha)
            RSS_NORM = np.linalg.norm(self._RSS_GRADIENT(H, Y, W, alpha))
        
        print "MSE: %f " %(self._compute_mse(H, Y, W))
        return W

    def _step_gradient(self, H, Y, W, alpha):   
        new_W = W + self._RSS_GRADIENT(H, Y, W, alpha)
        return new_W
    
    def _RSS_GRADIENT (self, H, Y, W, alpha):
        rate = 2 * alpha
        M1 = rate * H.T

        K = np.dot(H, W)
        M2 = Y-K

        return np.dot(M1, M2)

    def _compute_mse(self, H, Y, W):
        matrix = Y - np.dot(H, W)
        s = sum(np.array(matrix).reshape(-1,).tolist())

        return s

    def predict(self, X):
        y_pred = np.dot(X, self.coef_)
        return y_pred