
from sklearn.svm import SVC
from sklearn import grid_search
import numpy as np
from sklearn import cross_validation as cs
from sklearn.externals import joblib
# from picPreprocessing import loadPredict
import warnings
import time

def load_data():
    dataset = np.loadtxt('/Users/wanglang/Desktop/pic4/train_data.txt',delimiter=',')
    return dataset

def cross_validation():
    dataset = load_data();
    row,col = dataset.shape;
    X = dataset[:,:col-1]
    Y = dataset[:,-1]
    clf = SVC(kernel = 'rbf', C = 1000)
    clf.fit(X, Y)
    scores = cs.cross_val_score(clf, X, Y, cv=5)
    print "Accuracy: %0.2f (+- %0.2f)" % (scores.mean(),scores.std())
    return clf

def searchBestParameter():
    parameters = {'kernel':('linear','poly','rbf','sigmoid'),'C':[1,100]}
    dataset = load_data()
    row,col = dataset.shape
    X = dataset[:,:col-1]
    Y = dataset[:,-1]
    svr = SVC()
    clf = grid_search.GridSearchCV(svr,parameters)
    clf.fit(X,Y)

    print clf.best_params_

if __name__ == '__main__':
    cross_validation()
