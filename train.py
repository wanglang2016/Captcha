
from sklearn.svm import SVC
from sklearn import grid_search
import numpy as np
from sklearn import cross_validation as cs
from sklearn.externals import joblib
import warnings
import time
from PIL import Image
import os
from PIL import ImageFilter
from PIL import ImageEnhance

def getBinaryPix(im):
    im = Image.open(im)
    img = np.array(im)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if(img[i,j] <= 128):
                img[i,j] = 0
            else:
                img[i,j] = 1;
    binpix = np.ravel(img)
    return binpix

def segment(im):
    s = 2
    w = 14
    h = 20
    t = 0
    im_new = []
    for i in range(4):
        im1 = im.crop((s+w*i,t,s+w*(i+1), h))
        im_new.append(im1)
    return im_new

def imgTransfer(f_name):
    im = Image.open(f_name)
    im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(1)
    im = im.convert('L')
    return im

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

def train():
    dataset = load_data()
    row, col = dataset.shape
    X = dataset[:,:col-1]
    Y = dataset[:,-1]
    clf = SVC(kernel = 'rbf', C=1000)
    clf.fit(X,Y)
    joblib.dump(clf, '/Users/wanglang/Desktop/temp.pkl')

def cutPictures2(name):
    im = imgTransfer(name)
    pics = segment(im)
    for pic in pics:
        pic.save('/Users/wanglang/Desktop/test/%s.jpeg'%(int(time.time()*1000000)),'jpeg')

# def loadPredict(name):
#     cutPicture2(name)
#     dirs = '/Users/wanglang/Desktop/test/'
#     fs = os.listdir(dirs)
#     clf = cross_validation()
#     predictValue = [[]]
#
#     for fname in fs:
#         fn = dirs + fname
#         binpix = getBinaryPix(fn)
#         predictValue.append(clf.predict(binpix))
#
#     predictValue = [str(int(i)) for i in predictValue]
#     print 'the captcha is : %s' % (''.join(predictValue))
def loadPredict(name):
#
    cutPictures2(name)

    dirs = u'/Users/wanglang/Desktop/test/'
    fs = os.listdir(dirs)
    clf = cross_validation()
    predictValue = []

    for fname in fs:
        fn = dirs + fname
        binpix = getBinaryPix(fn)
        predictValue.append(clf.predict(binpix))

    predictValue = [str(int(i)) for i in predictValue]
    print "the picture number is :" ,"".join(predictValue)
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
    # cross_validation()
    searchBestParameter()
    train()
    loadPredict('/Users/wanglang/Desktop/1492068309780614.jpg')
