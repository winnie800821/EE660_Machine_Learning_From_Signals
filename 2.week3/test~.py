from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
np.set_printoptions(threshold=np.inf)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import math
import collections
from collections import Counter
import random
from sklearn.metrics import accuracy_score

Xtrain= np.loadtxt(open("Xtrain.csv"), delimiter=",")
Ytrain=np.loadtxt(open("ytrain.csv"), delimiter=",")

sh=Xtrain.shape

print('sh=',sh)
print(sh[0])
print(sh[1])