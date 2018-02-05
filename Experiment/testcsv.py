from sklearn import svm
import numpy as np

f = open('../data_reader/data/1000data/spambase.csv')
f.readline()
data = np.loadtxt(fname=f, delimiter=",")

train_ = data[:3000]
X = data[:,1:]
y = data[:,0]

learning_model = svm.SVC(probability=True, kernel='linear')
learning_model.fit(X,y)
print(learning_model.coef_)