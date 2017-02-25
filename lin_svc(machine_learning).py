import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals import joblib

data = np.loadtxt('Adata.csv', delimiter=',')
X = data[:,1:]
y = data[:,0]



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#lin_svc = svm.LinearSVC(C=1e3).fit(X_train,y_train)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(X, y)
lin_svc = svm.SVC(kernel='linear',C=1.0).fit(X,y)

print lin_svc.predict(X_test)
print y_test

score = cross_val_score(lin_svc, X, y, cv=5, scoring = 'accuracy')
print score

#joblib.dump(lin_svc,'lin_svc.pkl')