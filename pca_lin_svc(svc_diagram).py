import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import svm, datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score


# import some data to play with
"""
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

print X
print y
"""


data = np.loadtxt('Adata.csv', delimiter=',')
X = data[:,1:]
y = data[:,0]
"""

data = pandas.read_csv("Adata.csv", index_col=(0,1))

ylabels = [a for a, _ in data.index]
labels = [text for _, text in data.index]
encoder = preprocessing.LabelEncoder().fit(ylabels)

X = data.as_matrix(data.columns)
y = encoder.transform(ylabels)
target_names = encoder.classes_
"""
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
X = X_train
y = y_train


h = .02  # step size in the mesh
C = 1.0  # SVM regularization parameter

#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

"""
print lin_svc.score(X,y)
print lin_svc.score(X_test, y_test)

print (lin_svc.predict(X))
print y

print lin_svc.predict(X_test)
print y_test

score = cross_val_score(lin_svc, X, y, cv=5, scoring = 'accuracy')
print score

print y
"""

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


#for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    #plt.subplot(2, 2, i + 1)
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)

Z = lin_svc.predict(np.c_[xx.ravel(), yy.ravel()])


    # Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)

    # Plot also the training points

#for X_transformed, title in [(X,"PCA")]:
#  for color, i. name in zip(colors, [0,1,2,3,4,5], names):
plt.scatter(X[:,0], X[:,1],c=y ,cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles[1])
plt.show()

