import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import datasets
from itertools import cycle


iris = datasets.load_iris()
X = iris.data
y = iris.target

targets = range(len(iris.target_names))
colors = cycle('rgb')

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

for target, color in zip(targets, colors):
    plt.scatter(X_r[y == target, 0], X_r[y == target, 1], c=color, label=iris.target_names[target])

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()
