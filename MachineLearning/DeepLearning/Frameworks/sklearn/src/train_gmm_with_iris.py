from sklearn.mixture import GMM
from sklearn import datasets
from itertools import cycle, combinations
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def make_ellipses(gmm, ax, x, y):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


iris = datasets.load_iris()

gmm = GMM(n_components=3, covariance_type='full', n_iter=100)
gmm.fit(iris.data)

predictions = gmm.predict(iris.data)

colors = cycle('rgb')
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
targets = range(len(labels))

feature_index = range(len(iris.feature_names))
feature_names = iris.feature_names
combs = combinations(feature_index, 2)

f, axarr = plt.subplots(3, 2)
axarr_flat = axarr.flatten()

for comb, axflat in zip(combs, axarr_flat):
    x = comb[0]
    y = comb[1]
    for target, color, label in zip(targets, colors, labels):
        axflat.scatter(iris.data[predictions == target, x], iris.data[predictions == target, y], c=color, label=label)
        axflat.set_xlabel(feature_names[x])
        axflat.set_ylabel(feature_names[y])
        axflat.legend(loc='upper right')
    make_ellipses(gmm, axflat, x, y)

plt.tight_layout()
plt.show()
