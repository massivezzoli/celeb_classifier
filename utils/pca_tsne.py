import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()


def make_labels(y_old):
    #y_new = lab_enc.fit_transform(y_old)
    y_new = np.argmax(y_old, 1)
    return y_new


def pca_dim_red(x, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(x)
    x_transf = pca.transform(x)
    exp_var = pca.explained_variance_ratio_
    print('Explained variance ratio:')
    print(exp_var)
    cum_exp_var = np.cumsum(exp_var)
    print('Cumulative explained variance from PCA:')
    print(cum_exp_var)
    return x_transf


def tsne_output(pre_act, labels, perp, iters, filename='tsne.png'):
# Perform t-sne on data from pre-softmax output layer,
# saves picture of 2d representation with color coded labels

    model = TSNE(n_components=2, perplexity=perp,
                 init='pca', method='exact', n_iter=iters, random_state=0)
    #x_tsne = model.fit_transform(pre_act[-1])
    x_tsne = model.fit_transform(pre_act)

# from one-hot labels to categorical
    new_l = make_labels(labels)
    assert (pre_act[0].shape[0]) >= len(new_l), "More labels than samples"
    n_classes = len(set(new_l))

# Save figure:
    plt.figure(figsize=(6,6))
    plt.scatter(x_tsne[:,0], x_tsne[:,1], c=new_l,
                cmap=plt.cm.get_cmap('Vega10', n_classes))
    plt.colorbar(ticks=range(n_classes))
    plt.clim(-0.5, n_classes - 0.5)
    plt.savefig(filename)

    return x_tsne, new_l

def tsne_output2(pre_act, labels, perp, iters, train_size, filename='tsne.png'):
# Perform t-sne on data from pre-softmax output layer,
# saves picture of 2d representation with color coded labels

    model = TSNE(n_components=2, perplexity=perp,
                 init='pca', method='exact', n_iter=iters, random_state=0)
    #x_tsne = model.fit_transform(pre_act[-1])
    x_tsne = model.fit_transform(pre_act)

# from one-hot labels to categorical
    new_l = make_labels(labels)
    assert (pre_act.shape[0]) >= len(new_l), "More labels than samples"
    n_classes = len(set(new_l))

# Save figure:
    plt.figure(figsize=(6,6))
    plt.scatter(x_tsne[:train_size,0], x_tsne[:train_size,1], marker='o', alpha=0.5, c=new_l[:train_size],
                cmap=plt.cm.get_cmap('Vega20', n_classes))
    plt.scatter(x_tsne[train_size:,0], x_tsne[train_size:,1], marker='+', c=new_l[train_size:],
                cmap=plt.cm.get_cmap('Vega20', n_classes))
    plt.colorbar(ticks=range(n_classes))
    plt.clim(-0.5, n_classes - 0.5)
    plt.savefig(filename)

    return x_tsne, new_l
