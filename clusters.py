import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from tensorflow import keras

from data import get_encoded_data


def fashion_scatter(x, colors):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    # on choisit une palette de de couleur avec seaborn
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # on crée le scatter-plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # on ajoute les labels pour chaque digit correspondant au label
    txts = []

    for i in range(num_classes):
        # position de chaque label au milieu des points

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def kmeans_fit():
    _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    (x_train_encoded, y_train), (x_test_encoded, y_train) = get_encoded_data()
    k_means = KMeans(n_clusters=10, random_state=0)
    k_means.fit(x_train_encoded)
    test_clusters = k_means.predict(x_test_encoded)

    labels = np.zeros_like(test_clusters)
    for i in range(10):
        mask = (test_clusters == i)
        labels[mask] = mode(y_test[mask])[0]
        print("nombre de : ", i, " ", len(test_clusters[test_clusters == i]))

    cent = k_means.cluster_centers_
    score = accuracy_score(labels, test_clusters)

    print("Score k-means: ", score)


def tsne():
    import time
    from sklearn.manifold import TSNE
    (x_train_encoded, y_train), (x_test_encoded, y_test) = get_encoded_data()
    time_start = time.time()
    for per in [15, 25, 35, 45]:
        tsne = TSNE(n_components=2, verbose=1, perplexity=per, n_iter=500)
        tsne_results = tsne.fit_transform(x_train_encoded)
        figure, _, _, _ = fashion_scatter(tsne_results, y_train)
        figure.show()
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
