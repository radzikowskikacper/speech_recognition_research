import random, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import colors
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

def generate_data_set(n, d=1):
    random.seed()
    s = list()
    for i in range(n):
        s.append([random.randint(1, 100) for _ in range(d)])
    return s

def save_data(fname):
    with open(fname, 'w') as f:
        for l in generate_data_set(10, 2):
            f.write(','.join([str(i) for i in l]) + '\n')

def sim():
    #save_data('test.data')
    A = np.loadtxt('test1.data', delimiter=',')

    y = A[:, 0]

    # Remove targets from input data
    A = A[:, 1:]

    for i in [0, 1, 2, 4]:
        for j in range(len(A)):
            A[j][i] = random.randint(0, 100)

    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sff = sel.fit_transform(A)

    clf = RandomForestClassifier()
    clf = clf.fit(A, y)
    hh = clf.feature_importances_
    #jj = clf.predict([[1, 2, 3, 25, 50]])

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(A)
    hh2 = X_new.shape

    #plot(A, y)

    lda = LinearDiscriminantAnalysis(n_components=2)
    hh3 = lda.fit(A, y)
    drA = lda.transform(A)

    Z = generate_data_set(1, 5)
    Z = lda.transform(Z)
    z_lab = lda.predict(Z)
    z_prob = lda.predict_proba(Z)

    plt.figure()
    x = [l[0] for l in drA]
    y = [l[1] for l in drA]
    cls = [int(lda.predict([x1, y1])[0]) for x1, y1 in zip(x, y)]
    plt.scatter(x, y, c=[[1, 0, 0]])
    plt.savefig('a.png')

def plot(X, y):
    feature_dict = {i: label for i, label in zip(
        range(5),
        ('1', '2', '3', '4', '5'))}

    import pandas as pd
    '''
    df = pd.io.parsers.read_csv(
        filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None,
        sep=',',
    )
    df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
    df.dropna(how="all", inplace=True)  # to drop the empty line at file-end

    df.tail()
    '''

    from sklearn.preprocessing import LabelEncoder

    #X = df[[0, 1, 2, 3]].values
    #y = df['class label'].values

    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1

    label_dict = {1: '1', 2: '2', 3: '3'}

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 6))

    for ax, cnt in zip(axes.ravel(), range(5)):

        # set bin sizes
        min_b = math.floor(np.min(X[:, cnt]))
        max_b = math.ceil(np.max(X[:, cnt]))
        bins = np.linspace(min_b, max_b, 25)

        # plottling the histograms
        for lab, col in zip(range(1, 4), ('blue', 'red', 'green')):
            ax.hist(X[y == lab, cnt],
                    color=col,
                    label='class %s' % label_dict[lab],
                    bins=bins,
                    alpha=0.5, )
        ylims = ax.get_ylim()

        # plot annotation
        leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
        leg.get_frame().set_alpha(0.5)
        ax.set_ylim([0, max(ylims) + 2])
        ax.set_xlabel(feature_dict[cnt])
        ax.set_title('Iris histogram #%s' % str(cnt + 1))

        # hide axis ticks
        ax.tick_params(axis="both", which="both", bottom="off", top="off",
                       labelbottom="on", left="off", right="off", labelleft="on")

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[0][0].set_ylabel('count')
    axes[1][0].set_ylabel('count')

    fig.tight_layout()

    plt.show()