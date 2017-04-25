import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import colors

def generate_data_set(n, d=1):
    random.seed()
    s = list()
    for i in range(n):
        s.append([random.randint(1, 10) for _ in range(d)])
    return s

def save_data(fname):
    with open(fname, 'w') as f:
        for l in generate_data_set(10, 2):
            f.write(','.join([str(i) for i in l]) + '\n')

def sim():
    #save_data('test.data')
    A = np.loadtxt('wine.data', delimiter=',')
    y = A[:, 0]
    # Remove targets from input data
    A = A[:, 1:]
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(A, y)
    drA = lda.transform(A)

    Z = generate_data_set(1, 13)
    #Z = lda.transform(Z)
    z_lab = lda.predict(Z)
    z_prob = lda.predict_proba(Z)

    plt.figure()
    x = [l[0] for l in drA]
    y = [l[1] for l in drA]
    cls = [int(lda.predict([x1, y1])[0]) for x1, y1 in zip(x, y)]
    plt.scatter(x, y, c=[[1, 0, 0]])
    plt.savefig('a.png')