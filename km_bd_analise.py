import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# bibliotecas nescessarias para executar o algoritimo em requirements.txt

def main():
    X = pd.read_excel('BD_km.xlsx')
    X = np.array(X.drop('INDIVÍDUO', axis=1))

    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=5, random_state=0)
    pred_y = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()


if __name__ == '__main__':
    main()
