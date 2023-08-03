import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


def cluster_performance(train, TKDE_feats, Proposed_feats):

    pca_ori, pca_TDKE, pca_proposed =pca_graph(train, TKDE_feats, Proposed_feats)

    sil_score_ori = silhouette_score(pca_ori, train.Class, metric='euclidean')
    sil_score_TKDE = silhouette_score(pca_TDKE, train.Class, metric='euclidean')
    sil_score_proposed = silhouette_score(pca_proposed, train.Class, metric='euclidean')

    
    print('Original sil_score: {:.4f}'.format(sil_score_ori))
    print('TKDE sil_score: {:.4f}'.format(sil_score_TKDE))
    print('Proposed sil_score: {:.4f}'.format(sil_score_proposed))
    
    print('Original Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(pca_ori, train.Class)))
    print('TKDE Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(pca_TDKE, train.Class)))
    print('Proposed Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(pca_proposed, train.Class)))