from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numpy.linalg import inv

def get_MD_distance_1(x, data):
    normal_cov_mat = pd.DataFrame(data).corr()
    dist = np.matmul(np.matmul(x, inv(normal_cov_mat)), np.transpose(x)) / len(x)
    return(dist)

def pca_graph(train, TKDE_feats, proposed_feats):
    ## original data(20 dims)
    Origin = train.copy()
    TKDE = train[eval(TKDE_feats) + ['Class']]     
    Proposed = train[eval(proposed_feats) + ['Class']]
    
    
    ## all features 
    std_ = StandardScaler().fit(Origin.drop(columns='Class'))
    std_result = std_.transform(Origin.drop(columns = 'Class')) 
    pca_ = PCA(n_components=2).fit(std_result)
    pca_result = pca_.transform(std_result)
    
    ## with feathers using TKDE
    std_TKDE = StandardScaler().fit(TKDE.drop(columns='Class'))
    std_result_TKDE = std_TKDE.transform(TKDE.drop(columns = 'Class')) 
    pca_TKDE = PCA(n_components=2).fit(std_result_TKDE)
    pca_result_TKDE = pca_TKDE.transform(std_result_TKDE)
    
    ## with features using proposed
    std_Proposed = StandardScaler().fit(Proposed.drop(columns='Class'))
    std_result_Proposed = std_Proposed.transform(Proposed.drop(columns = 'Class')) 
    pca_Proposed = PCA(n_components=2).fit(std_result_Proposed)
    pca_result_Proposed = pca_Proposed.transform(std_result_Proposed)
    
    legend_labels = {0: 'Normal', 1: 'Gear-T', 2: 'Gear-R', 3: 'Gear-L',\
                     4: 'Bolt', 5: 'Oil', 6: 'Shaft'}
    legends = [legend_labels[label] for label in Origin.Class]
    legend_order = ['Normal','Gear-T','Gear-R','Gear-L','Bolt','Oil','Shaft']
    
    # x축 최소점
    # x축 최대점
    x_min = np.min([np.min(pca_result[:,0]), np.min(pca_result_TKDE[:,0]), np.min(pca_result_Proposed[:,0])])
    x_max = np.max([np.max(pca_result[:,0]), np.max(pca_result_TKDE[:,0]), np.max(pca_result_Proposed[:,0])])
    
    y_min = np.min([np.min(pca_result[:,1]), np.min(pca_result_TKDE[:,1]), np.min(pca_result_Proposed[:,1])]) 
    y_max = np.max([np.max(pca_result[:,1]), np.max(pca_result_TKDE[:,1]), np.max(pca_result_Proposed[:,1])])
        
    # pca graph
    fig, axes = plt.subplots(1,3, figsize=(13,4))
    palette = sns.color_palette("bright")
    
    sns.scatterplot(data= pca_result, x=pca_result[:,0], y =pca_result[:,1], hue=legends, ax=axes[0],s=3, palette =palette,legend=False, hue_order=legend_order) #1280
    sns.scatterplot(data= pca_result_TKDE, x=pca_result_TKDE[:,0], y =pca_result_TKDE[:,1], hue=legends, ax=axes[1],s=3, palette =palette,legend=False, hue_order=legend_order) #1280
    sns.scatterplot(data= pca_result_Proposed, x=pca_result_Proposed[:,0], y =pca_result_Proposed[:,1], hue=legends, ax=axes[2],s=3, palette =palette,legend=True,hue_order=legend_order) #1280
    
    
    min_ = np.min([x_min, y_min])
    max_ = np.max([x_max, y_max])

    x_limits = (min_, max_)
    y_limits = (min_, max_)

        
    for ax in axes:
        ax.set_xlim(x_limits)
        ax.set_xticks(np.arange(math.floor(min_)-1, math.ceil(max_)+1, 2))
        
    for ax in axes:
        ax.set_ylim(y_min-2, y_max+2)
        ax.set_yticks(np.arange(math.floor(min_)-1, math.ceil(max_)+1, 2))
        
    axes[0].set_title('MMDC', fontsize=14, fontweight='bold')
    axes[1].set_title('MMTS_s', fontsize=14, fontweight='bold')
    axes[2].set_title('MMTS_w', fontsize=14, fontweight='bold')
    
    
    
    plt.show()

    return pca_result, pca_result_TKDE, pca_result_Proposed


def contour(train, pca_result):
    
    normal_ = train[train.Class == 0].drop(columns = 'Class')
    X = train.drop(columns= 'Class')
    
    scaler = StandardScaler()
    scaler.fit(normal_)
    train_ = scaler.transform(X)
    
        
    legend_labels = {0: 'Normal', 1: 'Gear-T', 2: 'Gear-R', 3: 'Gear-L',\
                     4: 'Bolt', 5: 'Oil', 6: 'Shaft'}
    legends = [legend_labels[label] for label in train.Class]
    legend_order = ['Normal','Gear-T','Gear-R','Gear-L','Bolt','Oil','Shaft']
    
    x = pca_result[:,0]
    y = pca_result[:,1]
    z = get_MD_distance_1(train_, normal_).diagonal()
    log_z = np.log10(1 + z)
    
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    palette = sns.color_palette("bright")
    
    contour = axes[1].tricontourf(x[:], y[:], z[:], cmap='Spectral', alpha=0.3, levels = 10)
    contour_log = axes[2].tricontourf(x[:], y[:], log_z[:], cmap='Spectral', alpha=0.3, levels = 10)
    
    sns.scatterplot(data= pca_result, x=pca_result[:,0], y =pca_result[:,1], hue=legends,s=3, ax=axes[0], palette = palette,legend=False, hue_order=legend_order) #1280
    sns.scatterplot(data= pca_result, x=pca_result[:,0], y =pca_result[:,1], hue=legends,s=3, ax=axes[1], palette = palette, legend=False, hue_order=legend_order) #1280
    sns.scatterplot(data= pca_result, x=pca_result[:,0], y =pca_result[:,1], hue=legends,s=3, ax=axes[2], palette = palette, hue_order=legend_order) #1280
    
   
    
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PCA Plot')
    
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('PCA Plot using MD-based Contour')
    
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].set_title('PCA Plot using log(MD+1)-based Contour')
    

#     for ax in axes:
#         #ax.set_xticks(np.arange(x.min(), x.max(), 2))
#         #ax.set_xticks(np.arange(int(x.min()), int(x.max()), 1))
#         ax.set_xticks(np.arange(-2,8, 1))
#         ax.set_yticks(np.arange(-2,8, 1))
    
    fig.colorbar(contour, ax= axes[1])
    fig.colorbar(contour_log, ax= axes[2])
                
    plt.show()
    return fig, axes