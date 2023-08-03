import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler


def get_MD_distance(x, data):
    normal_cov_mat = pd.DataFrame(data).corr()
    dist = np.matmul(np.matmul(x, inv(normal_cov_mat)), np.transpose(x)) / len(x)
    return(dist)


def return_scale_dict(df, feature_selection=None):
    scale_dict={}
    if feature_selection != None:
        df = df[feature_selection+["Class"]]
    
    for i in df.Class.unique():
        df_ = df[df['Class'] == i].drop(columns= 'Class')
        scale_dict[f'class_{i}'] = StandardScaler()
        scale_dict[f'class_{i}'].fit(df_)
        
    return scale_dict

def training_acc(df, scale_dict, feature_selection = None):
    acc_dict = {}
    if feature_selection != None:
        df = df[feature_selection+["Class"]]
        
    dataPerclass_list = [df[df['Class'] == clss].drop(columns= 'Class') for clss in df.Class.unique()]

    for i in df.Class.unique():
        results = []
        class_ = df[df['Class'] == i].drop(columns= 'Class')
        A = [scale_dict[key].transform(class_) for key in scale_dict]
        for j in range(7):
            results.append(get_MD_distance(A[j], dataPerclass_list[j]).diagonal())
        pred = np.array(results)
        acc_ = np.mean(df.Class.unique()[np.argmin(pred, axis=0)] == df[df['Class']==i].Class)
        acc_dict[i] = acc_
        
    # 결과 Print
    for _ in sorted(acc_dict.keys()):
        print(f'{_}_class 정확도 : {acc_dict[_]}')
        
    print(f'Training 평균 정확도 {np.mean(np.mean(list(acc_dict.values())))}')
    
    
def test_acc(df, df_test, scale_dict, feature_selection = None):
    
    if feature_selection != None :
        df, df_test = df[feature_selection+["Class"]], df_test[feature_selection+["Class"]]
        
    B = [df[df['Class'] == clss].drop(columns= 'Class') for clss in np.unique(df.Class)]
    results = []

    test_ = df_test.drop(columns='Class')
    A = [scale_dict[key].transform(test_) for key in scale_dict]
    for j in range(7):
        results.append(get_MD_distance(A[j], B[j]).diagonal())
    pred = np.array(results)
    acc = np.mean(df_test.Class.unique()[np.argmin(pred, axis=0)] == df_test.Class)

    print(f'Test 정확도: {acc}')