import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
from numpy.linalg import inv
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

from read_OA import read_ortho_arr
file_path = "D:/oa.20.txt"
ortho_array = read_ortho_arr(file_path)
otho_df = pd.DataFrame(ortho_array, columns=['mean_ch1', 'std_ch1', 'skewness_ch1', 
                                             'kurtosis_ch1', 'peak_ch1', 'rms_ch1', 'crest_ch1', 
                                             'shape_ch1', 'clearance_ch1', 'impulse_ch1', 
                                             'mean_ch2', 'std_ch2', 'skewness_ch2', 'kurtosis_ch2',
                                             'peak_ch2', 'rms_ch2', 'crest_ch2', 'shape_ch2',
                                             'clearance_ch2', 'impulse_ch2', 'dummy_1','dummy_2',
                                             'dummy_3'])


def get_exp_col(n):
    condition = otho_df.iloc[n,:] == 0
    col = list(otho_df.loc[n,condition].index)
    dum_list = ['dummy_1', 'dummy_2', 'dummy_3']
    for dum in dum_list:
        if (dum in col): 
            col.remove(dum) #pop함수 말고 remove함수 사용해야함.
    return(col)

# MD 공식정의함수
def get_MD_distance_exp(x,data,n):
    col = get_exp_col(n)
    normal_corr_mat = pd.DataFrame(data[col]).corr()
    dist = np.matmul(np.matmul(x, inv(normal_corr_mat)), np.transpose(x)) / len(x)
    return(dist)

# MD 적용함수
def get_MD_exp(n,df, data):
    col = get_exp_col(n)
    target_df = df[col]
    dist = np.apply_along_axis(get_MD_distance_exp, 1, target_df,data, n)
    return(dist)


def get_MD_matrix(df, data):
    arr = np.empty([24, df.shape[0]])
    for i in range(0,24):
        arr[i,:] = get_MD_exp(i,df, data)
    return(arr)

def get_normal_train_gain(otho_df,func,normal_x_train_MD_matrix, col):
    columns = col
    gain_values = []
    for i in range(0,len(columns)):
        gain_values.append(cal_gain(otho_df, func, normal_x_train_MD_matrix, columns[i]))

    return gain_values

def cal_gain(otho_df,func, mat,col_name):
    idx_1 = list(otho_df.loc[otho_df[col_name] == 1].index)
    idx_2 = list(otho_df.loc[otho_df[col_name] == 0].index)

    SN = np.array(func(mat))
    values_1 = np.mean(SN[idx_1])
    values_2 = np.mean(SN[idx_2])

    gain = values_1 - values_2
    return gain

def cal_larger_better_SNRatio(mat):
    SNRatio = []
    for i in range(0,mat.shape[0]):
        dist = mat[i]
        SN = -10 * np.log10(np.mean(1/(dist)))
        SNRatio.append(SN)

    return(SNRatio)


def get_MD_distance_1(x, data):
    normal_cov_mat = pd.DataFrame(data).corr()
    dist = np.matmul(np.matmul(x, inv(normal_cov_mat)), np.transpose(x)) / len(x)
    return(dist)

def MMTS(train):
    data = {}

    for i in range(len(np.unique(train['Class']))):
        class_ = train[train['Class'] == i].drop(columns= 'Class')
        Scaled_class = StandardScaler().fit_transform(class_)
        Scaled_class = pd.DataFrame(Scaled_class, columns = class_.columns)

        class_MD_matrix_ = get_MD_matrix(Scaled_class, Scaled_class)
        class_gain_larger = get_normal_train_gain(otho_df, cal_larger_better_SNRatio, class_MD_matrix_, Scaled_class.columns)

        data[f'{i}th_class SNR'] = class_gain_larger 
    class_gain_df = pd.DataFrame(data, index=Scaled_class.columns)
    return class_gain_df

def ScalingPerClass(df): 
    scale_dict = {}
    scaled_normal_train_list = []

    for i in range(len(np.unique(df.Class))): #8 => 클래스 개수
        dataPerClass = df[df['Class'] == i].drop(columns = 'Class')

        scale_dict[f'class_{i}'] = StandardScaler()
        scale_dict[f'class_{i}'].fit(dataPerClass)
        scaled_normal_train_list.append(scale_dict[f'class_{i}'].transform(dataPerClass))

    return scale_dict, scaled_normal_train_list

def predict(df,test,test_y):
    scale_dict, scaled_normal_train_list  =  ScalingPerClass(df)

    values = []

    for key in scale_dict.keys():
        state_num = int(key[-1])

        scaled = scale_dict[key].transform(test)
        dist = get_MD_distance_1(scaled, scaled_normal_train_list[state_num])
        values.append(dist.diagonal())

    pred = np.array(values)
    acc = np.mean(np.argmin(pred, axis=0) == test_y)
    return acc

def mts_proposed_(train, test, alpha = 1):
    GainOfData= MMTS(train)
    # 음수에만 가중치를 준다 , 
    # alpha = 1 : Gain Sum , alpha = 0 : Union
    GainofData_weight = pd.DataFrame(np.where(GainOfData>0, GainOfData, GainOfData * alpha), columns = GainOfData.columns, index= GainOfData.index)
    selected_features = GainofData_weight[GainofData_weight.sum(axis= 1) >0].index.tolist()

    # 선별된 변수만으로 성능 확인
    train_mts = train[selected_features+["Class"]]
    test_mts = test[selected_features+["Class"]]

    result_ = predict(train_mts, test_mts.drop(columns='Class'), test_mts['Class'])
    return train_mts, test_mts, selected_features, result_
    
    

def mts_proposed_(train, test, alpha = 1):
    GainOfData= MMTS(train)
    # 음수에만 가중치를 준다 , 
    # alpha = 1 : Gain Sum , alpha = 0 : Union
    GainofData_weight = pd.DataFrame(np.where(GainOfData>0, GainOfData, GainOfData * alpha), columns = GainOfData.columns, index= GainOfData.index)
    selected_features = GainofData_weight[GainofData_weight.sum(axis= 1) >0].index.tolist()
    
    # 선별된 변수만으로 성능 확인
    train_mts = train[selected_features+["Class"]]
    test_mts = test[selected_features+["Class"]]
    
    result_ = predict(train_mts, test_mts.drop(columns='Class'), test_mts['Class'])
    return train_mts, test_mts, selected_features, result_

def best_alpha(train, val):
    result_ = pd.DataFrame(columns=['alpha','acc','features','num'])

    for alpha in  tqdm(np.arange(0, 2.1, 0.1)):
        train_, test_ ,selected_features, result = mts_proposed_(train, val, alpha= alpha)
        result_ = result_.append({'alpha':alpha, 'acc':result, 'features':selected_features, 'num':len(selected_features)}, ignore_index=True)
        
    return result_

def testdata_acc_by_alpha(train, test, result_):
    #result_['testdata_acc'] = 0
    testdata_acc = []
    for feat in result_['features']:
        train_mts = train[feat+["Class"]]
        test_mts = test[feat+["Class"]]

        testdata_acc.append(predict(train_mts, test_mts.drop(columns='Class'), test_mts['Class']))

    result_['testdata_acc'] = testdata_acc
    return result_


