import numpy as np
import pandas as pd
import seaborn as sns
import glob
import os

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.spatial import distance
from numpy.linalg import inv

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# 한글 폰트
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

from mmts import *
from check_accuracy import * 
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


def MMTS_TKDE(train, test):
    
    Normal = train[train['Class'] == 0].drop(columns= 'Class')
    fault1 = train[train['Class'] == 1].drop(columns= 'Class')
    fault2 = train[train['Class'] == 2].drop(columns= 'Class')
    fault3 = train[train['Class'] == 3].drop(columns= 'Class')
    fault4 = train[train['Class'] == 4].drop(columns= 'Class')
    fault5 = train[train['Class'] == 5].drop(columns= 'Class')
    fault6 = train[train['Class'] == 6].drop(columns= 'Class')

    scale0 =  StandardScaler().fit(Normal)
    scale1 =  StandardScaler().fit(fault1)
    scale2 =  StandardScaler().fit(fault2)
    scale3 =  StandardScaler().fit(fault3)
    scale4 =  StandardScaler().fit(fault4)
    scale5 =  StandardScaler().fit(fault5)
    scale6 =  StandardScaler().fit(fault6)
    
    SNR_SUM = []

    for state in [Normal, fault1,fault2, fault3,fault4,fault5,fault6]:
        Scaled_state_by_Normal = pd.DataFrame(scale0.transform(state), columns = state.columns)
        Scaled_state_by_fault1 = pd.DataFrame(scale1.transform(state), columns = state.columns) 
        Scaled_state_by_fault2 = pd.DataFrame(scale2.transform(state), columns = state.columns)
        Scaled_state_by_fault3 = pd.DataFrame(scale3.transform(state), columns = state.columns)
        Scaled_state_by_fault4 = pd.DataFrame(scale4.transform(state), columns = state.columns)
        Scaled_state_by_fault5 = pd.DataFrame(scale5.transform(state), columns = state.columns)
        Scaled_state_by_fault6 = pd.DataFrame(scale6.transform(state), columns = state.columns)

        state_MD_by_Noraml = get_MD_matrix(Scaled_state_by_Normal, Normal)
        state_MD_by_fault1 = get_MD_matrix(Scaled_state_by_fault1, fault1)
        state_MD_by_fault2 = get_MD_matrix(Scaled_state_by_fault2, fault2)
        state_MD_by_fault3 = get_MD_matrix(Scaled_state_by_fault3, fault3)
        state_MD_by_fault4 = get_MD_matrix(Scaled_state_by_fault4, fault4)
        state_MD_by_fault5 = get_MD_matrix(Scaled_state_by_fault5, fault5)
        state_MD_by_fault6 = get_MD_matrix(Scaled_state_by_fault6, fault6)

        SNRatio = []

        for experiment in range(0, 24): # row : 실험 순번, 실험에 따라 변수 조합이 다름. 
            dist =    state_MD_by_Noraml[experiment]
            dist_1 =  state_MD_by_fault1[experiment]
            dist_2 =  state_MD_by_fault2[experiment]
            dist_3 =  state_MD_by_fault3[experiment]
            dist_4 =  state_MD_by_fault4[experiment]
            dist_5 =  state_MD_by_fault5[experiment]
            dist_6 =  state_MD_by_fault6[experiment]

            if state is Normal:
                SN = (-10* np.log10(np.mean(dist/dist_1)) + (-10* np.log10(np.mean(dist/dist_2))) + (-10* np.log10(np.mean(dist/dist_3))) + (-10* np.log10(np.mean(dist/dist_4))) + (-10* np.log10(np.mean(dist/dist_5))) + (-10* np.log10(np.mean(dist/dist_6))))
            elif state is fault1:
                SN = (-10* np.log10(np.mean(dist_1/dist)) + (-10* np.log10(np.mean(dist_1/dist_2))) + (-10* np.log10(np.mean(dist_1/dist_3))) + (-10* np.log10(np.mean(dist_1/dist_4))) + (-10* np.log10(np.mean(dist_1/dist_5))) + (-10* np.log10(np.mean(dist_1/dist_6))))
            elif state is fault2:
                SN = (-10* np.log10(np.mean(dist_2/dist_1)) + (-10* np.log10(np.mean(dist_2/dist))) + (-10* np.log10(np.mean(dist_2/dist_3))) + (-10* np.log10(np.mean(dist_2/dist_4))) + (-10* np.log10(np.mean(dist_2/dist_5))) + (-10* np.log10(np.mean(dist_2/dist_6))))
            elif state is fault3:
                SN = (-10* np.log10(np.mean(dist_3/dist_1)) + (-10* np.log10(np.mean(dist_3/dist_2))) + (-10* np.log10(np.mean(dist_3/dist))) + (-10* np.log10(np.mean(dist_3/dist_4))) + (-10* np.log10(np.mean(dist_3/dist_5))) + (-10* np.log10(np.mean(dist_3/dist_6))))
            elif state is fault4:
                SN = (-10* np.log10(np.mean(dist_4/dist_1)) + (-10* np.log10(np.mean(dist_4/dist_2))) + (-10* np.log10(np.mean(dist_4/dist_3))) + (-10* np.log10(np.mean(dist_4/dist))) + (-10* np.log10(np.mean(dist_4/dist_5))) + (-10* np.log10(np.mean(dist_4/dist_6))))
            elif state is fault5:
                SN = (-10* np.log10(np.mean(dist_5/dist_1)) + (-10* np.log10(np.mean(dist_5/dist_2))) + (-10* np.log10(np.mean(dist_5/dist_3))) + (-10* np.log10(np.mean(dist_5/dist_4))) + (-10* np.log10(np.mean(dist_5/dist))) + (-10* np.log10(np.mean(dist_5/dist_6))))
            elif state is fault6:
                SN = (-10* np.log10(np.mean(dist_6/dist_1)) + (-10* np.log10(np.mean(dist_6/dist_2))) + (-10* np.log10(np.mean(dist_6/dist_3))) + (-10* np.log10(np.mean(dist_6/dist_4))) + (-10* np.log10(np.mean(dist_6/dist_5))) + (-10* np.log10(np.mean(dist_6/dist))))

            SNRatio.append(SN)
        SNR_SUM.append(SNRatio)
        
        
    sum_ = np.array(SNR_SUM[0])+np.array(SNR_SUM[1])+np.array(SNR_SUM[2])+np.array(SNR_SUM[3])+np.array(SNR_SUM[4])+np.array(SNR_SUM[5])+np.array(SNR_SUM[6])
    gain_data = []

    for col_name in train.columns.drop('Class'):
        idx_1 = list(otho_df.loc[otho_df[col_name] == 1].index)
        idx_2 = list(otho_df.loc[otho_df[col_name] == 0].index)

        value_1 = np.mean(np.array(sum_)[idx_1])
        value_2 = np.mean(np.array(sum_)[idx_2])
        gain = value_1 - value_2
        gain_data.append(gain)
        
    c_ = pd.DataFrame(gain_data, index = Normal.columns, columns=['value'])
    c_.sort_values(by='value', ascending=False)
    feature_s = c_[c_['value']>0].index.tolist()
    print(feature_s)

    scale_dict_fs = return_scale_dict(train,feature_s)
    training_acc(train, scale_dict_fs, feature_s)
    test_acc(train, test, scale_dict_fs, feature_s)
    return feature_s
