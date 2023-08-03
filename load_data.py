import pandas as pd

def generate_dateset(folder_path ='../LS실험데이터/정방향1_2_downsampling_1280/', exp_condition ='M3', exclude = None):
    
    train_ratio,val_ratio,test_ratio = 0.5, 0.25,0.25
    train_list, val_list, test_list = [], [], []
    mapping = {'normal': 0, 'gear_face_crack': 1, 'gear_root_crack': 2, 'gear_broken':3, 'loosened_bolt':4, 'no_oil':5, 'shaft_misalignment':6}

    for state in mapping.keys():
        _ = pd.read_csv(folder_path + f'{state}'+ '_'+f'{exp_condition}'+'.csv')
        _ = _.drop(columns = ['Gear','Speed'], inplace=False)
        _ = _.loc[:,_.columns.str.contains('ch1|ch2|Class')]
        _['Class'] = _['Class'].replace(mapping)
        
        size = _.shape[0]
        
        train_list.append(_[:int(size*train_ratio)])
        val_list.append(_[int(size*train_ratio):int(size*train_ratio)+int(size*val_ratio)])
        test_list.append(_[int(size*train_ratio)+int(size*val_ratio):])
        
    train_ = pd.concat(train_list, ignore_index=True)
    val_ = pd.concat(val_list, ignore_index=True)
    test_ = pd.concat(test_list, ignore_index=True)
    return train_, val_, test_     


