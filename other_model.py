import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score


def lda_predict(train, test):
    lda = LinearDiscriminantAnalysis()
    
    _ = train.drop(columns = 'Class')
    scaler = StandardScaler()
    scaler.fit(_)
    train_ = scaler.transform(_)
    test_ = scaler.transform(test.drop(columns= 'Class'))
    test_ = pd.DataFrame(test_, columns = _.columns)
    
    lda.fit(train_, train.Class)
    print(f'test acc : {np.mean(lda.predict(test_) == test.Class)}')


def svc_predict(train, test):
    svc = svm.SVC(kernel='linear')
    
    _ = train.drop(columns = 'Class')
    scaler = StandardScaler()
    scaler.fit(_)
    train_ = scaler.transform(_)
    test_ = scaler.transform(test.drop(columns= 'Class'))
    test_ = pd.DataFrame(test_, columns = _.columns)
    
    svc.fit(train_, train.Class)
    print(f'test acc : {np.mean(svc.predict(test_) == test.Class)}')

def logistic_predict(train, test):
    lr = LogisticRegression()
    
    _ = train.drop(columns = 'Class')
    scaler = StandardScaler()
    scaler.fit(_)
    train_ = scaler.transform(_)
    test_ = scaler.transform(test.drop(columns= 'Class'))
    test_ = pd.DataFrame(test_, columns = _.columns)
    
    lr.fit(train_, train.Class)
    print(f'test acc : {np.mean(lr.predict(test_) == test.Class)}')


def decisionTree_predict(train, test):
    dc = tree.DecisionTreeClassifier(max_depth=5)
    
    # standardization for train
    _ = train.drop(columns = 'Class')
    scaler = StandardScaler()
    scaler.fit(_)
    train_ = scaler.transform(_)
    # for test
    test_ = scaler.transform(test.drop(columns= 'Class'))
    test_ = pd.DataFrame(test_, columns = _.columns)
    
    # model fit
    dc.fit(train_, train.Class)
    print(f'test acc : {np.mean(dc.predict(test_) == test.Class)}')


def lgbm_predict(train, test):
    
    lgb_model = lgb.LGBMClassifier(objective='multiclass', metric='multi_logloss', random_state=42)
    
    param_grid = {
    'num_leaves': [10, 20,30,40],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [25, 50, 100, 200]}
    
    grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=3)
    grid_search.fit(train.drop(columns = 'Class'), train.Class)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    pred = grid_search.best_estimator_.predict(test.drop(columns='Class'))
    print(f'test accuracy : {accuracy_score(test.Class, pred)}')

    


def xgboost_predict(train, test):
    
    xgb_model = xgb.XGBClassifier(objective='multi:softmax')
    param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50 ,100, 200, 300]}
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3)
    grid_search.fit(train.drop(columns = 'Class'), train.Class)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    pred = grid_search.best_estimator_.predict(test.drop(columns='Class'))
    print(f'test accuracy : {accuracy_score(test.Class, pred)}')
    
    
def other_model(train3, train4, train5, test3, test4, test5):
    print('lda')
    lda_predict(train3,test3)
    lda_predict(train4,test4)
    lda_predict(train5,test5)
    #(train6_5120, test6_5120)
    print('svc')
    svc_predict(train3,test3)
    svc_predict(train4,test4)
    svc_predict(train5,test5)
    #svc_predict(train6_5120, test6_5120)
    print('logistic')
    logistic_predict(train3,test3)
    logistic_predict(train4,test4)
    logistic_predict(train5,test5)
    #logistic_predict(train6_5120, test6_5120)
    print('dc')
    decisionTree_predict(train3,test3)
    decisionTree_predict(train4,test4)
    decisionTree_predict(train5,test5)
    #decisionTree_predict(train6_5120, test6_5120)
    print('lgbm')
    lgbm_predict(train3,test3)
    lgbm_predict(train4,test4)
    lgbm_predict(train5,test5)
    #lgbm_predict(train6_5120, test6_5120)
    print('xg')
    xgboost_predict(train3,test3)
    xgboost_predict(train4,test4)
    xgboost_predict(train5,test5)
    #xgboost_predict(train6_5120, test6_5120)