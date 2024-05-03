import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import argparse
from xml.etree.ElementTree import parse
from statistics import mean

from catboost import CatBoostClassifier as catb
from lightgbm import LGBMClassifier as lgb
from xgboost import XGBClassifier as xgb
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rf

import seaborn as sns

import scipy.stats
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, GridSearchCV
from sklearn.decomposition import PCA

from math import sqrt
from scipy.special import ndtri

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import make_pipeline, Pipeline
from lifelines import KaplanMeierFitter as KM
from lifelines import CoxPHFitter as CPH
from lifelines.statistics import logrank_test


def get_args():
    """인자값들 저장해두는 Namespace"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, choices=['vital','any_cvd'])
    parser.add_argument('--out_path', default='E:\\temp\\15_ahi\\final', type=str)
    parser.add_argument('--save_path', default='E:\\temp\\15_ahi\\final', type=str)
    parser.add_argument('--feature_path', default=\
                        'D:\\Hyunji\\Research\\Thesis\\research\\sleep\\data\\SHHS_total_feature_final_hrv.xlsx', type=str)    
    return parser.parse_args()


class MyDataset: 
    """data column명, 학습용 data, 실제 정답"""
    feature_name: str
    x: np.ndarray
    y: np.ndarray

class MyResult:
    """예측결과, 실제 정답, 예측 확률값, feature importance 결과, gridsearch 결과 best parameter 값 """
    def __init__(self):
        self.preds: np.ndarray
        self.reals: np.ndarray
        self.probs: np.ndarray
        self.feature_imporatnces: list
        self.best_params: list

class tree(object):
    def __init__(self, args):
        self.args = args
        self.dataset: MyDataset

        self.target_func = {
            'vital': self.select_mortality_ahi_subject,
            'any_cvd': self.select_cvd_ahi_subject
        }

        self.classifier = {
            """
            classifier_name(str) : classifier(class)
            """
                        
            'xgb': xgb,
            'lgb': lgb,
            'catb': catb,
            'rf': rf,
            'lr': lr
        }

        self.result : dict[str, MyResult] = {}

        self.hyperparmeters = {
            """
            classifier_name(str) : classifier_hyperparameter list (dict)
            """
            
            'xgb' : {
                'xgbclassifier__max_depth': [2,3,4,5,6,7,8,9], 
                'xgbclassifier__n_estimators': [100,200,300,400,500],
                'xgbclassifier__learning_rate': [0.01],
                'xgbclassifier__min_child_weight' : [1, 3, 5],
                'xgbclassifier__objective': ['binary:logistic'],
                'xgbclassifier__use_label_encoder': ['False'],
                'xgbclassifier__random_state':[22]
                },
            'lgb' : {
                'lgbmclassifier__max_depth': [2,3,4,5,6,7,8,9],
                'lgbmclassifier__n_estimators': [100,200,300,400,500],
                'lgbmclassifier__learning_rate': [0.01],
                'lgbmclassifier__min_data_in_leaf': [10, 20, 30, 40, 50],            
                'lgbmclassifier__objective': ['binary'],
                'lgbmclassifier__random_state':[22]        
                },
            'catb' : {
                'catboostclassifier__iterations': [100,200,300,400,500],
                'catboostclassifier__depth': [2,3,4,5,6,7,8,9],
                'catboostclassifier__learning_rate': [0.01],
                'catboostclassifier__l2_leaf_reg': [3,5,7],
                'catboostclassifier__loss_function': ['Logloss'],
                'catboostclassifier__random_state':[22]
                },
            'rf': {
                'randomforestclassifier__max_depth': [2,3,4,5,6,7,8,9],
                'randomforestclassifier__min_samples_leaf': [1,5,10,20],
                'randomforestclassifier__n_estimators': [100,200,300,400,500],
                'randomforestclassifier__random_state':[22]
                },
            'lr': {
                'logisticregression__C': [0.0011],
                'logisticregression__random_state':[22]
            }
        }

        self.drop_list =\
        ['Unnamed: 0','name','vital','sleep_quality','any_cvd','pre_cvd','sleep_apnea','pre_sleep_apnea','cvd_death','censdate','cvd_dthdt','cvd_vital','cvd_date']

    def extract_features(self, outcome: str, total_data: pd.DataFrame) -> pd.DataFrame:
        """
        Target outcome에 따라 grouping된 feature 값을 반환하는 함수
        + grouping_data 에 concated_group 결과 저장
        + outcome에 target outcome 저장

        Args:
            outcome: 'vital' or 'any_cvd' (Str)
            total_data: 전체 feature data (DataFrame) 

        Returns:
            concated_group: grouping된 group의 concat 결과 (pd.DataFrame)
        """
        
        group_0, group_1 = self.target_func[outcome](total_data)
        concated_group = pd.concat([group_0, group_1])
        
        self.grouping_data = concated_group
        self.outcome = outcome
                
        return concated_group

    def prepare_dataset(self, x, y):                
        """
        x, y를 fold 별 train, test로 나누어 dataset을 준비하는 함수
        + cv = cross-validation 방법 저장
        + folded_dataset = fold별 train, test dataset 저장
        + dataset.featuren_name = feature column 이름 저장
        + dataset.x = 학습에 사용될 feature (drop_list 제외된 버전)
        + dataset.y = 결과지로 사용될 outcome 결과
        """  
              
        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        self.cv = cv
        self.folded_dataset = []
        
       # cross validation에 따라 train, test split
        for train_index, test_index in cv.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]            
            dataset = {
                'train' : {
                    'x' : x_train,
                    'y' : y_train
                },
                'test' : {
                    'x' : x_test,
                    'y' : y_test
                }
            }
            self.folded_dataset.append(dataset)
            
        self.dataset.feature_name = self.grouping_data.drop(self.drop_list, axis = 1).columns
        self.dataset.x = self.grouping_data.drop(self.drop_list, axis = 1).to_numpy()
        self.dataset.y = self.grouping_data[f'{self.outcome}'].to_numpy()

    def train(self, classifier_name: str):
        """
        학습 및 결과 저장
        + self.result[classifier_name] = result
        """
        args = self.args
        result = MyResult() #여기서 사용될 reuslt와 해당 함수 맨뒤에서 사용되는 self.reuslt 가 다른 것인지 (변수명은 같아도 되는지?)

        # initialize output (return값 list로 initialize)
        pred_list, real_list, prob_list = [], [], []
        feature_importance_list = []
        best_param_list =[]
        
        ## kfold prediction (5 stratified fold cross-validation 수행 + smote를 통해 data augmentation = make_pipeline 함수 사용)
        smote = SMOTE(random_state=37)
        smp_pipeline = make_pipeline(smote, self.classifier[classifier_name])
        hyperparameter = self.hyperparmeters[classifier_name]
        
        # prepare_dataset에서 저장한 train, test dataset 불러오기
        for kfold, dataset in enumerate(self.folded_dataset):
            x_train, y_train, x_test, y_test = \
                dataset['train']['x'], dataset['train']['y'],\
                dataset['test']['x'], dataset['test']['y']
            
            ## cross-validation에 맞춰 training (gridsearch)
            gird_search_clf = GridSearchCV(smp_pipeline, hyperparameter, scoring='roc_auc', cv=self.cv, refit=True)
            gird_search_clf.fit(x_train, y_train)

            # grid searched model 저장 (pkl로 각 grid search 학습결과 저장)
            with open(args.out_path + f'\\{classifier_name}_{self.outcome}_{str(kfold)}.pkl', 'wb') as f:
                pickle.dump(gird_search_clf, f)
            
            # grid search 결과 best parameter list에 저장
            best_param_list.append(gird_search_clf.best_params_)
            print('GridSearch 최고 점수: ', gird_search_clf.best_score_)
            print('GridSearch 최적 파라미터: ', gird_search_clf.best_params_)

            # gird searched model 중 가장 best estimator의 학습 결과 저장
            clf = gird_search_clf.best_estimator_.steps[1][1]
            pred_prob = clf.predict_proba(x_test)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 1]
                        
            pred_list.append(pred)
            real_list.append(y_test.reshape(-1))
            prob_list.append(pred_prob)
                   
            ## save fold dataset
            # save grid searched model (prediction 결과, predicted probability, 실제 label 값(정답), feature(x 값))
            save_dataset = [pred, pred_prob, y_test.reshape(-1), pd.DataFrame(x_test, columns = self.dataset.feature_name)]
            with open(args.out_path+f'\\{classifier_name}_{self.outcome}_{str(kfold)}_dataset.pkl', 'wb') as f: 
                pickle.dump(save_dataset, f)
            
            ## feature importance (logistic regression 제외 나머지 classifier의 feature importance 내장 함수 이용하여 feature importance 저장)
            feature_importance= clf.feature_importances_ if classifier_name != 'lr' else clf.coef_[0]
            feature_importance_list.append(feature_importance)
      
        result.preds = np.array(pred_list)
        result.reals = np.array(real_list)
        result.probs = np.array(prob_list)
        result.feature_imporatnces = feature_importance_list
        result.best_params = best_param_list
        
        self.result[classifier_name] = result

        
    def save_performance_result(self, outcome: str, grouping_data: pd.DataFrame):
        
        """
        Target outcome에 따라 classifier별 분류 성능을 excel로 저장하는 함수

        classifier 종류:
        1) xgboost
        2) lgbm
        3) catboost
        4) randomforest
        5) logistic regression
        
        excel 저장 결과
        1) classifier별 classification result
        2) classifier별 feature importnace
        3) classifier별 hyperparameter

        Args:
            outcome: 'vital' or 'any_cvd' (Str)
            grouping_data: target outcome에  따라 grouping된 feature data (DataFrame) 

        Returns:
            classifier 별 classificaiton 관련 결과를 excel로 저장

        """
        
        args = self.args
       
        ## gridsearch hyperparameter (classifier별로 gridsearch 할 hyperparmeter list) 
        xgb_param = {
            'xgbclassifier__max_depth': [2,3,4,5,6,7,8,9], 
            'xgbclassifier__n_estimators': [100,200,300,400,500],
            'xgbclassifier__learning_rate': [0.01],
            'xgbclassifier__min_child_weight' : [1, 3, 5],
            'xgbclassifier__objective': ['binary:logistic'],
            'xgbclassifier__use_label_encoder': ['False'],
            'xgbclassifier__random_state':[22]
        }
        
        lgb_param = {
            'lgbmclassifier__max_depth': [2,3,4,5,6,7,8,9],
            'lgbmclassifier__n_estimators': [100,200,300,400,500],
            'lgbmclassifier__learning_rate': [0.01],
            'lgbmclassifier__min_data_in_leaf': [10, 20, 30, 40, 50],            
            'lgbmclassifier__objective': ['binary'],
            'lgbmclassifier__random_state':[22]        
            }
            
        catb_param = {
            'catboostclassifier__iterations': [100,200,300,400,500],
            'catboostclassifier__depth': [2,3,4,5,6,7,8,9],
            'catboostclassifier__learning_rate': [0.01],
            'catboostclassifier__l2_leaf_reg': [3,5,7],
            'catboostclassifier__loss_function': ['Logloss'],
            'catboostclassifier__random_state':[22]
        }        

        rf_param = {
            'randomforestclassifier__max_depth': [2,3,4,5,6,7,8,9],
            'randomforestclassifier__min_samples_leaf': [1,5,10,20],
            'randomforestclassifier__n_estimators': [100,200,300,400,500],
            'randomforestclassifier__random_state':[22]
        }

        lr_param = {
            'logisticregression__C': [0.0011],
            'logisticregression__random_state':[22]
        }

        ## classifier result (classifier 별로 clasification result, feautre importnace, hyperparmeter 저장)
        xgb_result, xgb_feature, xgb_best_param = self.perform_importance_hyperparam('xgb', xgb_param, grouping_data, outcome)
        catb_result, catb_feature, catb_best_param = self.perform_importance_hyperparam('catb', catb_param, grouping_data, outcome)
        lgb_result, lgb_feature, lgb_best_param = self.perform_importance_hyperparam('lgb', lgb_param, grouping_data, outcome)
        rf_result, rf_feature, rf_best_param = self.perform_importance_hyperparam('rf', rf_param, grouping_data, outcome)
        lr_result, lr_feature, lr_best_param = self.perform_importance_hyperparam('lr', lr_param, grouping_data, outcome)

        ## concat result, feature, feature importance of classifier (classifier 별 classification result 값을 모두 합침)
        xgb_result.update(lgb_result)
        xgb_result.update(catb_result)
        xgb_result.update(rf_result)
        xgb_result.update(lr_result)

        ## dict to DataFrame (excel저장을 위해 dataframe으로 변경)
        tree_result = pd.DataFrame(xgb_result)
        tree_feature = pd.concat([xgb_feature, lgb_feature, catb_feature, rf_feature, lr_feature], axis=1)
        tree_param = pd.DataFrame([xgb_best_param, lgb_best_param, catb_best_param, rf_best_param, lr_best_param])

        ## result save (classifcation result, feautre importnace, hyperparmeter를 excel로 저장)
        writer = pd.ExcelWriter(args.out_path + '\\{}_result_demo.xlsx'.format(outcome))      
        
        tree_result.to_excel(writer, sheet_name = '{}_result'.format(outcome))
        tree_feature.to_excel(writer, sheet_name = '{}_feature'.format(outcome))
        tree_param.to_excel(writer, sheet_name = '{}_param'.format(outcome))

        writer.save()


    def perform_importance_hyperparam(self, classifier_name, hyperparameter_list: dict, grouping_data: pd.DataFrame, outcome: str) -> tuple[dict, pd.DataFrame, list]:

        """
        1)fold_perform: performance (fold별 결과, 평균값), 2)plot_feature_importance: feature importance plotting, 3)mean_perform: roc curve plotting

        Args:
            classifier: xgb or catb or lgb or rf or lr (Class)
            (삭제) hyperparameter_list: classifier별 grid search에 넣을 hyperparmeter 값들 (dict)
            (삭제) grouping_data: target outcome에  따라 grouping된 feature data (DataFrame) 
            (삭제) outcome: 'vital' or 'any_cvd' (Str)

        Returns:
            performance_result: fold별 성능값 (dict)
            feature_important_list:  feature 이름/ feature importance / feature의 종류에 따라 저장한 값 (DataFrame)
            hyperparmeter: hyperparmeter 결과 (list)

        """

        ## predicted feature, real feature, probabilities, feature importance, best parameter result
        self.train(classifier_name)
        
        ## save performance result with cross-validation : cross-validation 결과 저장        
        best_fold, mean_accuracy, mean_se, mean_sp, mean_ppv, mean_npv, mean_auc, performance_result = self.fold_perform(real_list, pred_list, prob_list, classifier_name, self.outcome)
        print('{}_mean_accuracy:'.format(classifier_name), mean_accuracy, ', {}_mean_auc:'.format(classifier_name), mean_auc)
        
        
        ## averaging feature importance from fold : fold별 획득된 feature importance값의 평균값을 구하기
        mean_list = []
        kv = len(feature_importance) # number of cv // == self.result[classifier_name].feature_importances
        for k in range(0,kv):
            mean_list.append(np.array(feature_importance[k]))
        feature_mean = np.sum(feature_importance, axis=0)/kv


        ## plot & save bar graph of feature importance : fold의 평균 feature importance 값 plotting
        feature_importance_list = self.plot_feature_importance(feature_mean, self.dataset.feature_name, classifier_name)        
        plt.savefig(args.out_path+'//{}_{}_feature_importance.eps'.format(self.outcome, classifier_name))
        plt.close()     


        ## load and save best model: 성능 가장 좋았던 fold 확인해서 classifier 불러오기 & best_model_ 로 따로 저장해주기
        with open(args.out_path+'\\{}_{}_{}.pkl'.format(classifier_name, self.outcome, str(best_fold)), 'rb') as f:
            grid_search_clf = pickle.load(f)
        clf = grid_search_clf

        with open(args.out_path+'\\{}_{}_best_model_fold{}.pkl'.format(classifier_name, self.outcome, str(best_fold)), 'wb') as f: 
            pickle.dump(clf, f)


        ## best model evaluation: 성능 가장 좋았던 classifier 성능 다시 확인  
        test_index_list=[]
        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        for train_index, test_index in cv.split(x, y):
            test_index_list.append(test_index)
        x_test = x[test_index_list[best_fold]]
        y_test = y[test_index_list[best_fold]]

        clf_temp = clf.best_estimator_.steps[1][1]
        best_param =clf.best_params_
        best_model_pred_prob = clf_temp.predict_proba(x_test)
        best_model_pred = best_model_pred_prob.argmax(1)
        best_model_pred_prob = best_model_pred_prob[:, 1]
        best_model_real = y_test.reshape(-1)
        

        ## result & roc curve
        mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result_final = self.mean_perform(best_model_real, best_model_pred, best_model_pred_prob, classifier_name, self.outcome)
        print('{}_mean_accuracy:'.format(classifier_name), mean_accuracy, '{}_mean_auc:'.format(classifier_name), mean_auc)    

        return performance_result, feature_importance_list, hyperparameter
    


    def save_concat_cv_result(self, outcome: str, grouping_data: pd.DataFrame):

        """
        Target outcome에 따라 classifier별 분류 성능을 excel로 저장하는 함수 (fold 별 예측 결과를 하나의 세트로 간주하여 계산)

        Args:
            outcome: 'vital' or 'any_cvd' (Str)
            grouping_data: target outcome에  따라 grouping된 feature data (DataFrame) 

        Returns: 
            classifier 별 classificaiton 관련 결과를 excel로 저장

        """

        args = self.args

        ## drop feature : input feature로 사용되지 않을 column명 list
        drop_list = ['Unnamed: 0','name','vital','sleep_quality','any_cvd','pre_cvd','sleep_apnea','pre_sleep_apnea','cvd_death','censdate','cvd_dthdt','cvd_vital','cvd_date']
        cv_trainset = []

        ## split data by features and label(outcome) : grouping된 data에서 feature_name(column명), x(feature 값), y(labeled outcome)으로 분리
        feature_name = grouping_data.drop(drop_list, axis=1).columns
        x = grouping_data.drop(drop_list, axis=1).to_numpy()
        y = grouping_data['{}'.format(outcome)].to_numpy()

        ## tree result        
        xgb_concat_result, real_set, xgb_prob_set  = self.concat_prediction_cv_pipeline(x, y, xgb, outcome, feature_name)
        catb_concat_result, real_set, catb_prob_set  = self.concat_prediction_cv_pipeline(x, y, catb, outcome, feature_name)
        lgb_concat_result, real_set, lgb_prob_set  = self.concat_prediction_cv_pipeline(x, y, lgb, outcome, feature_name)
        rf_concat_result, real_set, rf_prob_set  = self.concat_prediction_cv_pipeline(x, y, rf, outcome, feature_name)
        lr_concat_result, real_set, lr_prob_set  = self.concat_prediction_cv_pipeline(x, y, lr, outcome, feature_name)

        ## save plot together
        xgb_concat_result.update(catb_concat_result)
        xgb_concat_result.update(lgb_concat_result)
        xgb_concat_result.update(rf_concat_result)
        xgb_concat_result.update(lr_concat_result)

        ## print delong test 
        xgb_catb_pvalue = self.delong_roc_test(real_set, xgb_prob_set, catb_prob_set, sample_weight=None)
        xgb_lgbm_pvalue = self.delong_roc_test(real_set, xgb_prob_set, lgb_prob_set, sample_weight=None)
        catb_lgbm_pvalue = self.delong_roc_test(real_set, lgb_prob_set, catb_prob_set, sample_weight=None)
        xgb_RF_pvalue = self.delong_roc_test(real_set, xgb_prob_set, rf_prob_set, sample_weight=None)
        xgb_LR_pvalue = self.delong_roc_test(real_set, xgb_prob_set, lr_prob_set, sample_weight=None)
        lgbm_RF_pvalue = self.delong_roc_test(real_set, lgb_prob_set, rf_prob_set, sample_weight=None)
        lgbm_LR_pvalue = self.delong_roc_test(real_set, lgb_prob_set, lr_prob_set, sample_weight=None)
        catb_RF_pvalue = self.delong_roc_test(real_set, catb_prob_set, rf_prob_set, sample_weight=None)
        catb_LR_pvalue = self.delong_roc_test(real_set, catb_prob_set, lr_prob_set, sample_weight=None)
        RF_LR_pvalue = self.delong_roc_test(real_set, rf_prob_set, lr_prob_set, sample_weight=None)

        print('xgb_catb_pvalue: ', 10**xgb_catb_pvalue,' xgb_lgbm_pvalue: ',10**xgb_lgbm_pvalue, ' catb_lgbm_pvalue:',10**catb_lgbm_pvalue, 
            'xgb_RF_pvalue: ', 10**xgb_RF_pvalue,' xgb_LR_pvalue: ',10**xgb_LR_pvalue, ' lgbm_RF_pvalue:',10**lgbm_RF_pvalue, ' lgbm_LR_pvalue:',10**lgbm_LR_pvalue,
       'catb_RF_pvalue: ', 10**catb_RF_pvalue,' catb_LR_pvalue: ',10**catb_LR_pvalue, ' rf_LR_pvalue: ',10**RF_LR_pvalue)

        
        plt.legend()
        plt.savefig(args.save_path+'\\{}_roc_curve_concat.eps'.format(outcome), format='eps')        
        plt.clf()

        ## result save
        writer = pd.ExcelWriter(args.out_path + '\\{}_concat_result.xlsx'.format(outcome)) 
        tree_result = pd.DataFrame(xgb_concat_result, index=[0])

        tree_result.to_excel(writer, sheet_name = '{}_result'.format(outcome))

        writer.save()



    def save_best_result(self, outcome: str, grouping_data: pd.DataFrame):
    
        """
        저장된 best fold classifier를 불러와서 result값을 저장하는 함수

        Args:
            outcome: 'vital' or 'any_cvd' (Str)
            grouping_data: target outcome에  따라 grouping된 feature data (DataFrame) 

        Returns: 
            classifier 별 classificaiton 관련 결과를 excel로 저장

        """
        
        args = self.args

        ## drop feature
        drop_list = ['Unnamed: 0','name','vital','sleep_quality','any_cvd','pre_cvd','sleep_apnea','pre_sleep_apnea','cvd_death','censdate','cvd_dthdt','cvd_vital','cvd_date']
        cv_trainset = []


        # selecting feature (without drop_list) + x(학습용 데이터), y(label 데이터)
        index = grouping_data.drop(drop_list, axis=1).columns
        x = grouping_data.drop(drop_list, axis=1).to_numpy()
        y = grouping_data['{}'.format(outcome)].to_numpy()
     
        ## data split
        test_index_list=[]
        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        for train_index, test_index in cv.split(x, y):
            test_index_list.append(test_index)    

        ## load best cross-validation result classifier (xgb, catb, lgb)
        xgb_clf, xgb_fold = self.load_best_cv(xgb, outcome)
        catb_clf, catb_fold = self.load_best_cv(catb, outcome)
        lgb_clf, lgb_fold = self.load_best_cv(lgb, outcome)
        #clf_list = [xgb_clf, catb_clf, lgb_clf]
            
        xgb_dataset = [x[test_index_list[xgb_fold]], y[test_index_list[xgb_fold]]]
        catb_dataset =  [x[test_index_list[catb_fold]],y[test_index_list[catb_fold]]]
        lgb_dataset = [x[test_index_list[lgb_fold]], y[test_index_list[lgb_fold]]]        
        dataset_list = [xgb_dataset, catb_dataset, lgb_dataset]
        
        ## load best cross-validation result classifier (rf, lr)
        rf_clf, rf_fold = self.load_best_cv(rf, outcome)
        lr_clf, lr_fold = self.load_best_cv(lr, outcome)
        rf_dataset = [x[test_index_list[rf_fold]], y[test_index_list[rf_fold]]] 
        lr_dataset = [x[test_index_list[lr_fold]], y[test_index_list[lr_fold]]]        
        dataset_list = [xgb_dataset, catb_dataset, lgb_dataset, rf_dataset, lr_dataset]        
        clf_list = [xgb_clf, catb_clf,lgb_clf, rf_clf, lr_clf]
        #dataset_list = [lgb_dataset, rf_dataset, lr_dataset]


        ## save plot together
        result, pred_list, param_list = self.best_cv_plot(clf_list, dataset_list, outcome)
                
        plt.clf()
        
        xgb_dataset_df = grouping_data.iloc[test_index_list[xgb_fold]]
        catb_dataset_df = grouping_data.iloc[test_index_list[catb_fold]]
        lgb_dataset_df = grouping_data.iloc[test_index_list[lgb_fold]]

        ## result save
        writer = pd.ExcelWriter(args.save_path + '\\{}_survival_analysis.xlsx'.format(outcome)) 
        tree_result = pd.DataFrame(result)
        tree_pred_list=pd.DataFrame(pred_list)
        tree_param_list=pd.DataFrame(param_list)

        tree_result.to_excel(writer, sheet_name = '{}_result'.format(outcome))
        tree_pred_list.to_excel(writer, sheet_name = '{}_pred_prob'.format(outcome))
        tree_param_list.to_excel(writer, sheet_name = '{}_param'.format(outcome))

        xgb_dataset_df.to_excel(writer, sheet_name = '{}_xgb_test_Data'.format(outcome))
        catb_dataset_df.to_excel(writer, sheet_name = '{}_catb_test_Data'.format(outcome))
        lgb_dataset_df.to_excel(writer, sheet_name = '{}_lgb_test_Data'.format(outcome))
        
        writer.save()   
    

    def km_plot(self, outcome, grouping_data):

        """
        kaplan-meier estimates by high and low risk figure plotting함 함수 
        
        Args:
            outcome: target outcome (str)
            grouping_data: target outcome에  따라 grouping된 feature data (DataFrame) 

        Returns:

        """ 
        args = self.args
        
        ## km plot
        dataset = pd.read_excel('D:\\Hyunji\\Research\\Thesis\\research\\sleep\\result\\SHHS_mortality_prediction_in_ahi_patient.xlsx',sheet_name='survival_15')   
        risk_ix = dataset['risk_group']==1

        ax = plt.subplot(111)

        time = dataset[f'{self.outcome}_date'].to_numpy()
        event = dataset[f'{self.outcome}'].to_numpy()

        ## Fit the data into the model
        km_high_risk = KM()
        ax = km_high_risk.fit(time[risk_ix], event[risk_ix], label='High risk').plot(ax=ax)

        km_low_risk = KM()
        ax = km_low_risk.fit(time[~risk_ix], event[~risk_ix], label='Low risk').plot(ax=ax)


        ## log rank
        results = logrank_test(time[risk_ix], time[~risk_ix], event_observed_A=event[risk_ix], event_observed_B=event[~risk_ix])
        results.print_summary()

        ## Create an estimate
        plt.title('Kaplan-Meier estimates by High and Low risk')
        plt.ylabel('Survival probability (%)')
        plt.xlabel('Time (Days)')
        plt.savefig(args.save_path+'\\{}_km_plot_10.eps'.format(outcome), format='eps')
        plt.clf()


        
    def cox_func(self, outcome, grouping_data):

        """
        Cox Proportional Hazard model의 figure plotting함 함수 
        
        Args:
            outcome: target outcome (str)
            grouping_data: target outcome에  따라 grouping된 feature data (DataFrame) 

        Returns:

        """   

        args = self.args
        
        ## cox plot
        dataset = pd.read_excel('D:\\Hyunji\\Research\\Thesis\\research\\sleep\\result\\SHHS_mortality_prediction_in_ahi_patient.xlsx',sheet_name='survival_10')   
        data = dataset[['vital_date','vital','age','hypertension','diabetes','risk_group']]
        
        plt.figure(figsize=[8,4])
        cph = CPH()
        cph.fit(data, 'vital_date', event_col = 'vital')
        cph.print_summary()
        cph.plot()

        ## Create an estimate
        plt.savefig(args.save_path+'\\{}_cox_HR_10_3.eps'.format(outcome), format='eps')
        plt.clf()



## final result save
    def fold_prediction_importance_hyperparam(self, x: np.ndarray, outcome_label: np.ndarray, classifier, outcome: str, hyperparameter_list: dict, feature_name) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:

        """
        feature(x)를 classifier에 학습시켜 fold별 정답(outcome label)를 예측한 결과, 실제 정답, 예측한 확률값, feature별 importance, gridsearch에 따른 hyperparamter list를 return하는 함수 
        
        학습 방법
        1) stratified kfold cross-validation
        2) smote를 통해 data augmentation = make_pipeline 함수 사용
        3) gridsearch를 통해 hyperparameter 선정

        Args:
            x: features (ndarray)
            outcome_label: labeled outcome (0 or 1) (ndarray)
            classifier: classifier (Class)
            outcome: target outcome (Str)
            hyperparameter_list: hyperparmeter list (dict)
            feature_name: feature column명 

        Returns:
            pred_list: fold별 predicted result =예측 결과 (ndarray)
            real_list: fold별 실제 outcome label =정답 (ndarray)
            prob_list: fold별 predicted probability =예측한 확률값 (ndarray)
            feature_importance_list: fold별 feature importance 값 (list)
            best_param_list : fold별 grid search 결과 선정된 hyperparameter (list)
            
        """

        args = self.args

        # initialize output (return값 list로 initialize)
        pred_list, real_list, prob_list = [], [], []
        feature_importance_list = []
        best_param_list =[]
        
        ## kfold prediction (5 stratified fold cross-validation 수행 + smote를 통해 data augmentation = make_pipeline 함수 사용)

        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        smote = SMOTE(random_state=37)
        smp_pipeline = make_pipeline(smote, classifier())
        kfold=0 #fold initialize
        
        x_train_, y_train_ = x, outcome_label

        # cross validation에 따라 train, test split
        for train_index, test_index in cv.split(x_train_, y_train_):
            print("{} times {} fold:".format(str("1"),str(kfold)))
            x_train, x_test = x_train_[train_index], x_train_[test_index]
            y_train, y_test = y_train_[train_index], y_train_[test_index]
            
            gird_search_clf = GridSearchCV(smp_pipeline, hyperparameter_list, scoring='roc_auc', cv=cv, refit=True)
            gird_search_clf.fit(x_train, y_train)

            ## grid search
            # save grid searched model (pkl로 각 grid search별 best score 저장)
            with open(args.out_path+'\\{}_{}_{}.pkl'.format(str(classifier.__name__), outcome, str(kfold)), 'wb') as f: 
                pickle.dump(gird_search_clf, f)

            print('GridSearch 최고 점수: ', gird_search_clf.best_score_)
            print('GridSearch 최적 파라미터: ', gird_search_clf.best_params_)
    
            # load best estimator for gridsearched with training dataset
            best_param_list.append(gird_search_clf.best_params_)
            clf = gird_search_clf.best_estimator_.steps[1][1]

            pred_prob = clf.predict_proba(x_test)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 1]
            
            ## save fold dataset
            # save grid searched model (prediction 결과, predicted probability, 실제 label 값(정답), feature(x 값))
            save_dataset = [pred, pred_prob, y_test.reshape(-1), pd.DataFrame(x_test, columns=feature_name)]
            with open(args.out_path+'\\{}_{}_{}_dataset.pkl'.format(str(classifier.__name__), outcome, str(kfold)), 'wb') as f: 
                pickle.dump(save_dataset, f)

            pred_list.append(pred)
            real_list.append(y_test.reshape(-1))
            prob_list.append(pred_prob)       
            
            ## feature importance (logistic regression 제외 나머지 classifier의 feature importance 내장 함수 이용하여 feature importance 저장)
            feature_importance= clf.feature_importances_ if classifier.__name__ != 'LogisticRegression' else clf.coef_[0]
            feature_importance_list.append(feature_importance)
            kfold+=1
        
        pred_list = np.array(pred_list)
        real_list = np.array(real_list)
        prob_list = np.array(prob_list)

        return pred_list, real_list, prob_list, feature_importance_list, best_param_list 
    


    def load_best_cv(self, classifier, outcome: str) -> tuple[any, str]:

        """
        저장된 best model pkl값을 읽어오는 함수 

        Args:
            classifier: classifier (Class)
            outcome: target outcome; 'vital' or 'any_cvd' (Str)

        Returns: 
            clf: grid search classifier (Class)
            best_fold: 성능이 가장 높은 fold 값 (int)

        """

        args = self.args
        clf_file = glob.glob(os.path.join(args.out_path,'{}_{}_best_model_fold*.pkl'.format(str(classifier.__name__), outcome)))
        best_fold = int(clf_file[0].split('.')[0][-1])

        # load best parameter
        with open(clf_file[0], 'rb') as f:
            grid_search_clf = pickle.load(f)

        clf = grid_search_clf

        return clf, best_fold 
    
            
    def concat_prediction_cv_pipeline(self, x: np.ndarray, outcome_label: np.ndarray, classifier, outcome: str, feature_name: dict):

        """
        feature(x)를 classifier에 학습시켜 fold별 정답(outcome label)를 예측한 결과, 실제 정답, 예측한 확률값을 return하는 함수 
        
        학습 방법
        1) stratified kfold cross-validation
        2) smote를 통해 data augmentation = make_pipeline 함수 사용
        3) gridsearch를 통해 hyperparameter 선정

        Args:
            x: features (ndarray)
            outcome_label: labeled outcome (0 or 1) (ndarray)
            classifier: classifier (Class)
            outcome: target outcome (Str)
            hyperparameter_list: hyperparmeter list (dict)
            feature_name: feature column명 

        Returns:
            concat_result: fold 별 예측 결과를 하나의 세트로 간주하여 계산한 성능 결과 (dict)
            real_set: fold 별 실제 정답 값을 concatenate한 데이터 (ndarray)
            prob_set: fold 별 실제 예측 probability 값을 concatenate한 데이터 (ndarray)
            
        """

        args = self.args

        pred_list, real_list, prob_list, shap_list, test_index_list = [], [], [], [], []
        feature_importance_list = []
        best_param_list =[]
        
        ## kfold prediction
        n_times=1
        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        kfold=0

        for i in range(0, n_times):
            
            x_train_, y_train_ = x, outcome_label

            for train_index, test_index in cv.split(x_train_, y_train_):
                print("{} times {} fold:".format(str(i),str(kfold)))
                x_train, x_test = x_train_[train_index], x_train_[test_index]
                y_train, y_test = y_train_[train_index], y_train_[test_index]
 
                ## grid search
                # save grid searched model
                with open(args.out_path+'\\{}_{}_{}.pkl'.format(str(classifier.__name__), outcome, str(kfold)), 'rb') as f: 
                    gird_search_clf = pickle.load(f)

                print('GridSearch 최고 점수: ', gird_search_clf.best_score_)
                print('GridSearch 최적 파라미터: ', gird_search_clf.best_params_)
        
                # load best estimator for gridsearched with training dataset
                best_param_list.append(gird_search_clf.best_params_)
                clf = gird_search_clf.best_estimator_.steps[1][1]

                pred_prob = clf.predict_proba(x_test)
                pred = pred_prob.argmax(1)
                pred_prob = pred_prob[:, 1]
            
                pred_list.append(pred)
                real_list.append(y_test.reshape(-1))
                prob_list.append(pred_prob)       
                test_index_list.append(test_index)

                kfold+=1

        #combining results from all iterations
        test_set = test_index_list[0]
        pred_set = pred_list[0]
        real_set = real_list[0]
        prob_set = prob_list[0]        

        for i in range(1,len(test_index_list)):
            test_set = np.concatenate((test_set,test_index_list[i]),axis=0)
            pred_set = np.concatenate((pred_set,pred_list[i]),axis=0)
            real_set = np.concatenate((real_set,real_list[i]),axis=0)
            prob_set = np.concatenate((prob_set,prob_list[i]),axis=0)


        # fold 별 예측 결과를 하나의 세트로 간주하여 성능을 계산
        mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, concat_result = self.mean_perform(real_set, pred_set, prob_set, classifier.__name__, outcome) 
        print('{}_total_accuracy:'.format(classifier.__name__), mean_accuracy, '{}_total_auc:'.format(classifier.__name__), mean_auc)
        
        return concat_result, real_set, prob_set


    def best_cv_plot(self, clf_list: list, dataset_list, outcome):

        """
        classifier별 delong test 결과 print 및 roc curve 모두 겹쳐서 plotting
        

        Args:
            clf_list: classifier list (list = [class, class, class, ...])
            datset_list: classifier 별 best fold dataset list (list =[[x, y], [x, y], [x, y], ...])
            outcome: target outcome (Str)

        Returns:
            all_result: 
            pred_list: 
            param_list: 
            
        """

        args = self.args
        pred_list = []

        all_result=[]
        param_list =[]
        clf_name=['XGBoost','CatBoost','LGBMClassifier', 'RandomForestClassifier', 'LogisticRegression']
        
        ##shap figure
        #clf_name=['XGBoost','CatBoost','LGBMClassifier']


        for i in range(0, len(clf_name)):
            
            dataset = dataset_list[i]
            x_test, y_test = dataset[0], dataset[1]
            real = y_test.reshape(-1)

            clf = clf_list[i].best_estimator_.steps[1][1]
            best_param =clf_list[i].best_params_
            pred_prob = clf.predict_proba(x_test)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 1]
            pred_list.append(pred_prob)
            param_list.append(best_param)
            
            ## result & roc curve
            mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result = self.mean_perform(real, pred, pred_prob, clf_name[i], outcome)
            all_result.append(result) 
            print('{}_mean_accuracy:'.format(clf_name[i]), mean_accuracy, '{}_mean_auc:'.format(clf_name[i]), mean_auc)
            
            '''
            #shap figure
            ## feature importance & shap value
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(x_test)           
            shap_values = -1*shap_values if str(clf_name[i])!='LGBMClassifier' else shap_values[1]         
            
            plt.figure(figsize=[15,8])
            shap.summary_plot(shap_values, pd.DataFrame(x_test,columns=index), plot_size=None, max_display=20, show=False, plot_type='bar')
            plt.savefig(args.out_path+'\\{}_{}_shap_value_barplot_best.eps'.format(target, str(clf_name[i])), format='eps')
            
            plt.figure(figsize=[15,8])
            shap.summary_plot(shap_values, pd.DataFrame(x_test, columns=index), plot_size=None, max_display=20, show=False)
            plt.savefig(args.out_path+'\\{}_{}_shap_value_summaryplot_best.eps'.format(target, str(clf_name[i])), format='eps')
            '''

        plt.legend()
        xgb_lgbm_pvalue = self.delong_roc_test(real, pred_list[0], pred_list[2], sample_weight=None)
        catb_lgbm_pvalue = self.delong_roc_test(real, pred_list[1], pred_list[2], sample_weight=None)
        lgbm_RF_pvalue = self.delong_roc_test(real, pred_list[3], pred_list[2], sample_weight=None)
        lgbm_LR_pvalue = self.delong_roc_test(real, pred_list[4], pred_list[2], sample_weight=None)
        print(' xgb_lgbm_pvalue: ',10**xgb_lgbm_pvalue, ' catb_lgbm_pvalue:',10**catb_lgbm_pvalue, 
            ' lgbm_RF_pvalue:',10**lgbm_RF_pvalue, ' lgbm_LR_pvalue:',10**lgbm_LR_pvalue      )

        plt.savefig(args.save_path+'\\{}_roc_curve_best_model_all_model_0419.eps'.format(outcome), format='eps')
        plt.clf()

        return all_result, pred_list, param_list   


## accuracy/ feature importance
    def fold_perform(self, real: np.ndarray, pred: np.ndarray, prob: np.ndarray, classifier_name: str, outcome: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    
        """
        cross-validation 결과를 저장 + 가장 성능이 좋은 fold, 평균 accuracy, sensitivity, specificity, ppv, npv, auc, fold별 각 성능값을 return하는 함수 
        
        best_fold, mean_accuracy, mean_se, mean_sp, mean_ppv, mean_npv, mean_auc, performance_result

        Args:
                        
            real: fold별 실제 outcome label =정답 (ndarray) == self.result[classifier_name].reals
            pred: fold별 predicted result =예측 결과 (ndarray) == self.result[classifier_name].preds
            prob: fold별 predicted probability =예측한 확률값 (ndarray) == == self.result[classifier_name].probs
            classifier_name: classifier 이름 (str)
            outcome: target outcome (str)

        Returns:
            best_fold: 가장 성능이 좋은 fold (int)
            mean_accuracy: 평균 accuracy (ndarray)
            mean_se: 평균 sensitvity (ndarray)
            mean_sp: 평균 specificity (ndarray)
            mean_ppv: 평균 ppv (ndarray)
            mean_npv: 평균 npv (ndarray)
            mean_auc: 평균 auc (ndarray)
            result: fold별 성능값 (dict)
            
        """    

        args = self.args

        cv_accuracy = []
        cv_se = []
        cv_sp = []
        cv_ppv = []        
        cv_npv = []
        cv_roc = []
        fold = []

        for k in range(0, real.shape[0]):
            accuracy = accuracy_score(real[k], pred[k])
            recall = recall_score(real[k], pred[k], pos_label=1)
            f1 = f1_score(real[k], pred[k], pos_label=1)
            fpr, tpr, thresholds = roc_curve(real[k], prob[k], pos_label=1)
            auc = auc(fpr, tpr)
            se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci,optimal_threshold = self.roc_curve_func(pred[k], real[k], prob[k], classifier_name+': '+str(auc))


            cv_accuracy.append(accuracy)
            cv_se.append(se)
            cv_sp.append(sp)
            cv_ppv.append(ppv)
            cv_npv.append(npv)
            cv_roc.append(auc)
            fold.append(k)

        mean_accuracy = np.round(np.mean(cv_accuracy),4)
        mean_se = np.round(np.mean(cv_se), 4)
        mean_sp = np.round(np.mean(cv_sp),4)
        mean_ppv = np.round(np.mean(cv_ppv),4)
        mean_npv = np.round(np.mean(cv_npv),4)
        mean_auc = np.round(np.mean(cv_roc),4)

        cv_accuracy.append(mean_accuracy)
        cv_se.append(mean_se)
        cv_sp.append(mean_sp)
        cv_ppv.append(mean_ppv)
        cv_npv.append(mean_npv)
        cv_roc.append(mean_auc)
        fold.append('mean')
        best_fold = cv_roc.index(max(cv_roc))

        performance_result = {'fold_{}'.format(classifier_name): fold, 'accuracy_{}'.format(classifier_name):cv_accuracy, 'roc_{}'.format(classifier_name):cv_roc, 'se_{}'.format(classifier_name):cv_se, 'sp_{}'.format(classifier_name):cv_sp, 'ppv_{}'.format(classifier_name):cv_ppv, 'npv_{}'.format(classifier_name):cv_npv }
        plt.clf()

        return best_fold, mean_accuracy, mean_se, mean_sp, mean_ppv, mean_npv, mean_auc, performance_result  


    def mean_perform(self, real: np.ndarray, pred: np.ndarray, prob: np.ndarray, classifier_name: str, outcome: str):

        """
        예측 결과의 accuracy, precision, recall, f1, auc 값과 각 통계값의 mean(confidence_interval) 저장값을 return하며 roc curve를 plotting함 함수 
        
        Args:
            real: 실제 outcome label =정답 (ndarray)
            pred: predicted result =예측 결과 (ndarray)
            prob: predicted probability =예측한 확률값 (ndarray)
            classifier_name: classifier 이름 (str)
            outcome: target outcome (str)

        Returns:
            accuracy: accuracy (float)
            precision:  precision (float)
            recall:  recall (float)
            f1: f1 (float)
            acu_result[0]: auc (float)
            result: precision, recall, f1, roc, sensitivity, specificity, ppv, npv, accuracy, thershold 값 (dict)
            
        """    

        args = self.args

        accuracy = accuracy_score(real, pred)
        precision = precision_score(real, pred, pos_label=1)
        recall = recall_score(real, pred, pos_label=1)
        f1 = f1_score(real, pred, pos_label=1)        
        auc_result = self.auc_ci_Delong(real, prob)

        ## mean(confidence interval) 형태로 저장
        auc = str(round(auc_result[0],3))+' ('+str(round(auc_result[-1][0],3))+', '+str(round(auc_result[-1][1],3))+')'

        ## roc curve를 plotting
        se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci, optimal_threshold= self.roc_curve_func(pred, real, prob, classifier_name+': '+auc) #save path인지 output path인지 확인
        sensitivity = str(round(se,3))+' ('+str(round(se_ci[0],3))+', '+str(round(se_ci[1],3))+')'
        specificity =str(round(sp,3))+' ('+str(round(sp_ci[0],3))+', '+str(round(sp_ci[1],3))+')'
        ppv =str(round(ppv,3))+' ('+str(round(ppv_ci[0],3))+', '+str(round(ppv_ci[1],3))+')'
        npv =str(round(npv,3))+' ('+str(round(npv_ci[0],3))+', '+str(round(npv_ci[1],3))+')'
        result = {'precision_{}'.format(classifier_name):precision, 'recall_{}'.format(classifier_name):recall, 'f1_{}'.format(classifier_name):f1,  'roc_{}'.format(classifier_name):auc,'sensitivity_{}'.format(classifier_name):sensitivity,'specificity_{}'.format(classifier_name):specificity,'ppv_{}'.format(classifier_name):ppv,'npv_{}'.format(classifier_name):npv, 'accuracy_{}'.format(classifier_name):accuracy,'threshold_{}'.format(classifier_name):optimal_threshold}

        return accuracy, precision, recall, f1, auc_result[0], result


    def plot_feature_importance(self, feature__importance, feature_name: list, classifier_name: str) -> pd.DataFrame:

        """
        feature importance 값을 feature의 종류(demographic, sleep feature, HRV)에 따라 
        1) bar plotting하고
        2) feature 이름/ feature importance / feature의 종류에 따라 DataFrame으로 return하는 함수

        Args:
            feature__importance: feature 별 classification 결과에 영향을 준 정도(importance) 값
            feature_name: feature의 name (list)
            classifier_name:  classifier의 이름 (Str)

        Returns:
            feature_importance_category:  feature 이름/ feature importance / feature의 종류에 따라 저장한 값 (DataFrame)
        
        """

        #Create arrays from feature importance and feature names
        feature_importance = np.array(feature__importance)
        feature_names = np.array(feature_name, dtype=object)
        feature_category= []
        for i in range(11): feature_category.append('Demographics') 
        for i in range(18): feature_category.append('Sleep features')
        for i in range(27): feature_category.append('HRV')  

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance, 'feature_category':feature_category}
        feature_importance_category = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_top30 = feature_importance_category.sort_values(by=['feature_importance'], ascending=False)[:30]

        #Define size of bar plot
        plt.figure(figsize=(18,12))
        
        #Plot Searborn bar chart
        sns.barplot(x='feature_importance', y='feature_names', hue='feature_category', data=fi_top30)

        #Add chart labels
        plt.title(classifier_name + ' FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')

        return feature_importance_category


## statistics
    def _proportion_confidence_interval(self, r, n, z):  
        A = 2*r + z**2
        B = z*sqrt(z**2 + 4*r*(1 - r/n))
        C = 2*(n + z**2)
        return ((A-B)/C, (A+B)/C)


    def sensitivity_and_specificity_with_confidence_intervals(self, TP, FP, FN, TN, alpha=0.95):
        z = -ndtri((1.0-alpha)/2)
        
        sensitivity_point_estimate = TP/(TP + FN)
        sensitivity_confidence_interval = self._proportion_confidence_interval(TP, TP + FN, z)
        
        specificity_point_estimate = TN/(TN + FP)
        specificity_confidence_interval = self._proportion_confidence_interval(TN, TN + FP, z)
        print('sensitivity : {} ({} {})'.format(sensitivity_point_estimate, sensitivity_confidence_interval[0], sensitivity_confidence_interval[1]))
        print('specificity : {} ({} {})'.format(specificity_point_estimate, specificity_confidence_interval[0], specificity_confidence_interval[1]))
        
        return sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval, specificity_confidence_interval


    def ppv_and_npv_with_confidence_intervals(self, TP, FP, FN, TN, alpha=0.95):
        z = -ndtri((1.0-alpha)/2)
        
        ppv_estimate = TP/(TP+FP)
        ppv_confidence_interval = self._proportion_confidence_interval(TP, TP + FP, z)
        
        npv_estimate = TN/(TN+FN)
        npv_confidence_interval = self._proportion_confidence_interval(TN, TN + FN, z)
        
        print('ppv : {} ({} {})'.format(ppv_estimate, ppv_confidence_interval[0], ppv_confidence_interval[1]))
        print('npv : {} ({} {})'.format(npv_estimate, npv_confidence_interval[0], npv_confidence_interval[1]))

        
        return ppv_estimate, npv_estimate, ppv_confidence_interval, npv_confidence_interval


    def find_optimal_cutoff(self, real, pred):
        fpr, tpr, thresholds = roc_curve(real, pred)
 
        J = tpr - fpr
        ix = np.argmax(J)
        optimal_threshold = thresholds[ix]

        print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (optimal_threshold, tpr[ix], 1-fpr[ix], J[ix]))

        temp = []
        for t in list(pred):
            if t >= optimal_threshold:
                temp.append(1)
            else:
                temp.append(0)

        TN, FP, FN, TP = confusion_matrix(real, temp).ravel()
        
        sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval, specificity_confidence_interval= self.sensitivity_and_specificity_with_confidence_intervals(TP=TP, FP=FP, FN=FN, TN=TN)
        ppv_estimate, npv_estimate, ppv_confidence_interval, npv_confidence_interval= self.ppv_and_npv_with_confidence_intervals(TP=TP, FP=FP, FN=FN, TN=TN)

        return optimal_threshold, ix, sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval, specificity_confidence_interval, ppv_estimate, npv_estimate, ppv_confidence_interval, npv_confidence_interval

        
    def roc_curve_func(self, pred: np.ndarray, real: np.ndarray, prob: np.ndarray, figure_legend: str):
        """
        roc curve를 plotting함 함수 
        
        Args:
            pred: predicted result =예측 결과 (ndarray)
            real: 실제 outcome label =정답 (ndarray)
            prob: predicted probability =예측한 확률값 (ndarray)
            save_file: target outcome (str)

        Returns:
           sensitivity, specificity, ppv, npv, ppv_ci, npv_ci, optimal_threshold 값 ()
            
        """    
        optimal_threshold, ix, se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci = self.find_optimal_cutoff(real, prob)
        fpr, tpr, thresholds = roc_curve(real, prob, pos_label=1)
        #plt.clf()
        plt.plot(fpr, tpr, label=figure_legend)
        #plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        #plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.savefig(figure_legend, format='eps')

        return se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci,optimal_threshold

           
    def compute_midrank(self, x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1

        return T2


    def compute_midrank_weight(self, x, sample_weight):
        J = np.argsort(x)
        Z = x[J]
        cumulative_weight = np.cumsum(sample_weight[J])
        N = len(x)
        T = np.zeros(N, dtype=np.float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = cumulative_weight[i:j].mean()
            i = j
        T2 = np.empty(N, dtype=np.float)
        T2[J] = T
        return T2


    def fastDeLong(self, predictions_sorted_transposed, label_1_count, sample_weight):
        if sample_weight is None:
            return self.fastDeLong_no_weights(
                predictions_sorted_transposed,
                label_1_count)
        else:
            return self.fastDeLong_weights(
                predictions_sorted_transposed,
                label_1_count,
                sample_weight)


    def fastDeLong_weights(self, pred_sorted_transposed, label_1_count, sample_weight):
        # Short variables are named as they are in the paper
        m = label_1_count
        n = pred_sorted_transposed.shape[1] - m
        positive_examples = pred_sorted_transposed[:, :m]
        negative_examples = pred_sorted_transposed[:, m:]
        k = pred_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float)
        ty = np.empty([k, n], dtype=np.float)
        tz = np.empty([k, m + n], dtype=np.float)
        for r in range(k):
            tx[r, :] = self.compute_midrank_weight(
                positive_examples[r, :], sample_weight[:m])
            ty[r, :] = self.compute_midrank_weight(
                negative_examples[r, :], sample_weight[m:])
            tz[r, :] = self.compute_midrank_weight(
                pred_sorted_transposed[r, :], sample_weight)
        total_positive_weights = sample_weight[:m].sum()
        total_negative_weights = sample_weight[m:].sum()
        pair_weights = np.dot(
            sample_weight[:m, np.newaxis],
            sample_weight[np.newaxis, m:])
        total_pair_weights = pair_weights.sum()
        aucs = (
            sample_weight[:m]*(tz[:, :m] - tx)
        ).sum(axis=1) / total_pair_weights
        v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
        v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov


    def fastDeLong_no_weights(self, predictions_sorted_transposed, label_1_count):
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float)
        ty = np.empty([k, n], dtype=np.float)
        tz = np.empty([k, m + n], dtype=np.float)
        for r in range(k):
            tx[r, :] = self.compute_midrank(positive_examples[r, :])
            ty[r, :] = self.compute_midrank(negative_examples[r, :])
            tz[r, :] = self.compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov


    def calc_pvalue(self, aucs, sigma):
        l_aux = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l_aux, sigma), l_aux.T))
        return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


    def compute_ground_truth_statistics(self, ground_truth, sample_weight):
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        if sample_weight is None:
            ordered_sample_weight = None
        else:
            ordered_sample_weight = sample_weight[order]

        return order, label_1_count, ordered_sample_weight


    def delong_roc_variance(self, ground_truth, predictions, sample_weight=None):
        ground_truth_stats = self.compute_ground_truth_statistics(
            ground_truth,
            sample_weight)
        order, label_1_count, ordered_sample_weight = ground_truth_stats

        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = self.fastDeLong(
            predictions_sorted_transposed,
            label_1_count,
            ordered_sample_weight)

        assert_msg = "There is a bug in the code, please forward this to the devs"
        assert len(aucs) == 1, assert_msg
        return aucs[0], delongcov


    def delong_roc_test(self, ground_truth, pred_one, pred_two, sample_weight=None):
        order, label_1_count, _ = self.compute_ground_truth_statistics(
            ground_truth,
            sample_weight)

        predictions_sorted_transposed = np.vstack(
            (pred_one, pred_two))[:, order]

        aucs, delongcov = self.fastDeLong(
            predictions_sorted_transposed,
            label_1_count,
            sample_weight)

        # print(aucs, delongcov)
        return self.calc_pvalue(aucs, delongcov)


    def auc_ci_Delong(self, y_true, y_scores, alpha=.95):

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Get AUC and AUC variance
        auc, auc_var = self.delong_roc_variance(
            y_true,
            y_scores)

        auc_std = np.sqrt(auc_var)

        # Confidence Interval
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        lower_upper_ci = stats.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)

        lower_upper_ci[lower_upper_ci > 1] = 1

        return auc, auc_var, lower_upper_ci

## select target subjects        
    def select_mortality_ahi_subject(self, total_data: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:

        """
        전체 feature data를 alive, deceased로 grouping하는 함수

        ahi 값이 5 이상인 data 중 아래 기준에 따라 grouping
        1) alive: vital(생존) 값이 1 이거나 vital 값이 0이면서 censdate(사망날짜)가 365*15 초과 
        2) deceased: vital 값이 0이면서 censdate가 365*15 이하
        
        Args:
            total_data: 전체 feature data (DataFrame) 

        Returns:
            alive: 1번 조건을 만족하는 feature (DataFrame)
            decease: 2번 조건을 만족하는 feature (DataFrame)
        """

        ## including ahi >= 5 
        df_ahi: pd.DataFrame = total_data[total_data['ahi']>=5].reset_index(drop=True)
        
        ## select vital ==0 or vital ==1 subject
        alive: pd.DataFrame = df_ahi[df_ahi['vital'] == 1 or (df_ahi['vital'] == 0 and df_ahi['censdate'] > 365*15)].reset_index(drop=True)
        alive[:, 'vital'] = 0
        deceased: pd.DataFrame = df_ahi[df_ahi['vital'] == 0 and df_ahi['censdate'] <= 365*15].reset_index(drop=True)
        deceased[:, 'vital'] = 1
        
        print('alive:', len(alive), '\n deceased:', len(deceased))

        return alive, deceased


    def select_cvd_ahi_subject(self, total_data: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    
        """
        전체 feature data를 non_cvd, any_cvd로 grouping하는 함수

        ahi 값이 5 이상이면서, pre_cvd 값이 0인 data 중 아래 기준에 따라 grouping
        1) non_cvd: any_cvd(심혈관 질환 유무) 값이 0 
        2) any_Cvd: any_cvd 값이 0이 아님
        
        Args:
            total_data: 전체 feature data (DataFrame) 

        Returns:
            non_cvd: 1번 조건을 만족하는 feature (DataFrame)
            any_cvd: 2번 조건을 만족하는 feature (DataFrame)
        """
   
        ## including ahi >= 5 
        df_ahi: pd.DataFrame = total_data[total_data['ahi']>=5].reset_index(drop=True)

        ## excluding pre-cvd 
        df_pre_cvd: pd.DataFrame = df_ahi[df_ahi['pre_cvd']==0].reset_index(drop=True)

        ## select non_cvd & any_cvd subject
        non_cvd: pd.DataFrame = df_pre_cvd[df_pre_cvd['any_cvd']==0].reset_index(drop=True)

        any_cvd: pd.DataFrame = df_pre_cvd[df_pre_cvd['any_cvd']!=0 and df_pre_cvd['cvd_date']!=0].reset_index(drop=True)
        any_cvd[:,'any_cvd'] = 1

        print('any_cvd:', len(any_cvd), '\n non_cvd:', len(non_cvd))
        return non_cvd, any_cvd



if __name__ == '__main__':
    args = get_args()
    path = args.feature_path
    total_feature = pd.read_excel(path)
    tree_shhs= tree(args)
    
    # classification -mortality subject (target outcome에 따라 feautre를 grouping)
    group_data = tree_shhs.extract_features(args.target, total_feature)

    # 학습이 필요한 경우
    tree_shhs.save_performance_result(args.target, group_data)

    # 이미 학습된 classifier가 저장된 경우    
    tree_shhs.save_concat_cv_result(args.target, group_data)
    tree_shhs.save_best_result(args.target, group_data)
    tree_shhs.km_plot(args.target,group_data)
    tree_shhs.cox_func(args.target,group_data)
    