import os
import glob
import numpy as np
from scipy.signal import butter, lfilter
import sys
sys.path.extend(['D:\\Hyunji\\Research\\Sleep\\Quality prediction\\code\\src'])
import pandas as pd
from pyhrv import tools as tools
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import biosppy
import pickle
from hrvanalysis import *
import matplotlib.pyplot as plt
from itertools import chain
from collections import defaultdict
import argparse
from ctypes import DEFAULT_MODE
import numpy as np
from xml.etree.ElementTree import parse
from scipy.signal import butter, lfilter
from statistics import mean

import sys
sys.path.extend(['D:\\Hyunji\\Research\\Sleep\\Quality prediction\\code\src'])
import pandas as pd
from pyhrv import tools as tools

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split
from hrvanalysis import *
import matplotlib.pyplot as plt 
import plotly.offline as py

import plotly.graph_objs as go
import plotly.tools as tls

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report
import seaborn as sns

import os
from sklearn.utils import resample
import numpy as np
import pandas as pd
import pickle
import scipy.stats
from catboost import CatBoostClassifier as catb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold, ParameterGrid, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from math import sqrt
from scipy.special import ndtri
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier as lgb
from xgboost import XGBClassifier as xgb
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rf
from xgbse.metrics import concordance_index
import shap
from imblearn.pipeline import make_pipeline, Pipeline
from lifelines import KaplanMeierFitter as KM
from lifelines import CoxPHFitter as CPH

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='E:\\temp\\15_ahi\\final'
                        , type=str)
    parser.add_argument('--save_path', default='E:\\temp\\15_ahi\\final'
                        , type=str)
    parser.add_argument('--feature_path', default='D:\\Hyunji\\Research\\Thesis\\research\\sleep\\data\\SHHS_total_feature_final_hrv.xlsx', type=str)
    
    return parser.parse_args()


class tree(object):
    def __init__(self, args):
        self.args = args
## save result

    def performance_result(self, target, group):
        args = self.args
       
        ## tree parameter

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
            #'logisticregression__max_iter': [200, 300, 400, 500, 1000, 1500,2000],
            #'logisticregression__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'logisticregression__random_state':[22]
            }

        ## tree result
        xgb_result, xgb_feature, xgb_best_param = self.model_cv_result(xgb, xgb_param, group, target)
        catb_result, catb_feature, catb_best_param = tree_shhs.model_cv_result(catb, catb_param, group, target)
        lgb_result, lgb_feature, lgb_best_param = tree_shhs.model_cv_result(lgb, lgb_param, group, target)
        rf_result, rf_feature, rf_best_param = tree_shhs.model_cv_result(rf, rf_param, group, target)
        lr_result, lr_feature, lr_best_param = tree_shhs.model_cv_result(lr, lr_param, group, target)

        ## result, feature, feature importance
        xgb_result.update(lgb_result)
        xgb_result.update(catb_result)
        xgb_result.update(rf_result)
        xgb_result.update(lr_result)

        tree_result = pd.DataFrame(xgb_result)
        tree_feature = pd.concat([xgb_feature, lgb_feature, catb_feature, rf_feature, lr_feature], axis=1)
        tree_param = pd.DataFrame([xgb_best_param, lgb_best_param, catb_best_param, rf_best_param, lr_best_param])

        ## result save
        writer = pd.ExcelWriter(args.out_path + '\\{}_result_demo.xlsx'.format(target))      
        
        tree_result.to_excel(writer, sheet_name = '{}_result'.format(target))
        tree_feature.to_excel(writer, sheet_name = '{}_feature'.format(target))
        tree_param.to_excel(writer, sheet_name = '{}_param'.format(target))


        writer.save()

    def model_cv_result(self, model, param, group, target):
        args = self.args
        ## drop feature
        drop_list = ['Unnamed: 0','name','vital','sleep_quality','any_cvd','pre_cvd','sleep_apnea','pre_sleep_apnea','cvd_death','censdate','cvd_dthdt','cvd_vital','cvd_date']

        # for only hrv features       
        index = group.drop(drop_list, axis=1).columns
        x = group.drop(drop_list, axis=1).to_numpy()
        y = group['{}'.format(target)].to_numpy()

        pred_list, real_list, prob_list, feature_importance, param_list = self.prediction_cv_pipeline(x, y, model, target, param, index)
        
        model = model.__name__
        best_fold, mean_accuracy, mean_se, mean_sp, mean_ppv, mean_npv, mean_auc, result   = self.cv_accuracy(real_list, pred_list, prob_list,model, target)
        print('{}_mean_accuracy:'.format(model), mean_accuracy, ', {}_mean_auc:'.format(model), mean_auc)
        
        feature_mean = self.feature_mean_(feature_importance)
        feature_importance_list = self.plot_feature_importance(feature_mean, index, model)
        plt.savefig(args.out_path+'//{}_{}_feature_importance.eps'.format(target, model))
        plt.close()     


        clf_temp = self.final_result(model, target, best_fold)
        # best model evaluation : 
        test_index_list=[]
        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        for train_index, test_index in cv.split(x, y):
            test_index_list.append(test_index)
        x_test = x[test_index_list[best_fold]]
        y_test = y[test_index_list[best_fold]]

        clf = clf_temp.best_estimator_.steps[1][1]
        best_param =clf_temp.best_params_
        pred_prob = clf.predict_proba(x_test)
        pred = pred_prob.argmax(1)
        pred_prob = pred_prob[:, 1]
        real = y_test.reshape(-1)
        
        ## result & roc curve
        mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result_final = self.accuracy_pipeline(real, pred, pred_prob, model, target)
        print('{}_mean_accuracy:'.format(model), mean_accuracy, '{}_mean_auc:'.format(model), mean_auc)        


        return result, feature_importance_list, param_list

    def save_concat_cv_result(self, target, group):
        args = self.args
        ## drop feature
        drop_list = ['Unnamed: 0','name','vital','sleep_quality','any_cvd','pre_cvd','sleep_apnea','pre_sleep_apnea','cvd_death','censdate','cvd_dthdt','cvd_vital','cvd_date']
        cv_trainset = []
        # for only hrv features       
        index = group.drop(drop_list, axis=1).columns
        x = group.drop(drop_list, axis=1).to_numpy()
        y = group['{}'.format(target)].to_numpy()

        ## tree result        
        xgb_fold_result, xgb_concat_result, real_set, xgb_prob_set  = self.concat_prediction_cv_pipeline(x, y, xgb, target, index)
        catb_fold_result, catb_concat_result, real_set, catb_prob_set  = self.concat_prediction_cv_pipeline(x, y, catb, target, index)
        lgb_fold_result, lgb_concat_result, real_set, lgb_prob_set  = self.concat_prediction_cv_pipeline(x, y, lgb, target, index)
        rf_fold_result, rf_concat_result, real_set, rf_prob_set  = self.concat_prediction_cv_pipeline(x, y, rf, target, index)
        lr_fold_result, lr_concat_result, real_set, lr_prob_set  = self.concat_prediction_cv_pipeline(x, y, lr, target, index)

        ## save plot together
        xgb_concat_result.update(catb_concat_result)
        xgb_concat_result.update(lgb_concat_result)
        xgb_concat_result.update(rf_concat_result)
        xgb_concat_result.update(lr_concat_result)

        ## delong test
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
        plt.savefig(args.save_path+'\\{}_roc_curve_concat.eps'.format(target), format='eps')        
        plt.clf()

        ## result save
        writer = pd.ExcelWriter(args.out_path + '\\{}_concat_result.xlsx'.format(target)) 
        tree_result = pd.DataFrame(xgb_concat_result, index=[0])

        tree_result.to_excel(writer, sheet_name = '{}_result'.format(target))

        '''
        ## save plot together
        xgb_fold_result.update(catb_fold_result)
        xgb_fold_result.update(lgb_fold_result)
        xgb_fold_result.update(rf_fold_result)
        xgb_fold_result.update(lr_fold_result)        
        tree_result_fold = pd.DataFrame(xgb_fold_result)
        tree_result_fold.to_excel(writer, sheet_name = '{}_fold_result'.format(target))
        '''

        writer.save()

    def save_best_result(self, target, group):
        args = self.args
         ## drop feature
        drop_list = ['Unnamed: 0','name','vital','sleep_quality','any_cvd','pre_cvd','sleep_apnea','pre_sleep_apnea','cvd_death','censdate','cvd_dthdt','cvd_vital','cvd_date']
        cv_trainset = []
        # for only hrv features       
        index = group.drop(drop_list, axis=1).columns
        x = group.drop(drop_list, axis=1).to_numpy()
        y = group['{}'.format(target)].to_numpy()
     
        ## data split
        test_index_list=[]
        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        for train_index, test_index in cv.split(x, y):
            test_index_list.append(test_index)    

        ## tree result
        xgb_clf, xgb_fold = self.load_best_cv(xgb, target)
        catb_clf, catb_fold = self.load_best_cv(catb, target)
        lgb_clf, lgb_fold = self.load_best_cv(lgb, target)
        #clf_list = [xgb_clf, catb_clf, lgb_clf]
            
        xgb_dataset = [x[test_index_list[xgb_fold]], y[test_index_list[xgb_fold]]]
        catb_dataset =  [x[test_index_list[catb_fold]],y[test_index_list[catb_fold]]]
        lgb_dataset = [x[test_index_list[lgb_fold]], y[test_index_list[lgb_fold]]]        
        dataset_list = [xgb_dataset, catb_dataset, lgb_dataset]
        
        ## rf & lr          
        rf_clf, rf_fold = self.load_best_cv(rf, target)
        lr_clf, lr_fold = self.load_best_cv(lr, target)
        rf_dataset = [x[test_index_list[rf_fold]], y[test_index_list[rf_fold]]] 
        lr_dataset = [x[test_index_list[lr_fold]], y[test_index_list[lr_fold]]]        
        dataset_list = [xgb_dataset, catb_dataset, lgb_dataset, rf_dataset, lr_dataset]        
        clf_list = [xgb_clf, catb_clf,lgb_clf, rf_clf, lr_clf]
        #dataset_list = [lgb_dataset, rf_dataset, lr_dataset]


        ## save plot together
        result, pred_list, param_list = self.best_cv_plot(clf_list, dataset_list, cv_trainset, index, target)
                
        plt.clf()
        
        xgb_dataset_df = group.iloc[test_index_list[xgb_fold]]
        catb_dataset_df = group.iloc[test_index_list[catb_fold]]
        lgb_dataset_df = group.iloc[test_index_list[lgb_fold]]

        ## result save
        writer = pd.ExcelWriter(args.save_path + '\\{}_survival_analysis.xlsx'.format(target)) 
        tree_result = pd.DataFrame(result)
        tree_pred_list=pd.DataFrame(pred_list)
        tree_param_list=pd.DataFrame(param_list)

        tree_result.to_excel(writer, sheet_name = '{}_result'.format(target))
        tree_pred_list.to_excel(writer, sheet_name = '{}_pred_prob'.format(target))
        tree_param_list.to_excel(writer, sheet_name = '{}_param'.format(target))

        xgb_dataset_df.to_excel(writer, sheet_name = '{}_xgb_test_Data'.format(target))
        catb_dataset_df.to_excel(writer, sheet_name = '{}_catb_test_Data'.format(target))
        lgb_dataset_df.to_excel(writer, sheet_name = '{}_lgb_test_Data'.format(target))
        

        writer.save()   
    
    def km_plot(self, target, group):
        args = self.args
        
        ## km plot
        dataset = pd.read_excel('D:\\Hyunji\\Research\\Thesis\\research\\sleep\\result\\SHHS_mortality_prediction_in_ahi_patient.xlsx',sheet_name='survival_15')   
        risk_ix = dataset['risk_group']==1

        ax = plt.subplot(111)

        time = dataset['{}_date'.format(target)].to_numpy()
        event = dataset['{}'.format(target)].to_numpy()

        ## Fit the data into the model
        km_high_risk = KM()
        ax = km_high_risk.fit(time[risk_ix], event[risk_ix], label='High risk').plot(ax=ax)

        km_low_risk = KM()
        ax = km_low_risk.fit(time[~risk_ix], event[~risk_ix], label='Low risk').plot(ax=ax)


        ## log rank
        from lifelines.statistics import logrank_test
        results = logrank_test(time[risk_ix], time[~risk_ix], event_observed_A=event[risk_ix], event_observed_B=event[~risk_ix])
        results.print_summary()

        ## Create an estimate
        plt.title('Kaplan-Meier estimates by High and Low risk')
        plt.ylabel('Survival probability (%)')
        plt.xlabel('Time (Days)')
        plt.savefig(args.save_path+'\\{}_km_plot_10.eps'.format(target), format='eps')
        plt.clf()

        
    def cox_func(self, target, group):
        args = self.args
        
        ## km plot
        dataset = pd.read_excel('D:\\Hyunji\\Research\\Thesis\\research\\sleep\\result\\SHHS_mortality_prediction_in_ahi_patient.xlsx',sheet_name='survival_10')   
        data = dataset[['vital_date','vital','age','hypertension','diabetes','risk_group']]
        
        plt.figure(figsize=[8,4])
        cph = CPH()
        cph.fit(data, 'vital_date', event_col = 'vital')
        cph.print_summary()
        cph.plot()

        ## Create an estimate
        plt.savefig(args.save_path+'\\{}_cox_HR_10_3.eps'.format(target), format='eps')
        plt.clf()


## final result save
    def prediction_cv_pipeline(self, x, label, classify, target, param, index):

        args = self.args

        pred_list, real_list, prob_list = [], [], []
        feature_importance_list = []
        best_param_list =[]
        
        ## kfold prediction
        from imblearn.pipeline import make_pipeline
        from sklearn.decomposition import PCA
        n_times=1
        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        smote = SMOTE(random_state=37)
        #smp_pipeline = make_pipeline(smote, classify(**param))
        smp_pipeline = make_pipeline(smote, classify())
        kfold=0

        for i in range(0, n_times):
            
            #x_train_, x_test_, y_train_, y_test_ = train_test_split(x, label, test_size=0.2, stratify = label, random_state=387)
            x_train_, y_train_ = x, label

            for train_index, test_index in cv.split(x_train_, y_train_):
                print("{} times {} fold:".format(str(i),str(kfold)))
                x_train, x_test = x_train_[train_index], x_train_[test_index]
                y_train, y_test = y_train_[train_index], y_train_[test_index]
                
                gird_search_clf = GridSearchCV(smp_pipeline, param, scoring='roc_auc', cv=cv, refit=True)
                gird_search_clf.fit(x_train, y_train)

                ## grid search
                # save grid searched model
                with open(args.out_path+'\\{}_{}_{}.pkl'.format(str(classify.__name__), target, str(kfold)), 'wb') as f: 
                    pickle.dump(gird_search_clf, f)

                print('GridSearch 최고 점수: ', gird_search_clf.best_score_)
                print('GridSearch 최적 파라미터: ', gird_search_clf.best_params_)
        
                # load best estimator for gridsearched with training dataset
                best_param_list.append(gird_search_clf.best_params_)
                clf = gird_search_clf.best_estimator_.steps[1][1]
                
                ## without grid search
                #clf = smp_pipeline.fit(x_train, y_train)

                pred_prob = clf.predict_proba(x_test)
                pred = pred_prob.argmax(1)
                pred_prob = pred_prob[:, 1]
                
                ## save fold dataset
                # save grid searched model
                save_dataset = [pred, pred_prob, y_test.reshape(-1), pd.DataFrame(x_test, columns=index)]
                with open(args.out_path+'\\{}_{}_{}_dataset.pkl'.format(str(classify.__name__), target, str(kfold)), 'wb') as f: 
                    pickle.dump(save_dataset, f)

                pred_list.append(pred)
                real_list.append(y_test.reshape(-1))
                prob_list.append(pred_prob)       
                
                ## feature importance
                feature_importance= clf.feature_importances_ if classify.__name__ != 'LogisticRegression' else clf.coef_[0]
                feature_importance_list.append(feature_importance)
                kfold+=1
        
        pred_list = np.array(pred_list)
        real_list = np.array(real_list)
        prob_list = np.array(prob_list)

        return pred_list, real_list, prob_list, feature_importance_list, best_param_list 

    def load_best_cv(self, classify, target):
        args = self.args
        clf_file = glob.glob(os.path.join(args.out_path,'{}_{}_best_model_fold*.pkl'.format(str(classify.__name__), target)))
        best_fold = int(clf_file[0].split('.')[0][-1])

        # load best parameter
        with open(clf_file[0], 'rb') as f:
            grid_search_clf = pickle.load(f)

        clf = grid_search_clf

        return clf, best_fold 
            
    def concat_prediction_cv_pipeline(self, x, label, classify, target, index):

        args = self.args

        pred_list, real_list, prob_list, shap_list,test_index_list = [], [], [], [], []
        feature_importance_list = []
        best_param_list =[]
        
        ## kfold prediction

        n_times=1
        cv = StratifiedKFold(5, shuffle=True, random_state=250)
        kfold=0

        for i in range(0, n_times):
            
            x_train_, y_train_ = x, label

            for train_index, test_index in cv.split(x_train_, y_train_):
                print("{} times {} fold:".format(str(i),str(kfold)))
                x_train, x_test = x_train_[train_index], x_train_[test_index]
                y_train, y_test = y_train_[train_index], y_train_[test_index]
 
                ## grid search
                # save grid searched model
                with open(args.out_path+'\\{}_{}_{}.pkl'.format(str(classify.__name__), target, str(kfold)), 'rb') as f: 
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
                                
                ## feature importance & shap value
                #explainer = shap.TreeExplainer(clf)
                #shap_values = explainer.shap_values(x_test) if classify.__name__ !='LGBMClassifier' else explainer.shap_values(x_test)[1]             
                #shap_list.append(shap_values)

                kfold+=1
        
        pred_list_temp = np.array(pred_list)
        real_list_temp = np.array(real_list)
        prob_list_temp = np.array(prob_list)
        #best_fold, mean_accuracy, mean_se, mean_sp, mean_ppv, mean_npv, mean_auc, fold_result   = self.cv_accuracy(real_list_temp, pred_list_temp, prob_list_temp ,classify.__name__, target)
        fold_result=[]

        #combining results from all iterations
        test_set = test_index_list[0]
        pred_set = pred_list[0]
        real_set = real_list[0]
        prob_set = prob_list[0]
        
        #shap_values = np.array(shap_list[0])


        for i in range(1,len(test_index_list)):
            test_set = np.concatenate((test_set,test_index_list[i]),axis=0)
            pred_set = np.concatenate((pred_set,pred_list[i]),axis=0)
            real_set = np.concatenate((real_set,real_list[i]),axis=0)
            prob_set = np.concatenate((prob_set,prob_list[i]),axis=0)

            #shap_values = np.concatenate((shap_values,np.array(shap_list[i])),axis=0)
        '''
        #bringing back variable names    
        X_test = pd.DataFrame(x[test_set],columns=index)

        ## plotting shap plot
        plt.figure(figsize=[15,8])
        shap.summary_plot(shap_values, X_test, plot_size=None, max_display=20, show=False, plot_type='bar')
        plt.savefig(args.out_path+'\\{}_{}_shap_value_barplot_all.eps'.format(target, str(classify.__name__)), format='eps')
        
        plt.figure(figsize=[15,8])
        shap.summary_plot(shap_values, X_test, plot_size=None, max_display=20, show=False)
        plt.savefig(args.out_path+'\\{}_{}_shap_value_summaryplot_all.eps'.format(target, str(classify.__name__)), format='eps')    
        '''
        mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, concat_result = self.accuracy_pipeline(real_set, pred_set, prob_set, classify.__name__, target) 
        print('{}_total_accuracy:'.format(classify.__name__), mean_accuracy, '{}_total_auc:'.format(classify.__name__), mean_auc)
        
        return fold_result, concat_result, real_set, prob_set

    def best_cv_plot(self, clf_list, dataset_list, cv_trainset, index, target):
        args = self.args
        pred_list = []

        all_result=[]
        param_list =[]
        clf_name=['XGBoost','CatBoost','LGBMClassifier', 'RandomForestClassifier', 'LogisticRegression']
        #clf_name=['LGBMClassifier', 'RandomForestClassifier', 'LogisticRegression']

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
            mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result = self.accuracy_pipeline(real, pred, pred_prob, clf_name[i], target)
            all_result.append(result) 
            print('{}_mean_accuracy:'.format(clf_name[i]), mean_accuracy, '{}_mean_auc:'.format(clf_name[i]), mean_auc)
            
            '''
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

        plt.savefig(args.save_path+'\\{}_roc_curve_best_model_all_model_0419.eps'.format(target), format='eps')
        plt.clf()

        return all_result, pred_list, param_list   


## accuracy/ feature importance
    def cv_accuracy(self, real, pred, prob, classify, target):
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
            fpr, tpr, thresholds = metrics.roc_curve(real[k], prob[k], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci,optimal_threshold = self.roc_curve_func(pred[k], real[k], prob[k],classify+': '+str(auc))


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

        result = {'fold_{}'.format(classify): fold, 'accuracy_{}'.format(classify):cv_accuracy, 'roc_{}'.format(classify):cv_roc, 'se_{}'.format(classify):cv_se, 'sp_{}'.format(classify):cv_sp, 'ppv_{}'.format(classify):cv_ppv, 'npv_{}'.format(classify):cv_npv }
        plt.clf()

        return best_fold, mean_accuracy, mean_se, mean_sp, mean_ppv, mean_npv, mean_auc, result  

    def accuracy_pipeline(self, real, pred, prob, classify, target):
        args = self.args

        accuracy = accuracy_score(real, pred)
        precision = precision_score(real, pred, pos_label=1)
        recall = recall_score(real, pred, pos_label=1)
        f1 = f1_score(real, pred, pos_label=1)
        auc_result = self.auc_ci_Delong(real, prob)
        auc = str(round(auc_result[0],3))+' ('+str(round(auc_result[-1][0],3))+', '+str(round(auc_result[-1][1],3))+')'
        se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci, optimal_threshold= self.roc_curve_func(pred, real, prob, classify+': '+auc) #save path인지 output path인지 확인
        #se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci, optimal_threshold= self.roc_curve_func(pred, real, prob, args.out_path+'\\{}_{}_roc_curve_LR_RF.eps'.format(classify, target)) 
        sensitivity = str(round(se,3))+' ('+str(round(se_ci[0],3))+', '+str(round(se_ci[1],3))+')'
        specificity =str(round(sp,3))+' ('+str(round(sp_ci[0],3))+', '+str(round(sp_ci[1],3))+')'
        ppv =str(round(ppv,3))+' ('+str(round(ppv_ci[0],3))+', '+str(round(ppv_ci[1],3))+')'
        npv =str(round(npv,3))+' ('+str(round(npv_ci[0],3))+', '+str(round(npv_ci[1],3))+')'
        result = {'precision_{}'.format(classify):precision, 'recall_{}'.format(classify):recall, 'f1_{}'.format(classify):f1,  'roc_{}'.format(classify):auc,'sensitivity_{}'.format(classify):sensitivity,'specificity_{}'.format(classify):specificity,'ppv_{}'.format(classify):ppv,'npv_{}'.format(classify):npv, 'accuracy_{}'.format(classify):accuracy,'threshold_{}'.format(classify):optimal_threshold}

        return accuracy, precision, recall, f1, auc_result[0], result



    def feature_mean_(self, list):
        mean_list = []
        kv = len(list) # number of cv
        for k in range(0,kv):
            mean_list.append(np.array(list[k]))
        feature_mean = np.sum(mean_list, axis=0)/kv
        return feature_mean

    def plot_feature_importance(self, importance, names, model_type):

        #Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names, dtype=object)
        feature_category= []
        for i in range(11): feature_category.append('Demographics') 
        #for i in range(18): feature_category.append('Sleep features')
        #for i in range(27): feature_category.append('HRV')  

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance, 'feature_category':feature_category}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        #fi_top30 = fi_df.sort_values(by=['feature_importance'], ascending=False)[:30]
        fi_top30 = fi_df.sort_values(by=['feature_importance'], ascending=False)[:10]

        #Define size of bar plot
        plt.figure(figsize=(18,12))
        
        #Plot Searborn bar chart
        #sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        sns.barplot(x='feature_importance', y='feature_names', hue='feature_category', data=fi_top30)

        #Add chart labels
        plt.title(model_type + ' FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')

        return fi_df

    def final_result(self, classify, target, best_fold):
        args = self.args

        # load best parameter
        with open(args.out_path+'\\{}_{}_{}.pkl'.format(classify, target, str(best_fold)), 'rb') as f:
            grid_search_clf = pickle.load(f)

        clf = grid_search_clf

        # save grid searched model
        with open(args.out_path+'\\{}_{}_best_model_fold{}.pkl'.format(classify, target, str(best_fold)), 'wb') as f: 
            pickle.dump(clf, f)

        return clf

    def final_plot(self, clf_list, dataset, cv_trainset, index, target):
        args = self.args
        pred_list = []
        x_train, x_test, y_train, y_test = dataset[0], dataset[1], dataset[2], dataset[3]
        real = y_test.reshape(-1)
        all_result=[]
        param_list =[]
        #clf_name=['XGBoost','CatBoost','LGBMClassifier', 'RandomForestClassifier', 'LogisticRegression']
        clf_name=['XGBoost','CatBoost','LGBMClassifier']

        for i in range(0, len(clf_name)):
            
            x_encoded = clf_list[i].best_estimator_.steps[0][1].fit_resample(x_train, y_train)
            best_model = clf_list[i].best_estimator_.steps[1][1].fit(x_encoded[0], x_encoded[1])
            
            clf = clf_list[i].best_estimator_.steps[1][1]
            best_param =clf_list[i].best_params_
            pred_prob = clf.predict_proba(x_test)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 1]
            pred_list.append(pred_prob)
            param_list.append(best_param)

            
            ## result & roc curve
            mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result = self.accuracy_pipeline(real, pred, pred_prob, clf_name[i], target)
            all_result.append(result) 
            print('{}_mean_accuracy:'.format(clf_name[i]), mean_accuracy, '{}_mean_auc:'.format(clf_name[i]), mean_auc)
            
            
            ## feature importance & shap value
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(x_encoded[0])           
            shap_values = -1*shap_values if str(clf_name[i])!='LGBMClassifier' else shap_values[1]         
            
            plt.figure(figsize=[15,8])
            shap.summary_plot(shap_values, pd.DataFrame(x_encoded[0],columns=index), plot_size=None, max_display=20, show=False, plot_type='bar')
            plt.savefig(args.out_path+'\\{}_{}_shap_value_barplot_all.eps'.format(target, str(clf_name[i])), format='eps')
            
            plt.figure(figsize=[15,8])
            shap.summary_plot(shap_values, pd.DataFrame(x_encoded[0], columns=index), plot_size=None, max_display=20, show=False)
            plt.savefig(args.out_path+'\\{}_{}_shap_value_summaryplot_all.eps'.format(target, str(clf_name[i])), format='eps')
            '''
            
            vals= np.abs(shap_values).mean(0)
            f_i = pd.DataFrame(list(zip(index,vals)),columns=['col_name','feature_importance_vals'])
            f_i.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            
            ## retraining with top 30 feature list
            feature_list = f_i['col_name'][0:20].tolist()
            index_temp = index.tolist()            
            feature_index = [index_temp.index(v) for i, v in enumerate(feature_list) if v in index_temp]
            x_train_rev = x_train[:,feature_index]
            x_test_rev = x_test[:,feature_index]
            
            ## retraining
            x_encoded_rev = clf_list[i].best_estimator_.steps[0][1].fit_resample(x_train_rev, y_train)
            clf_rev = clf_list[i].best_estimator_.steps[1][1].fit(x_encoded_rev[0], x_encoded_rev[1])
         
            pred_prob = clf_rev.predict_proba(x_test_rev)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 0]
            pred_list.append(pred_prob)

                
            ## result & roc curve
            mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result = self.accuracy_pipeline(real, pred, pred_prob, clf_name[i], target)
            all_result.append(result) 
            print('{}_mean_accuracy:'.format(clf_name[i]), mean_accuracy, '{}_mean_auc:'.format(clf_name[i]), mean_auc)
            
            ## feature importance & shap value
            #feature_importance= clf.feature_importances_ if clf_name[i] ==  'RandomForestClassifier' else clf.coef_[0]
            explainer = shap.TreeExplainer(clf_rev)
            shap_values = explainer.shap_values(x_encoded_rev[0])           
            shap_values = -1*shap_values if str(clf_name[i])!='LGBMClassifier' else shap_values[1]         
            plt.figure(figsize=[15,8])
            shap.summary_plot(shap_values, pd.DataFrame(x_encoded_rev[0],columns=feature_list), plot_size=None, max_display=30, show=False, plot_type='bar')
            plt.savefig(args.out_path+'\\{}_{}_shap_value_barplot.eps'.format(target, str(clf_name[i])), format='eps')
            
            plt.figure(figsize=[15,8])
            shap.summary_plot(shap_values, pd.DataFrame(x_encoded_rev[0],columns=feature_list), plot_size=None, max_display=30, show=False)
            plt.savefig(args.out_path+'\\{}_{}_shap_value_summaryplot.eps'.format(target, str(clf_name[i])), format='eps')
            '''
        
        '''
        ## delong test
        xgb_catb_pvalue = self.delong_roc_test(real, pred_list[0], pred_list[1], sample_weight=None)
        xgb_lgbm_pvalue = self.delong_roc_test(real, pred_list[0], pred_list[2], sample_weight=None)
        catb_lgbm_pvalue = self.delong_roc_test(real, pred_list[1], pred_list[2], sample_weight=None)
        xgb_RF_pvalue = self.delong_roc_test(real, pred_list[0], pred_list[3], sample_weight=None)
        xgb_LR_pvalue = self.delong_roc_test(real, pred_list[0], pred_list[4], sample_weight=None)
        lgbm_RF_pvalue = self.delong_roc_test(real, pred_list[2], pred_list[3], sample_weight=None)
        lgbm_LR_pvalue = self.delong_roc_test(real, pred_list[2], pred_list[4], sample_weight=None)
        catb_RF_pvalue = self.delong_roc_test(real, pred_list[1], pred_list[3], sample_weight=None)
        catb_LR_pvalue = self.delong_roc_test(real, pred_list[1], pred_list[4], sample_weight=None)
        RF_LR_pvalue = self.delong_roc_test(real, pred_list[3], pred_list[4], sample_weight=None)

        print('xgb_catb_pvalue: ', 10**xgb_catb_pvalue,' xgb_lgbm_pvalue: ',10**xgb_lgbm_pvalue, ' catb_lgbm_pvalue:',10**catb_lgbm_pvalue, 
            'xgb_RF_pvalue: ', 10**xgb_RF_pvalue,' xgb_LR_pvalue: ',10**xgb_LR_pvalue, ' lgbm_RF_pvalue:',10**lgbm_RF_pvalue, ' lgbm_LR_pvalue:',10**lgbm_LR_pvalue,
       'catb_RF_pvalue: ', 10**catb_RF_pvalue,' catb_LR_pvalue: ',10**catb_LR_pvalue, ' rf_LR_pvalue: ',10**RF_LR_pvalue)
        
        plt.legend()
        plt.savefig(args.save_path+'\\{}_roc_curve.eps'.format(target), format='eps')
        plt.clf()
        '''
        return all_result, pred_list, param_list



## rest
           
    def model_result(self, model, param, group, target):
        args = self.args
        ## drop feature
        drop_list = ['Unnamed: 0','name','vital','sleep_quality','any_cvd','pre_cvd','sleep_apnea','pre_sleep_apnea','cvd_death','censdate','cvd_dthdt','cvd_vital','cvd_date']

        # for only hrv features       
        index = group.drop(drop_list, axis=1).columns
        x = group.drop(drop_list, axis=1).to_numpy()
        y = group['{}'.format(target)].to_numpy()

        ## train/test split and gridsearch and prediction
        pred_list, real_list, prob_list, feature_importance_list, param_list = self.prediction_func_pipeline(x, y, model, target, param, index)
        model = model.__name__
        mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result = self.accuracy_pipeline(real_list, pred_list, prob_list, model, target) 

        return result, feature_importance_list, param_list
    
    def prediction_func_pipeline(self, x, label, classify, target, param, index):

        args = self.args
        
        ## split train/test data and gridsearched with training dataset
        x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.2, stratify = label, random_state=387)
        print('train:',len(x_train), ' test:', len(x_test))
        
        # grid search
        from imblearn.pipeline import make_pipeline
        from sklearn.decomposition import PCA
        cv = StratifiedKFold(5, shuffle=True, random_state=37)
        smote = SMOTE(random_state=37)
        #pca = PCA(n_components=5)
        smp_pipeline = make_pipeline(smote, classify())
        gird_search_clf = GridSearchCV(smp_pipeline, param, scoring='roc_auc', cv=cv, refit=True)
        gird_search_clf.fit(x_train, y_train)

        # save grid searched model
        with open(args.out_path+'\\{}_{}_ahi.pkl'.format(str(classify.__name__), target), 'wb') as f: 
            pickle.dump(gird_search_clf, f)

        print('GridSearch 최고 점수: ', gird_search_clf.best_score_)
        print('GridSearch 최적 파라미터: ', gird_search_clf.best_params_)
        
        # load best estimator for gridsearched with training dataset
        best_param =gird_search_clf.best_params_
        clf = gird_search_clf.best_estimator_.steps[1][1]
       
        pred_list, real_list, prob_list = [], [], []
        pred_prob = clf.predict_proba(x_test)
        pred = pred_prob.argmax(1)
        pred_prob = pred_prob[:, 0]
        pred_list = np.array(pred)
        real_list = np.array(y_test.reshape(-1))
        prob_list = np.array(pred_prob)       

        '''
        ## shap value and feature importance
        feature_importance= clf.feature_importances_ if classify.__name__ != 'LogisticRegression' else clf.coef_[0]
        feature_importance_list = self.plot_feature_importance(feature_importance, index, str(classify.__name__))       
        plt.savefig(args.out_path+'//{}_{}_feature_importance.eps'.format(target, str(classify.__name__)))
        plt.close()
        '''
        feature_importance_list=[]
        return pred_list, real_list, prob_list, feature_importance_list, best_param 
        
    def final_plot_delong(self, clf_list, dataset, cv_trainset, index, target):
        args = self.args
        pred_list = []
        x_train, x_test, y_train, y_test = dataset[0], dataset[1], dataset[2], dataset[3]
        real = y_test.reshape(-1)
        all_result=[]
        clf_name=['XGBoost','CatBoost','LGBMClassifier', 'RandomForestClassifier', 'LogisticRegression']

        # load best models' pred probability
        with open(args.out_path+'\\gridsearched_model\\best_{}_pred_prob.pkl'.format( target), 'rb') as f: 
            pred_prob = pickle.load(f)
        
        mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result = self.accuracy_pipeline(real, pred_prob.argmax(1), pred_prob, 'Best model', target)
        pred_list.append(pred_prob)

        for i in range(0, len(clf_name)):
            clf = clf_list[i].best_estimator_.steps[1][1]
            pred_prob = clf.predict_proba(x_test)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 1]
            mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, result = self.accuracy_pipeline(real, pred, pred_prob, clf_name[0], target)
            pred_list.append(pred_prob)
            pred_list.append(pred_prob)

        
        ## delong test
        best_xgb = self.delong_roc_test(real, pred_list[0], pred_list[1], sample_weight=None)
        best_catb = self.delong_roc_test(real, pred_list[0], pred_list[2], sample_weight=None)
        best_lgbm = self.delong_roc_test(real, pred_list[0], pred_list[3], sample_weight=None)
        best_rf = self.delong_roc_test(real, pred_list[0], pred_list[4], sample_weight=None)
        best_lr = self.delong_roc_test(real, pred_list[0], pred_list[5], sample_weight=None)

        print('best_xgb: ', 10**best_xgb,' best_Catb: ',10**best_catb, ' best_lgbm:',10**best_lgbm, 
            'best_rf: ', 10**best_rf,' best_lr: ',10**best_lr)
        
        plt.legend()
        plt.savefig(args.save_path+'\\result_figure\\eds\\{}_roc_curve.eps'.format(target), format='eps')
        

        return all_result, pred_list, real
## statistics
    from sklearn.metrics import average_precision_score

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
        '''
        (1-tpr) ** 2 + (1-(1-fpr)) ** 2
        ix = np.argmin((1-tpr) ** 2 + (1-(1-fpr)) ** 2)
        
        '''
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

        
    def roc_curve_func(self, pred, real, prob, save_file):
        from sklearn.metrics import RocCurveDisplay
        optimal_threshold, ix, se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci = self.find_optimal_cutoff(real, prob)
        fpr, tpr, thresholds = metrics.roc_curve(real, prob, pos_label=1)
        #plt.clf()
        plt.plot(fpr, tpr, label=save_file)
        #plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        #plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.savefig(save_file, format='eps')

        return se, sp, se_ci, sp_ci, ppv, npv, ppv_ci, npv_ci,optimal_threshold

        
    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, h
        
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
        import numpy as np
        from scipy import stats
        import scipy.stats
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
    def select_mortality_subject(self, df):
        
        ## including eds 
        #quality_index = [id for id, value in data.iterrows() if value['ahi']>=5]
        #df = data.iloc[quality_index,:]
        #df.reset_index(drop=True, inplace=True)

        ## grouping subjective sleep
        alive_index = [id for id, value in df.iterrows() if (value['vital']==1 or (value['vital']==0 and value['censdate']>365*10))]
        df['vital'][alive_index]=1
        alive = df.iloc[alive_index,:]        
        alive.reset_index(drop=True, inplace=True)
        alive['vital']=0

        deceased_index = [id for id, value in df.iterrows() if (value['vital']==0 and value['censdate']<=365*10)]
        deceased = df.iloc[deceased_index,:]
        deceased.reset_index(drop=True, inplace=True)
        deceased['vital']=1
        
        print('alive:', len(alive), '\n deceased:', len(deceased))

        return alive, deceased


    def select_mortality_ahi_subject(self, data):
        
        ## including eds 
        quality_index = [id for id, value in data.iterrows() if value['ahi']>=5]
        df = data.iloc[quality_index,:]
        df.reset_index(drop=True, inplace=True)

        ## grouping subjective sleep
        alive_index = [id for id, value in df.iterrows() if (value['vital']==1 or (value['vital']==0 and value['censdate']>365*15))]
        df['vital'][alive_index]=1
        alive = df.iloc[alive_index,:]        
        alive.reset_index(drop=True, inplace=True)
        alive['vital']=0

        deceased_index = [id for id, value in df.iterrows() if (value['vital']==0 and value['censdate']<=365*15)]
        deceased = df.iloc[deceased_index,:]
        deceased.reset_index(drop=True, inplace=True)
        deceased['vital']=1
        
        print('alive:', len(alive), '\n deceased:', len(deceased))

        return alive, deceased


    def select_cvd_subject(self, df_temp):
       
        ## including ess<=10 
        #quality_index = [id for id, value in data.iterrows() if value['ahi']>=5]
        #df_temp = data.iloc[quality_index,:]
        #df_temp.reset_index(drop=True, inplace=True)

        ## excluding pre-cvd 
        cvd_index = [id for id, value in df_temp.iterrows() if value['pre_cvd']==0]
        df = df_temp.iloc[cvd_index,:]
        df.reset_index(drop=True, inplace=True)        
        
        non_cvd_index = [id for id, value in df.iterrows() if value['any_cvd']==0]
        any_cvd_index = [id for id, value in df.iterrows() if value['any_cvd']!=0]
        df['any_cvd'][non_cvd_index]=0
        df['any_cvd'][any_cvd_index]=1
        
        ## any cvd subject
        non_cvd_index = [id for id, value in df.iterrows() if value['any_cvd']==0]
        non_cvd = df.iloc[non_cvd_index,:]
        non_cvd.reset_index(drop=True, inplace=True)

        any_cvd_index = [id for id, value in df.iterrows() if value['any_cvd']==1]
        any_cvd = df.iloc[any_cvd_index,:]
        any_cvd.reset_index(drop=True, inplace=True)

        print('any_cvd:', len(any_cvd), '\n non_cvd:', len(non_cvd))
        return non_cvd, any_cvd

    def select_cvd_ahi_subject(self, data):
       
        ## including ess<=10 
        quality_index = [id for id, value in data.iterrows() if value['ahi']>=5]
        df_temp = data.iloc[quality_index,:]
        df_temp.reset_index(drop=True, inplace=True)

        ## excluding pre-cvd 
        cvd_index = [id for id, value in df_temp.iterrows() if value['pre_cvd']==0]
        df = df_temp.iloc[cvd_index,:]
        df.reset_index(drop=True, inplace=True)        
        
        non_cvd_index = [id for id, value in df.iterrows() if value['any_cvd']==0]
        any_cvd_index = [id for id, value in df.iterrows() if value['any_cvd']!=0]
        df['any_cvd'][non_cvd_index]=0
        df['any_cvd'][any_cvd_index]=1
        
        ## any cvd subject
        non_cvd_index = [id for id, value in df.iterrows() if value['any_cvd']==0]
        non_cvd = df.iloc[non_cvd_index,:]
        non_cvd.reset_index(drop=True, inplace=True)

        any_cvd_index = [id for id, value in df.iterrows() if value['any_cvd']==1 and value['cvd_date']!=0]
        any_cvd = df.iloc[any_cvd_index,:]
        any_cvd.reset_index(drop=True, inplace=True)

        print('any_cvd:', len(any_cvd), '\n non_cvd:', len(non_cvd))
        return non_cvd, any_cvd


    def select_cvd_vital_subject(self, df):

        quality_index = [id for id, value in df.iterrows() if value['pre_cvd']==0]
        df = df.iloc[quality_index,:]
        df.reset_index(drop=True, inplace=True)

        
        ## grouping subjective sleep
        alive_index = [id for id, value in df.iterrows() if value['cvd_vital']==1]
        alive = df.iloc[alive_index,:]
        alive.reset_index(drop=True, inplace=True)

        deceased_temp_index = [id for id, value in df.iterrows() if value['cvd_vital']==0]        
        deceased = df.iloc[deceased_temp_index,:]
        deceased.reset_index(drop=True, inplace=True)

        print('alive:', len(alive), '\n deceased:', len(deceased))

        return alive, deceased

if __name__ == '__main__':


    arguments = get_args()
    path = arguments.feature_path
    total_feature = pd.read_excel(path)
    tree_shhs= tree(arguments)
    
    ## classification -mortality subject (group1: good (1) / group2: bad (0))
    alive, deceased = tree_shhs.select_mortality_ahi_subject(total_feature)
    vital_group = pd.concat([alive, deceased])
    #tree_shhs.performance_result('vital', vital_group)
    #tree_shhs.save_concat_cv_result('vital', vital_group)
    tree_shhs.save_best_result('vital', vital_group)
    #tree_shhs.km_plot('vital',vital_group)
    #tree_shhs.cox_func('vital',vital_group)
    
    ## classification -mortality subject (group1: good (1) / group2: bad (0))
    #alive, deceased = tree_shhs.select_cvd_ahi_subject(total_feature)
    #vital_group = pd.concat([alive, deceased])
    #tree_shhs.performance_result('any_cvd', vital_group)
    #tree_shhs.save_concat_cv_result('any_cvd', vital_group)
    #tree_shhs.save_best_result('any_cvd', vital_group)
    #tree_shhs.km_plot('any_cvd',vital_group)
    #tree_shhs.cox_func('any_cvd',vital_group)

    ## classification -mortality subject (group1: good (1) / group2: bad (0))
    #alive, deceased = tree_shhs.select_mortality_ahi_subject(total_feature)
    #vital_group = pd.concat([alive, deceased])
    #tree_shhs.performance_result('vital', vital_group)
    #tree_shhs.save_final_cv_result('vital', vital_group)
    #tree_shhs.km_plot('vital',vital_group)


    ## cvd subject
    #non_cvd, cvd = tree_shhs.select_cvd_subject(total_feature)
    #cvd_group = pd.concat([non_cvd, cvd])
    #tree_shhs.performance_result('any_cvd', cvd_group)
    #tree_shhs.save_final_result('any_cvd', cvd_group)
    #tree_shhs.km_plot('any_cvd', cvd_group)
    
    '''    
    ## cvd & vital subject
    non_cvd, cvd = tree_shhs.select_cvd_vital_subject(total_feature)
    cvd_group = pd.concat([non_cvd, cvd])
    tree_shhs.performance_result('cvd_vital', cvd_group)
    #tree_shhs.save_final_result('cvd_vital', cvd_group)
    ##tree_shhs.km_plot('any_cvd', cvd_group)
    '''
    
