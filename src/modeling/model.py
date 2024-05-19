import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import argparse
from statistics import mean
import itertools

from catboost import CatBoostClassifier as catb
from lightgbm import LGBMClassifier as lgb
from xgboost import XGBClassifier as xgb
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from scipy import interp
import scipy.stats 
from sklearn.metrics import (
    roc_curve, 
    confusion_matrix, 
    accuracy_score, 
    auc as auc_func
)
    
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA

from math import sqrt
from scipy.special import ndtri

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import make_pipeline
from lifelines import KaplanMeierFitter as KM
from lifelines import CoxPHFitter as CPH
from lifelines.statistics import logrank_test


def get_args():
    """인자값들 저장해두는 Namespace"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default = 'vital', type=str) # required=True, choices = ['vital','any_cvd']
    parser.add_argument('--feature_path', default='src\dataset\data.xlsx', type=str)
    parser.add_argument('--out_path', default='output\\15', type=str)
    parser.add_argument('--plot_feature_path', default='src\dataset\plot_data.xlsx', type=str)    

    return parser.parse_args()

class MyDataset: 
    """data column명, 학습용 data, 실제 정답"""
    feature_name: list
    x: np.ndarray
    y: np.ndarray

class MyTrainResult:
    """예측결과, 실제 정답, 예측 확률값, feature importance 결과, gridsearch 결과 best parameter 값"""
    def __init__(self):
        self.preds: np.ndarray
        self.reals: np.ndarray
        self.probs: np.ndarray
        self.feature_imporatnces: list
        self.best_params: list

class tree(object):
    def __init__(self, args):
        self.args = args
        self.outcome = args.target
        
        self.dataset = MyDataset  #선언이 안됨
        

        self.target_func = {
            "vital": self.select_mortality_ahi_subject,
            "any_cvd": self.select_cvd_ahi_subject
        }

        self.classifier = {
            "xgb": xgb(),
            "lgb": lgb(),
            "catb": catb(),
            "rf": rf(),
            "lr": lr()
        }

        self.best_fold = {
            "xgb": int,
            "lgb": int,
            "catb": int,
            "rf":int,
            "lr": int        
            }

        self.feature_importance_cat = {}

        self.train_result : dict[str, MyTrainResult] = {}

        self.hyperparameters = {
            
            "xgb" : {
                'xgbclassifier__max_depth': [3,5,7,9],
                'xgbclassifier__n_estimators': [100, 200, 300, 400, 500],
                'xgbclassifier__learning_rate': [0.01],
                'xgbclassifier__min_child_weight' : [1, 3, 5],
                'xgbclassifier__objective': ['binary:logistic'],
                'xgbclassifier__use_label_encoder': ['False'],
                'xgbclassifier__random_state':[22]
                },
            "lgb" : {
                'lgbmclassifier__max_depth': [2,4,6,8],
                'lgbmclassifier__n_estimators': [100,200,300,400,500],
                'lgbmclassifier__learning_rate': [0.01],
                'lgbmclassifier__min_data_in_leaf': [10, 20, 30, 40, 50],            
                'lgbmclassifier__objective': ['binary'],
                'lgbmclassifier__random_state':[22]        
                },
            
            "catb" : {
                'catboostclassifier__iterations': [100,200,300,400,500],
                'catboostclassifier__depth': [3,5,7,9],
                'catboostclassifier__learning_rate': [0.01],
                'catboostclassifier__l2_leaf_reg': [3,5,7],
                'catboostclassifier__loss_function': ['Logloss'],
                'catboostclassifier__random_state':[22]
                },
            "rf": {
                'randomforestclassifier__max_depth': [2,4,6,8],
                'randomforestclassifier__min_samples_leaf': [1,5,10,20],
                'randomforestclassifier__n_estimators': [100,200,300,400,500],
                'randomforestclassifier__random_state':[22]
                },
            "lr": {
                'logisticregression__C': [0.0011],
                'logisticregression__random_state':[22]
            }
        }

        self.drop_list =[
            'Unnamed: 0',
            'name',
            'vital',
            'sleep_quality',
            'any_cvd',
            'pre_cvd',
            'sleep_apnea',
            'pre_sleep_apnea',
            'cvd_death',
            'censdate',
            'cvd_dthdt',
            'cvd_vital',
            'cvd_date',
            'waso_c'
            ]

        self.root_outdir = os.path.join(args.out_path, self.outcome)
        self.set_outdir(
            root_dir = self.root_outdir,
            classifier_name = self.classifier.keys()
            )
        

    ## data save 지정
    def set_outdir(self, root_dir, classifier_name):
        """ out directory 만들기 (save&load)

        Args:
            root_dir (str): output root directory
            classifier_name (str): classifier name (e.g. xgboost, lgbm, catboost)
        """
        dir_names = []
        for clf_name in classifier_name:
            dir_name = os.path.join(root_dir, clf_name)
            os.makedirs(dir_name, exist_ok=True)
            dir_names.append(dir_name)
    
    def get_outpath(self, file_name, classifier_name =""):
        """ 최종 output path 받아오기 (save&load)

        Args:
            file_name (str): 저장 하고자 하는 file 이름의 앞부분
            classifier_name (str, optional): classifier name (e.g. xgboost, lgbm, catboost). Defaults to "".
        """
        return os.path.join(self.root_outdir, classifier_name, file_name)
     
    def save_pkl(self, path, data):
        """
        path에 data를 pkl로 저장 (save&load)
        """
        
        with open(path, 'wb') as f:
                pickle.dump(data, f)


    ## data preparation
    def extract_features(self, total_data: pd.DataFrame):
        """ Target outcome에 따라 grouping된 feature 값을 반환하는 함수 (dataset)
        + grouping_data 에 concated_group 결과 저장
        + outcome에 target outcome 저장

        Args:
            total_data: 전체 feature data (DataFrame) 
        """
        group_0, group_1 = self.target_func[self.outcome](total_data)
        concated_group = pd.concat([group_0, group_1])
        
        # grouping data / outcome 저장
        self.grouping_data = concated_group

    def prepare_dataset(self):                
        """ x, y를 fold 별 train, test로 나누어 dataset을 준비하는 함수 (dataset)
        + cv = cross-validation 방법 저장
        + folded_dataset = fold별 train, test dataset 저장
        + dataset.featuren_name = feature column 이름 저장
        + dataset.x = 학습에 사용될 feature (drop_list 제외된 버전)
        + dataset.y = 결과지로 사용될 outcome 결과
        """   
        cv = StratifiedKFold(5, shuffle = True, random_state = 10)
        self.cv = cv
        self.folded_dataset = []
        
        # dadtaset에 feature_name, x, y 저장 ->'tree' object has no attribute 'dataset' error
        self.dataset.feature_name = self.grouping_data.drop(self.drop_list, axis = 1).columns.to_list() 
        self.dataset.x = self.grouping_data.drop(self.drop_list, axis = 1).to_numpy()
        self.dataset.y = self.grouping_data[f'{self.outcome}'].to_numpy()

        x = self.dataset.x
        y = self.dataset.y

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
            # fold별 train, test data 저장
            self.folded_dataset.append(dataset)

    def select_mortality_ahi_subject(self, total_data: pd.DataFrame):
        """ (dataset)
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
        alive: pd.DataFrame = df_ahi[df_ahi['vital'] == 1 | ((df_ahi['vital'] == 0) & (df_ahi['censdate'] > 365*15))].reset_index(drop=True)
        alive['vital'] = 1
        deceased: pd.DataFrame = df_ahi[(df_ahi['vital'] == 0) & (df_ahi['censdate'] <= 365*15)].reset_index(drop=True)
        deceased['vital'] = 0
        
        print('alive:', len(alive), '\n deceased:', len(deceased))

        return alive, deceased

    def select_cvd_ahi_subject(self, total_data: pd.DataFrame):
        """(dataset)
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

        any_cvd: pd.DataFrame = df_pre_cvd[df_pre_cvd['any_cvd']!=0 & df_pre_cvd['cvd_date']!=0].reset_index(drop=True)
        any_cvd[:,'any_cvd'] = 1

        print('any_cvd:', len(any_cvd), '\n non_cvd:', len(non_cvd))
        return non_cvd, any_cvd


    ## train               
    def train(self, classifier_name: str):
        """학습 모델 및 결과 데이터셋 저장 (train)
        + self.result[classifier_name] = result
        
        Args:
            classifier_name (str, optional): classifier name (e.g. xgboost, lgbm, catboost). Defaults to "".

        """
        train_result = MyTrainResult()

        # initialize output (return값 list로 initialize)
        pred_list, real_list, prob_list = [], [], []
        feature_importance_list = []
        best_param_list = []
        hyperparameter = self.hyperparameters[classifier_name]
        
        ## kfold prediction (5 stratified fold cross-validation 수행 + smote를 통해 data augmentation = make_pipeline 함수 사용)
        smote = SMOTE(random_state=37)
        smp_pipeline = make_pipeline(StandardScaler(), smote, self.classifier[classifier_name])
        
        # prepare_dataset에서 저장한 train, test dataset 불러오기
        for kfold, dataset in enumerate(self.folded_dataset):
            x_train, y_train, x_test, y_test = (
                dataset["train"]["x"], 
                dataset["train"]["y"],
                dataset["test"]["x"], 
                dataset["test"]["y"]
                )
            
            ## cross-validation에 맞춰 training (gridsearch)
            gird_search_clf = GridSearchCV(
                smp_pipeline, hyperparameter, scoring='roc_auc', cv=self.cv, refit=True
            )
            gird_search_clf.fit(x_train, y_train)
            
            # grid search 결과 best parameter list에 저장
            best_param_list.append(gird_search_clf.best_params_)
            print('GridSearch 최고 점수: ', gird_search_clf.best_score_)
            print('GridSearch 최적 파라미터: ', gird_search_clf.best_params_)

            # gird searched model 중 가장 best estimator의 학습 결과 저장
            clf = gird_search_clf.best_estimator_.steps[2][1]
            pred_prob = clf.predict_proba(x_test)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 1]
                        
            pred_list.append(pred)
            real_list.append(y_test.reshape(-1))
            prob_list.append(pred_prob)

            ## feature importance (logistic regression 제외 나머지 classifier의 feature importance 내장 함수 이용하여 feature importance 저장)
            feature_importance= (
            clf.feature_importances_ if classifier_name != 'lr' else clf.coef_[0]
            )
            feature_importance_list.append(feature_importance)
                               
            ## save classifier & fold dataset
            # grid searched model 저장 (pkl로 각 grid search 학습결과 저장)
            self.save_pkl(self.get_outpath(f"{str(kfold)}_classifier.pkl", classifier_name), gird_search_clf)
            
            # save grid searched model (prediction 결과, predicted probability, 실제 label 값(정답), feature(x 값))
            save_dataset = [
                pred,
                pred_prob, 
                y_test.reshape(-1), 
                pd.DataFrame(x_test, columns = self.dataset.feature_name)
                ]
            self.save_pkl(self.get_outpath(f"{str(kfold)}_dataset.pkl", classifier_name), save_dataset)
            
        train_result.preds = np.array(pred_list)
        train_result.reals = np.array(real_list)
        train_result.probs = np.array(prob_list)
        train_result.feature_imporatnces = feature_importance_list
        train_result.best_params = best_param_list
        
        self.train_result[classifier_name] = train_result
        self.save_pkl(self.get_outpath("train_result.pkl"), train_result)

    def classifier_train(self):
        """ (train)
        Target outcome에 따라 classifier별 train 함수 돌리기 
        """
        for classifier_name in self.classifier.keys():
            self.train(classifier_name)


    ## result 결과 정리 + save
    def save_performance_result(self):
        """ (result)
        Target outcome에 따라 classifier별 분류 성능을 excel로 저장하는 

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
        """          
        ## classifier result (classifier 별로 clasification result, feautre importnace, hyperparmeter 저장)
        clf_result = {}
        clf_feature = pd.DataFrame()
        clf_param = []

        for classifier_name in self.classifier.keys():
            result = self.get_perform_metric(classifier_name)
            best_param = self.get_best_param(classifier_name)
            feature = self.get_feature_importance(classifier_name)

            clf_result.update(result)
            clf_feature = pd.concat([clf_feature, feature], axis=1)
            clf_param.append(best_param)
            
        ## dict to DataFrame (excel저장을 위해 dataframe으로 변경)
        tree_result = pd.DataFrame(clf_result)
        tree_feature = clf_feature
        tree_param = pd.DataFrame(clf_param)

        ## result save (classifcation result, feautre importnace, hyperparmeter를 excel로 저장)
        writer = pd.ExcelWriter(
            self.get_outpath("performanceresult_featureimportance_bestparameter.xlsx")
            )      
        tree_result.to_excel(writer, sheet_name = f"{self.outcome}_result")
        tree_feature.to_excel(writer, sheet_name = f"{self.outcome}_feature")
        tree_param.to_excel(writer, sheet_name = f"{self.outcome}_param")
        writer.close()


    ## TRAIN 결과 계산 함수
    def get_best_param(self, classifier_name: str) -> list:
        """ (result)
        1)perform_metric: performance (fold별 결과, 평균값), 2) best parameter 결과 

        Args:
            classifier_name: xgb or catb or lgb or rf or lr (str)
           
        Returns:
            clf.best_params: hyperparmeter 결과 (list)
        """

        ## save performance result with cross-validation : cross-validation 결과 저장        
        best_fold = self.best_fold[classifier_name]
        
        ## load and save best model: 성능 가장 좋았던 fold 확인해서 classifier 불러오기 & best_model_ 로 따로 저장해주기
        with open(self.get_outpath(f"{str(best_fold)}.pkl", classifier_name), 'rb') as f:
            grid_search_clf = pickle.load(f)
        clf = grid_search_clf
        
        self.save_pkl(self.get_outpath(f"best_model_fold{str(best_fold)}.pkl", classifier_name), clf)

        return clf.best_params_
    
    def get_feature_importance(self, classifier_name: str) -> pd.DataFrame:
        """ (result)
        get mean feature importance with categorical index
        +) feature importance category
        Args:
            classifier_name: xgb or catb or lgb or rf or lr (str)
           
        Returns:
            feature_important_list:  feature 이름/ feature importance / feature의 종류에 따라 저장한 값 (DataFrame)
        """
        
        ## averaging feature importance from fold : fold별 획득된 feature importance값의 평균값을 구하기
        feature_importance = self.train_result[classifier_name].feature_imporatnces
        feature_mean = np.sum(feature_importance, axis=0) / len(feature_importance)

        #Create arrays from feature importance and feature names
        feature_mean_nparray = np.array(feature_mean)
        feature_names = np.array(self.dataset.feature_name, dtype=object)
        feature_category= []
        for _ in range(12): 
            feature_category.append('Demographics') 
        for _ in range(17): 
            feature_category.append('Sleep features')
        for _ in range(27): 
            feature_category.append('HRV')  

        #Create a DataFrame using a Dictionary
        data={
            'feature_names':feature_names,
            'feature_importance':feature_mean_nparray, 
            'feature_category':feature_category
            }
        feature_importance_category = pd.DataFrame(data)

        self.feature_importance_cat[classifier_name] = feature_importance_category

        return feature_importance_category

    def get_perform_metric(self, classifier_name: str) :
        """(result)
        cross-validation 결과를 저장 + 가장 성능이 좋은 fold, 평균 accuracy, sensitivity, specificity, ppv, npv, auc, fold별 각 성능값을 return하는 함수 
        +) best_fold: 가장 성능이 좋은 fold (int)
        +) roc_plot_val: roc curve plotting용 fpr, tpr값 저장

        Args:
            classifier_name: classifier 이름 (str)

        Returns:
            result: fold별 성능값 (dict)
        """
        fold = []

        real = self.train_result[classifier_name].reals
        pred = self.train_result[classifier_name].preds
        prob = self.train_result[classifier_name].probs

        metrics = {
            'accuracy' : [],
            'roc' : [],
            'se' : [],
            'sp' : [],
            'ppv' : [],
            'npv' : [] 
        }
        
        # roc curve plotting용 변수 저장
        mean_fpr = np.linspace(0,1,100)
        tprs = []
        self.roc_plot_val = {}

        for k in range(0, real.shape[0]):
            accuracy = accuracy_score(real[k], pred[k])
            fpr, tpr, _ = roc_curve(real[k], prob[k], pos_label=1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            auc = auc_func(fpr, tpr)
            _, _, se, sp, _, _, ppv, npv, _, _ = self.find_optimal_cutoff(real[k], prob[k])
                        
            metrics['accuracy'].append(accuracy)
            metrics['se'].append(se)
            metrics['sp'].append(sp)
            metrics['ppv'].append(ppv)
            metrics['npv'].append(npv)
            metrics['roc'].append(auc)

            fold.append(k)

        # 평균값 계산
        # roc curve plotting용 저장
        mean_tpr = np.mean(tprs, axis = 0)
        self.roc_plot_val[classifier_name] = [mean_fpr, mean_tpr]

        for k, v in metrics.items():
            mean_val = np.round(np.mean(v), 4)
            v.append(mean_val)
        fold.append('mean')
        
        # roc 기준 best fold 저장
        best_fold = metrics['roc'].index(max(metrics['roc']))
        self.best_fold[classifier_name] = best_fold
        
        performance_result = {
            f"accuracy_{classifier_name}": metrics['accuracy'], 
            f"roc_{classifier_name}": metrics['roc'], 
            f"se_{classifier_name}": metrics['se'], 
            f"sp_{classifier_name}": metrics['sp'], 
            f"ppv_{classifier_name}": metrics['ppv'], 
            f"npv_{classifier_name}": metrics['npv'] 
            }

        return performance_result  


## plotting  
    def plot_classifier(self):
        """(plotting)
        classifier별로 plotting하고 싶은 함수 부르는 곳
        """

        for clf_name in self.classifier.keys():
            self.plot_feature_importance(clf_name)
            self.plot_roc_curve(clf_name)

    def plot_feature_importance(self, classifier_name: str):
        """(plotting)
        feature importance 값을 feature의 종류(demographic, sleep feature, HRV)에 따라 
        1) bar plotting하고

        Args:
            classifier_name:  classifier의 이름 (Str)
        """
        # get the feature importance category
        feature_importance_category = self.get_feature_importance(classifier_name) 

        #Sort the DataFrame in order decreasing feature importance
        fi_top30 = feature_importance_category.sort_values(by=['feature_importance'], ascending=False)[:30]

        #Define size of bar plot
        plt.figure(figsize = (18,12))
        
        #Plot Searborn bar chart
        sns.barplot(x='feature_importance', y='feature_names', hue='feature_category', data=fi_top30)

        #Add chart labels
        plt.title(classifier_name + ' FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')

        ## plot & save bar graph of feature importance : fold의 평균 feature importance 값 plotting
        plt.savefig(self.get_outpath("feature_importance.eps", classifier_name))
        plt.close()     
    
    def plot_roc_curve(self, classifier_name: str):
        """(plotting)
        roc curve를 plotting함 함수 
        
        Args:
            real: 실제 outcome label =정답 (ndarray)
            prob: predicted probability =예측한 확률값 (ndarray)
            classifier_name: classifier name (str)

        """

        fpr =  self.roc_plot_val[classifier_name][0]
        tpr = self.roc_plot_val[classifier_name][1]
            
        plt.clf()
        figure_legend = f'{classifier_name}'
        plt.plot(fpr, tpr, label = figure_legend)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.savefig(self.get_outpath("roc_curve_figure.svg", classifier_name))
           
    def plot_km(self):
        """ (plotting)
        kaplan-meier estimates by high and low risk figure plotting함 함수 
        """ 
        args = self.args
        
        ## km plot (특정 dataset - probability 결과)
        dataset = pd.read_excel(args.plot_feature_path, sheet_name='survival_10')   
        risk_ix = dataset['risk_group'] == 1

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
        plt.savefig(self.get_outpath("km_plot_10.eps"))
        plt.clf()

    def plot_cox_func(self):
        """
        Cox Proportional Hazard model의 figure plotting함 함수 
        """   
        args = self.args
        outcome = self.outcome
        
        ## km plot (특정 dataset)
        dataset = pd.read_excel(args.plot_feature_path, sheet_name='survival_10')   
        data = dataset[['vital_date','vital','age','hypertension','diabetes','risk_group']]
        
        plt.figure(figsize = (8, 4))
        cph = CPH()
        cph.fit(data, 'vital_date', event_col = 'vital')
        cph.print_summary()
        cph.plot()

        ## Create an estimate
        plt.savefig(self.get_outpath("cox_HR_10_3.eps"))
        plt.clf()
    
    
## load final result    
    def load_classifier(self):
        """load classifier (save&load)
        """
        for classifier_name in self.classifier.keys():
            self.load_all_pkl(classifier_name)               

    def load_all_pkl(self, classifier_name: str):
        """저장된 모든 pkl(학습 완료된 classifier와 folded dataset) 불러오기

        Args:
            classifier_name (str): classifier name
        """
        
        train_result = MyTrainResult()

        # initialize output (return값 list로 initialize)
        pred_list, real_list, prob_list = [], [], []
        feature_importance_list = []
        best_param_list = []

        # load dataset
        for kfold, dataset in enumerate(self.folded_dataset):
            x_test, y_test = (
                dataset["test"]["x"], 
                dataset["test"]["y"]
                )
            
            with open(self.get_outpath(f"{str(kfold)}.pkl", classifier_name), 'rb') as f: 
                gird_search_clf = pickle.load(f)

            # grid search 결과 best parameter list에 저장
            best_param_list.append(gird_search_clf.best_params_)
            print('GridSearch 최고 점수: ', gird_search_clf.best_score_)
            print('GridSearch 최적 파라미터: ', gird_search_clf.best_params_)

            # gird searched model 중 가장 best estimator의 학습 결과 저장
            clf = gird_search_clf.best_estimator_.steps[2][1]
            pred_prob = clf.predict_proba(x_test)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 1]
                        
            pred_list.append(pred)
            real_list.append(y_test.reshape(-1))
            prob_list.append(pred_prob)

            ## feature importance (logistic regression 제외 나머지 classifier의 feature importance 내장 함수 이용하여 feature importance 저장)
            feature_importance= (
            clf.feature_importances_ if classifier_name != 'lr' else clf.coef_[0]
            )
            feature_importance_list.append(feature_importance)         

        train_result.preds = np.array(pred_list)
        train_result.reals = np.array(real_list)
        train_result.probs = np.array(prob_list)
        train_result.feature_imporatnces = feature_importance_list
        train_result.best_params = best_param_list

        self.train_result[classifier_name] = train_result

            
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
      
    def compute_midrank(self, x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float32)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float32)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1

        return T2

    def compute_midrank_weight(self, x, sample_weight):
        J = np.argsort(x)
        Z = x[J]
        cumulative_weight = np.cumsum(sample_weight[J])
        N = len(x)
        T = np.zeros(N, dtype=np.float32)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = cumulative_weight[i:j].mean()
            i = j
        T2 = np.empty(N, dtype=np.float32)
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

        tx = np.empty([k, m], dtype=np.float32)
        ty = np.empty([k, n], dtype=np.float32)
        tz = np.empty([k, m + n], dtype=np.float32)
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

        tx = np.empty([k, m], dtype=np.float32)
        ty = np.empty([k, n], dtype=np.float32)
        tz = np.empty([k, m + n], dtype=np.float32)
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

    def cal_delong_test(self):
        """ (statistics)
        calculate delong test
        +) delong_pvalue : {clf1_clf2 : delong_pvalue} (저장 필요)
        """
        args = self.args
        # delong_pvalue 저장하기
        delong_test_result = {} #dict(str, int)

        for clf_1, clf_2 in itertools.combinations(self.classifier.keys(), 2):
            real = self.train_result[clf_1].reals[0]
            clf_1_prob = self.train_result[clf_1].probs[0]
            clf_2_prob = self.train_result[clf_2].probs[0]
            delong_pvalue = self.delong_roc_test(real, clf_1_prob, clf_2_prob, sample_weight= None)
            delong_test_result[f'{clf_1}_{clf_2}'] = delong_pvalue
        delong_pvalue_result = pd.DataFrame(delong_test_result)

        ## result save (delong pvalue result를 excel로 저장)
        writer = pd.ExcelWriter(args.out_path + '\\delong_pvalue_result.xlsx')      
        delong_pvalue_result.to_excel(writer)
        writer.close()

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
        lower_upper_ci = scipy.stats.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)

        lower_upper_ci[lower_upper_ci > 1] = 1

        return auc, auc_var, lower_upper_ci



if __name__ == '__main__':
    args = get_args()
    total_feature = pd.read_excel(args.feature_path)
    tree_shhs= tree(args)

    # data prepare
    tree_shhs.extract_features(total_feature)
    tree_shhs.prepare_dataset()

    # train and save model
    tree_shhs.classifier_train()
    #tree_shhs.load_classifier()
    tree_shhs.save_performance_result()

    # 통계 + plotting (독립적으로 실행 가능)
    tree_shhs.cal_delong_test()
    #tree_shhs.plot_classifier()
    #tree_shhs.plot_km()
    #tree_shhs.plot_cox_func()
  
    