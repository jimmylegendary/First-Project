import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

from output.io_module import IOModule
from input.model_input import ModelInput

class MyTrainResult:
    """예측결과, 실제 정답, 예측 확률값, feature importance 결과, gridsearch 결과 best parameter 값"""
    def __init__(self, preds: np.ndarray, reals: np.ndarray, probs: np.ndarray, feature_imporatnces: list, best_params: list):
        self.preds = preds 
        self.reals = reals 
        self.probs = probs 
        self.feature_imporatnces = feature_imporatnces 
        self.best_params = best_params 
        
class GSCV:
    def __init__(self, io_module):
        
        self.hyperparameters = None
        self.classifier = None
        self.train_result = {}
        self.dataset = None
        self.io_module: IOModule = io_module
        
    
    def train(self, classifier_name: str, model_input: ModelInput):
        """학습 모델 및 결과 데이터셋 저장 (train)
        + self.result[classifier_name] = result
        
        Args:
            classifier_name (str, optional): classifier name (e.g. xgboost, lgbm, catboost). Defaults to "".

        """
        # initialize output (return값 list로 initialize)
        pred_list, real_list, prob_list = [], [], []
        feature_importance_list = []
        best_param_list = []
        
        ## kfold prediction (5 stratified fold cross-validation 수행 + smote를 통해 data augmentation = make_pipeline 함수 사용)
        smote = SMOTE(random_state=37)
        smp_pipeline = make_pipeline(StandardScaler(), smote, model_input.classifier)
        
        # prepare_dataset에서 저장한 train, test dataset 불러오기
        for kfold, dataset in enumerate(model_input.dataset.folded_dataset):
            x_train, y_train, x_test, y_test = (
                dataset["train"]["x"], 
                dataset["train"]["y"],
                dataset["test"]["x"], 
                dataset["test"]["y"]
                )
            
            ## cross-validation에 맞춰 training (gridsearch)
            gird_search_clf = GridSearchCV(
                smp_pipeline, model_input.hyperparameter, scoring='roc_auc', cv=model_input.cv, refit=True
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
            # save grid searched model (prediction 결과, predicted probability, 실제 label 값(정답), feature(x 값))
            save_dataset = [
                pred,
                pred_prob, 
                y_test.reshape(-1), 
                pd.DataFrame(x_test, columns = model_input.dataset.feature_name)
                ]
            self.io_module.to_picklefile(filename=f"{str(kfold)}_dataset.pkl", data=save_dataset, classifier_name=classifier_name)
             
        
        self.train_result[classifier_name] = MyTrainResult(
            preds=np.array(pred_list),
            reals=np.array(real_list),
            probs=np.array(prob_list),
            feature_imporatnces=feature_importance_list,
            best_params=best_param_list,
        )

    
    def test(self):
        pass