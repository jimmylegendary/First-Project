from model.gscv import MyTrainResult
from output.io_module import IOModule
from sklearn.metrics import roc_curve, accuracy_score, auc as auc_func
from scipy import interp
import numpy as np
import pandas as pd
import itertools
import postprocessing.my_statistics as mystatistics


class Performance:
    def __init__(self, io_module, classifier_name):
        self.io_module: IOModule = io_module
        self.best_fold = {}
        self.classifier_name = classifier_name

    def get_performance(self, classifier_name, feature_name, train_result: MyTrainResult):
        # clf_result, clf_feature, clf_param = self.get_metric_param_importance()
        result = self.get_perform_metric(classifier_name, train_result)
        best_param = self.get_best_param(classifier_name)
        feature = self.get_feature_importance(feature_name, train_result.feature_imporatnces)

        return result, best_param, feature

    def get_best_param(self, classifier_name: str) -> list:
        """(result)
        1)perform_metric: performance (fold별 결과, 평균값), 2) best parameter 결과
        Args:
            classifier_name: xgb or catb or lgb or rf or lr (str)
        Returns:
            clf.best_params: hyperparmeter 결과 (list)
        """

        ## save performance result with cross-validation : cross-validation 결과 저장
        best_fold = self.best_fold[classifier_name]

        ## FIXED: load and save best model: 성능 가장 좋았던 fold 확인해서 classifier 불러오기 & best_model_ 로 따로 저장해주기

        grid_search_clf = self.io_module.from_picklefile(
            filename=f"{str(best_fold)}.pkl", classifier_name=classifier_name
        )

        self.io_module.to_picklefile(
            filename=f"best_model_fold{str(best_fold)}.pkl",
            data=grid_search_clf,
            classifier_name=classifier_name,
        )

        return grid_search_clf.best_params_

    def get_feature_importance(self, feature_name, feature_importance) -> pd.DataFrame:
        """(result)
        get mean feature importance with categorical index
        +) feature importance category
        Args:
            classifier_name: xgb or catb or lgb or rf or lr (str)

        Returns:
            feature_important_list:  feature 이름/ feature importance / feature의 종류에 따라 저장한 값 (DataFrame)
        """

        ## averaging feature importance from fold : fold별 획득된 feature importance값의 평균값을 구하기
        feature_mean = np.sum(feature_importance, axis=0) / len(feature_importance)

        # Create arrays from feature importance and feature names
        feature_mean_nparray = np.array(feature_mean)
        feature_names = np.array(feature_name, dtype=object)
        feature_category = []
        for _ in range(12):
            feature_category.append("Demographics")
        for _ in range(17):
            feature_category.append("Sleep features")
        for _ in range(27):
            feature_category.append("HRV")

        # Create a DataFrame using a Dictionary
        data = {
            "feature_names": feature_names,
            "feature_importance": feature_mean_nparray,
            "feature_category": feature_category,
        }
        feature_importance_category = pd.DataFrame(data)

        return feature_importance_category

    def get_perform_metric(self, classifier_name: str, train_result: MyTrainResult):
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

        real = train_result.reals
        pred = train_result.preds
        prob = train_result.probs

        metrics = {"accuracy": [], "roc": [], "se": [], "sp": [], "ppv": [], "npv": []}

        # roc curve plotting용 변수 저장
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        self.roc_plot_val = {}

        for k in range(0, real.shape[0]):
            accuracy = accuracy_score(real[k], pred[k])
            fpr, tpr, _ = roc_curve(real[k], prob[k], pos_label=1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            auc = auc_func(fpr, tpr)
            _, _, se, sp, _, _, ppv, npv, _, _ = mystatistics.find_optimal_cutoff(real[k], prob[k])

            metrics["accuracy"].append(accuracy)
            metrics["se"].append(se)
            metrics["sp"].append(sp)
            metrics["ppv"].append(ppv)
            metrics["npv"].append(npv)
            metrics["roc"].append(auc)

            fold.append(k)

        # 평균값 계산
        # roc curve plotting용 저장
        mean_tpr = np.mean(tprs, axis=0)
        self.roc_plot_val[classifier_name] = [mean_fpr, mean_tpr]

        for k, v in metrics.items():
            mean_val = np.round(np.mean(v), 4)
            v.append(mean_val)
        fold.append("mean")

        # roc 기준 best fold 저장
        best_fold = metrics["roc"].index(max(metrics["roc"]))
        self.best_fold[classifier_name] = best_fold

        performance_result = {
            f"accuracy_{classifier_name}": metrics["accuracy"],
            f"roc_{classifier_name}": metrics["roc"],
            f"se_{classifier_name}": metrics["se"],
            f"sp_{classifier_name}": metrics["sp"],
            f"ppv_{classifier_name}": metrics["ppv"],
            f"npv_{classifier_name}": metrics["npv"],
        }

        return performance_result

    def get_metric_param_importance(self) -> tuple[dict, pd.DataFrame, list]:
        """(result)
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

        for classifier_name in self.classifier_list.keys():
            result = self.get_perform_metric(classifier_name)
            best_param = self.get_best_param(classifier_name)
            feature = self.get_feature_importance(classifier_name)

            clf_result.update(result)
            clf_feature = pd.concat([clf_feature, feature], axis=1)
            clf_param.append(best_param)

        # FIXED: result 값 return

        return clf_result, clf_feature, clf_param

    def export(self, clf_result, clf_feature, clf_param):
        #! FIXED: io_module의 to_excel과 add_excel_sheetd을 활용해서 아래 내용 수정

        ## dict to DataFrame (excel저장을 위해 dataframe으로 변경)
        tree_result = pd.DataFrame(clf_result)
        tree_feature = clf_feature
        tree_param = pd.DataFrame(clf_param)

        ## FIXED: result save (classifcation result, feautre importnace, hyperparmeter를 excel로 저장) ->writer.close() 필요하지 않은지?
        self.io_module.to_excelfile(
            "performanceresult_featureimportance_bestparameter.xlsx", tree_result
        )
        self.io_module.add_excel_sheet(
            "performanceresult_featureimportance_bestparameter.xlsx",
            sheet_name=f"{self.outcome}_result",
            data=tree_feature,
        )
        self.io_module.add_excel_sheet(
            "performanceresult_featureimportance_bestparameter.xlsx",
            sheet_name=f"{self.outcome}_param",
            data=tree_param,
        )
