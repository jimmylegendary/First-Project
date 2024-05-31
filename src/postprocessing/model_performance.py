from output.io_module import IOModule
<<<<<<< HEAD
from sklearn.metrics import roc_curve, accuracy_score, auc as auc_func
from scipy import interp
import numpy as np
import pandas as pd
import itertools
from postprocessing.my_statistics import MyStatistics

=======
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, auc as auc_func
import numpy as np
import pandas as pd
from scipy import interp
from math import sqrt
from scipy.special import ndtri
>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401


class Performance:
    def __init__(self, train_result, io_module):
        self.io_module: IOModule = io_module
        self.train_result = train_result
        self.best_fold = {}
<<<<<<< HEAD
        self.get_performance() # self.export() ->바로 이함수 부르면?

    def get_delong_pvalue(args, classifier_list, train_result):
        """calculate delong test
        +) delong_pvalue : {clf1_clf2 : delong_pvalue} (저장 필요)
        """
        # delong_pvalue 저장하기
        delong_test_result = {}  # dict(str, int)

        for clf_1, clf_2 in itertools.combinations(classifier_list, 2):
            real = train_result[clf_1].reals[0]
            clf_1_prob = train_result[clf_1].probs[0]
            clf_2_prob = train_result[clf_2].probs[0]
            delong_pvalue = MyStatistics.delong_roc_test(real, clf_1_prob, clf_2_prob, sample_weight=None)
            delong_test_result[f"{clf_1}_{clf_2}"] = delong_pvalue
        delong_pvalue_result = pd.DataFrame(delong_test_result)

        #FIXED!: result save (delong pvalue result를 excel로 저장)
        IOModule.to_excelfile("\\delong_pvalue_reuslt.xlsx", delong_pvalue_result)
        '''
        writer = pd.ExcelWriter(args.out_path + "\\delong_pvalue_result.xlsx")
        delong_pvalue_result.to_excel(writer)
        writer.close()
        '''

    def get_performance(self):
        clf_result, clf_feature, clf_param = self.get_metric_param_importance() 
        self.export(clf_result, clf_feature, clf_param)

=======

    def get_performance(self):
        pass
>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401

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

<<<<<<< HEAD
        ## FIXED: load and save best model: 성능 가장 좋았던 fold 확인해서 classifier 불러오기 & best_model_ 로 따로 저장해주기
=======
        ## load and save best model: 성능 가장 좋았던 fold 확인해서 classifier 불러오기 & best_model_ 로 따로 저장해주기
>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401
        grid_search_clf = self.io_module.from_picklefile(
            filename=f"{str(best_fold)}.pkl", 
            classifier_name=classifier_name
        )
        
        self.io_module.to_picklefile(
            filename=f"best_model_fold{str(best_fold)}.pkl", 
            data=grid_search_clf, 
            classifier_name=classifier_name
        )

        return grid_search_clf.best_params_
<<<<<<< HEAD
    
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

        return feature_importance_category
    
=======

>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401
    def get_perform_metric(self, classifier_name: str):
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
<<<<<<< HEAD
            _, _, se, sp, _, _, ppv, npv, _, _ = MyStatistics.find_optimal_cutoff(real[k], prob[k])
=======
            _, _, se, sp, _, _, ppv, npv, _, _ = self.find_optimal_cutoff(real[k], prob[k])
>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401

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

<<<<<<< HEAD
    def get_metric_param_importance(self) -> tuple(dict, pd.DataFrame, list):
=======
    def find_optimal_cutoff(self, real, pred):
        fpr, tpr, thresholds = roc_curve(real, pred)

        J = tpr - fpr
        ix = np.argmax(J)
        optimal_threshold = thresholds[ix]

        print(
            "Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f"
            % (optimal_threshold, tpr[ix], 1 - fpr[ix], J[ix])
        )

        temp = []
        for t in list(pred):
            if t >= optimal_threshold:
                temp.append(1)
            else:
                temp.append(0)

        TN, FP, FN, TP = confusion_matrix(real, temp).ravel()

        (
            sensitivity_point_estimate,
            specificity_point_estimate,
            sensitivity_confidence_interval,
            specificity_confidence_interval,
        ) = self.sensitivity_and_specificity_with_confidence_intervals(TP=TP, FP=FP, FN=FN, TN=TN)
        ppv_estimate, npv_estimate, ppv_confidence_interval, npv_confidence_interval = (
            self.ppv_and_npv_with_confidence_intervals(TP=TP, FP=FP, FN=FN, TN=TN)
        )

        return (
            optimal_threshold,
            ix,
            sensitivity_point_estimate,
            specificity_point_estimate,
            sensitivity_confidence_interval,
            specificity_confidence_interval,
            ppv_estimate,
            npv_estimate,
            ppv_confidence_interval,
            npv_confidence_interval,
        )

    def _proportion_confidence_interval(self, r, n, z):
        A = 2 * r + z**2
        B = z * sqrt(z**2 + 4 * r * (1 - r / n))
        C = 2 * (n + z**2)
        return ((A - B) / C, (A + B) / C)

    def ppv_and_npv_with_confidence_intervals(self, TP, FP, FN, TN, alpha=0.95):
        z = -ndtri((1.0 - alpha) / 2)

        ppv_estimate = TP / (TP + FP)
        ppv_confidence_interval = self._proportion_confidence_interval(TP, TP + FP, z)

        npv_estimate = TN / (TN + FN)
        npv_confidence_interval = self._proportion_confidence_interval(TN, TN + FN, z)

        print(
            "ppv : {} ({} {})".format(
                ppv_estimate, ppv_confidence_interval[0], ppv_confidence_interval[1]
            )
        )
        print(
            "npv : {} ({} {})".format(
                npv_estimate, npv_confidence_interval[0], npv_confidence_interval[1]
            )
        )

        return ppv_estimate, npv_estimate, ppv_confidence_interval, npv_confidence_interval

    def sensitivity_and_specificity_with_confidence_intervals(self, TP, FP, FN, TN, alpha=0.95):
        z = -ndtri((1.0 - alpha) / 2)

        sensitivity_point_estimate = TP / (TP + FN)
        sensitivity_confidence_interval = self._proportion_confidence_interval(TP, TP + FN, z)

        specificity_point_estimate = TN / (TN + FP)
        specificity_confidence_interval = self._proportion_confidence_interval(TN, TN + FP, z)
        print(
            "sensitivity : {} ({} {})".format(
                sensitivity_point_estimate,
                sensitivity_confidence_interval[0],
                sensitivity_confidence_interval[1],
            )
        )
        print(
            "specificity : {} ({} {})".format(
                specificity_point_estimate,
                specificity_confidence_interval[0],
                specificity_confidence_interval[1],
            )
        )

        return (
            sensitivity_point_estimate,
            specificity_point_estimate,
            sensitivity_confidence_interval,
            specificity_confidence_interval,
        )

    def save_performance_result(self):
>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401
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

        for classifier_name in self.classifier.keys():
            result = self.get_perform_metric(classifier_name)
            best_param = self.get_best_param(classifier_name)
            feature = self.get_feature_importance(classifier_name)

            clf_result.update(result)
            clf_feature = pd.concat([clf_feature, feature], axis=1)
            clf_param.append(best_param)

<<<<<<< HEAD
        # FIXED: result 값 return

        return clf_result, clf_feature, clf_param

    def export(self, clf_result, clf_feature, clf_param):
        #! FIXED: io_module의 to_excel과 add_excel_sheetd을 활용해서 아래 내용 수정
=======
    def export(self, clf_result, clf_feature, clf_param):
        #! TODO io_module의 to_excel과 add_excel_sheetd을 활용해서 아래 내용 수정하기
>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401

        ## dict to DataFrame (excel저장을 위해 dataframe으로 변경)
        tree_result = pd.DataFrame(clf_result)
        tree_feature = clf_feature
        tree_param = pd.DataFrame(clf_param)

<<<<<<< HEAD
        ## FIXED: result save (classifcation result, feautre importnace, hyperparmeter를 excel로 저장) ->writer.close() 필요하지 않은지?
        IOModule.to_excelfile("performanceresult_featureimportance_bestparameter.xlsx", tree_result)
        IOModule.add_excel_sheet(
            "performanceresult_featureimportance_bestparameter.xlsx", 
            sheet_name=f"{self.outcome}_result",
            data= tree_feature)
        IOModule.add_excel_sheet(
            "performanceresult_featureimportance_bestparameter.xlsx", 
            sheet_name=f"{self.outcome}_param",
            data= tree_param)
        
        ''' 이전 작성본
=======
        ## result save (classifcation result, feautre importnace, hyperparmeter를 excel로 저장)
>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401
        writer = pd.ExcelWriter(
            self.get_outpath("performanceresult_featureimportance_bestparameter.xlsx")
        )
        tree_result.to_excel(writer, sheet_name=f"{self.outcome}_result")
        tree_feature.to_excel(writer, sheet_name=f"{self.outcome}_feature")
        tree_param.to_excel(writer, sheet_name=f"{self.outcome}_param")
<<<<<<< HEAD
        writer.close()
        '''
=======
        writer.close()
>>>>>>> a0fc390776d4bc9cba664725f6f980dea1412401
