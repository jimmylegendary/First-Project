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
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, auc as auc_func

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA

from math import sqrt
from scipy.special import ndtri

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import make_pipeline
from lifelines import KaplanMeierFitter as KM
from lifelines import CoxPHFitter as CPH
from lifelines.statistics import logrank_test


class tree(object):
    def __init__(self, args):
        self.args = args
        self.outcome = args.target

        self.dataset = MyDataset  # 선언이 안됨

        self.target_func = {
            "vital": self.select_mortality_ahi_subject,
            "any_cvd": self.select_cvd_ahi_subject,
        }

        self.classifier = {"xgb": xgb(), "lgb": lgb(), "catb": catb(), "rf": rf(), "lr": lr()}

        self.best_fold = {"xgb": int, "lgb": int, "catb": int, "rf": int, "lr": int}

        self.feature_importance_cat = {}

        self.train_result: dict[str, MyTrainResult] = {}
        self.root_outdir = os.path.join(args.out_path, self.outcome)
        self.set_outdir(root_dir=self.root_outdir, classifier_name=self.classifier.keys())

    ## data save 지정
    def set_outdir(self, root_dir, classifier_name):
        """out directory 만들기 (save&load)
        Args:
            root_dir (str): output root directory
            classifier_name (str): classifier name (e.g. xgboost, lgbm, catboost)
        """
        dir_names = []
        for clf_name in classifier_name:
            dir_name = os.path.join(root_dir, clf_name)
            os.makedirs(dir_name, exist_ok=True)
            dir_names.append(dir_name)

    ## TRAIN 결과 계산 함수
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

        ## load and save best model: 성능 가장 좋았던 fold 확인해서 classifier 불러오기 & best_model_ 로 따로 저장해주기
        with open(self.get_outpath(f"{str(best_fold)}.pkl", classifier_name), "rb") as f:
            grid_search_clf = pickle.load(f)
        clf = grid_search_clf

        self.save_pkl(
            self.get_outpath(f"best_model_fold{str(best_fold)}.pkl", classifier_name), clf
        )

        return clf.best_params_

    def get_feature_importance(self, classifier_name: str) -> pd.DataFrame:
        """(result)
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

        # Create arrays from feature importance and feature names
        feature_mean_nparray = np.array(feature_mean)
        feature_names = np.array(self.dataset.feature_name, dtype=object)
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

        self.feature_importance_cat[classifier_name] = feature_importance_category

        return feature_importance_category

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

        # Sort the DataFrame in order decreasing feature importance
        fi_top30 = feature_importance_category.sort_values(
            by=["feature_importance"], ascending=False
        )[:30]

        # Define size of bar plot
        plt.figure(figsize=(18, 12))

        # Plot Searborn bar chart
        sns.barplot(
            x="feature_importance", y="feature_names", hue="feature_category", data=fi_top30
        )

        # Add chart labels
        plt.title(classifier_name + " FEATURE IMPORTANCE")
        plt.xlabel("FEATURE IMPORTANCE")
        plt.ylabel("FEATURE NAMES")

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

        fpr = self.roc_plot_val[classifier_name][0]
        tpr = self.roc_plot_val[classifier_name][1]

        plt.clf()
        figure_legend = f"{classifier_name}"
        plt.plot(fpr, tpr, label=figure_legend)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve")
        plt.savefig(self.get_outpath("roc_curve_figure.svg", classifier_name))

    def plot_km(self):
        """(plotting)
        kaplan-meier estimates by high and low risk figure plotting함 함수
        """
        args = self.args

        ## km plot (특정 dataset - probability 결과)
        dataset = pd.read_excel(args.plot_feature_path, sheet_name="survival_10")
        risk_ix = dataset["risk_group"] == 1

        ax = plt.subplot(111)

        time = dataset[f"{self.outcome}_date"].to_numpy()
        event = dataset[f"{self.outcome}"].to_numpy()

        ## Fit the data into the model
        km_high_risk = KM()
        ax = km_high_risk.fit(time[risk_ix], event[risk_ix], label="High risk").plot(ax=ax)

        km_low_risk = KM()
        ax = km_low_risk.fit(time[~risk_ix], event[~risk_ix], label="Low risk").plot(ax=ax)

        ## log rank
        results = logrank_test(
            time[risk_ix],
            time[~risk_ix],
            event_observed_A=event[risk_ix],
            event_observed_B=event[~risk_ix],
        )
        results.print_summary()

        ## Create an estimate
        plt.title("Kaplan-Meier estimates by High and Low risk")
        plt.ylabel("Survival probability (%)")
        plt.xlabel("Time (Days)")
        plt.savefig(self.get_outpath("km_plot_10.eps"))
        plt.clf()

    def plot_cox_func(self):
        """
        Cox Proportional Hazard model의 figure plotting함 함수
        """
        args = self.args
        outcome = self.outcome

        ## km plot (특정 dataset)
        dataset = pd.read_excel(args.plot_feature_path, sheet_name="survival_10")
        data = dataset[["vital_date", "vital", "age", "hypertension", "diabetes", "risk_group"]]

        plt.figure(figsize=(8, 4))
        cph = CPH()
        cph.fit(data, "vital_date", event_col="vital")
        cph.print_summary()
        cph.plot()

        ## Create an estimate
        plt.savefig(self.get_outpath("cox_HR_10_3.eps"))
        plt.clf()

    ## load final result
    def load_classifier(self):
        """load classifier (save&load)"""
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
            x_test, y_test = (dataset["test"]["x"], dataset["test"]["y"])

            with open(self.get_outpath(f"{str(kfold)}.pkl", classifier_name), "rb") as f:
                gird_search_clf = pickle.load(f)

            # grid search 결과 best parameter list에 저장
            best_param_list.append(gird_search_clf.best_params_)
            print("GridSearch 최고 점수: ", gird_search_clf.best_score_)
            print("GridSearch 최적 파라미터: ", gird_search_clf.best_params_)

            # gird searched model 중 가장 best estimator의 학습 결과 저장
            clf = gird_search_clf.best_estimator_.steps[2][1]
            pred_prob = clf.predict_proba(x_test)
            pred = pred_prob.argmax(1)
            pred_prob = pred_prob[:, 1]

            pred_list.append(pred)
            real_list.append(y_test.reshape(-1))
            prob_list.append(pred_prob)

            ## feature importance (logistic regression 제외 나머지 classifier의 feature importance 내장 함수 이용하여 feature importance 저장)
            feature_importance = (
                clf.feature_importances_ if classifier_name != "lr" else clf.coef_[0]
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
        A = 2 * r + z**2
        B = z * sqrt(z**2 + 4 * r * (1 - r / n))
        C = 2 * (n + z**2)
        return ((A - B) / C, (A + B) / C)

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

    def delong_roc_variance(self, ground_truth, predictions, sample_weight=None):
        ground_truth_stats = self.compute_ground_truth_statistics(ground_truth, sample_weight)
        order, label_1_count, ordered_sample_weight = ground_truth_stats

        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = self.fastDeLong(
            predictions_sorted_transposed, label_1_count, ordered_sample_weight
        )

        assert_msg = "There is a bug in the code, please forward this to the devs"
        assert len(aucs) == 1, assert_msg
        return aucs[0], delongcov

    def auc_ci_Delong(self, y_true, y_scores, alpha=0.95):

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Get AUC and AUC variance
        auc, auc_var = self.delong_roc_variance(y_true, y_scores)

        auc_std = np.sqrt(auc_var)

        # Confidence Interval
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        lower_upper_ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

        lower_upper_ci[lower_upper_ci > 1] = 1

        return auc, auc_var, lower_upper_ci


if __name__ == "__main__":
    tree_shhs = tree(args)

    # tree_shhs.load_classifier()
    tree_shhs.save_performance_result()