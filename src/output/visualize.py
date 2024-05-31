## plotting
class Visualize:
    def __init__(self):
        pass

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
