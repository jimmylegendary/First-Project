import numpy as np
import pandas as pd
import scipy.stats
from math import sqrt
from scipy.special import ndtri
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, auc as auc_func


# Class 사용하는 version, !FIXED: 필요하다면 class로, 추가적인 정리 필요함
class MyStatistics:
    def __init__(self):
        # INIT할게 없을 수 잇을지 -> 다른 방식으로 표현? (class 말고 그냥 function들)
        pass
    def compute_ground_truth_statistics(self, ground_truth, sample_weight):
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        if sample_weight is None:
            ordered_sample_weight = None
        else:
            ordered_sample_weight = sample_weight[order]

        return order, label_1_count, ordered_sample_weight

    def calc_pvalue(self, aucs, sigma):
        l_aux = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l_aux, sigma), l_aux.T))
        return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

    def delong_roc_test(self, ground_truth, pred_one, pred_two, sample_weight=None):
        order, label_1_count, _ = self.compute_ground_truth_statistics(ground_truth, sample_weight)

        predictions_sorted_transposed = np.vstack((pred_one, pred_two))[:, order]

        aucs, delongcov = self.fastDeLong(
            predictions_sorted_transposed, label_1_count, sample_weight
        )

        # print(aucs, delongcov)
        return self.calc_pvalue(aucs, delongcov)

    def fastDeLong(self, predictions_sorted_transposed, label_1_count, sample_weight):
        if sample_weight is None:
            return self.fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
        else:
            return self.fastDeLong_weights(
                predictions_sorted_transposed, label_1_count, sample_weight
            )

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
            tx[r, :] = self.compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
            ty[r, :] = self.compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
            tz[r, :] = self.compute_midrank_weight(pred_sorted_transposed[r, :], sample_weight)
        total_positive_weights = sample_weight[:m].sum()
        total_negative_weights = sample_weight[m:].sum()
        pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
        total_pair_weights = pair_weights.sum()
        aucs = (sample_weight[:m] * (tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
        v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / total_positive_weights
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
            T[i:j] = 0.5 * (i + j - 1)
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

'''
def cal_delong_test(args, classifier_list, train_result):
    """(statistics)
    calculate delong test
    +) delong_pvalue : {clf1_clf2 : delong_pvalue} (저장 필요)
    """
    # delong_pvalue 저장하기
    delong_test_result = {}  # dict(str, int)

    for clf_1, clf_2 in itertools.combinations(classifier_list, 2):
        real = train_result[clf_1].reals[0]
        clf_1_prob = train_result[clf_1].probs[0]
        clf_2_prob = train_result[clf_2].probs[0]
        delong_pvalue = delong_roc_test(real, clf_1_prob, clf_2_prob, sample_weight=None)
        delong_test_result[f"{clf_1}_{clf_2}"] = delong_pvalue
    delong_pvalue_result = pd.DataFrame(delong_test_result)

    ## result save (delong pvalue result를 excel로 저장)
    writer = pd.ExcelWriter(args.out_path + "\\delong_pvalue_result.xlsx")
    delong_pvalue_result.to_excel(writer)
    writer.close()


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def calc_pvalue(aucs, sigma):
    l_aux = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l_aux, sigma), l_aux.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def delong_roc_test(ground_truth, pred_one, pred_two, sample_weight=None):
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth, sample_weight)

    predictions_sorted_transposed = np.vstack((pred_one, pred_two))[:, order]

    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight)

    # print(aucs, delongcov)
    return calc_pvalue(aucs, delongcov)


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(pred_sorted_transposed, label_1_count, sample_weight):
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
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(pred_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m] * (tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
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
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1

    return T2


def compute_midrank_weight(x, sample_weight):
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
'''