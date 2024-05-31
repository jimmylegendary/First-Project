import argparse

import pandas as pd
from input.model_input import ModelInput
from model.gscv import GSCV
from output.io_module import IOModule
from postprocessing.model_performance import Performance
from postprocessing.my_statistics import cal_delong_test


def get_args():
    """인자값들 저장해두는 Namespace"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outcome", default="vital", type=str
    )  # required=True, choices = ['vital','any_cvd']
    parser.add_argument("--data_path", default="dataset\\data.xlsx", type=str)
    parser.add_argument("--out_path", default="output\\15", type=str)
    parser.add_argument("--plot_feature_path", default="dataset\\plot_data.xlsx", type=str)

    parser.add_argument("--hyper_params", required=True, type=str)
    parser.add_argument("--drop_list", required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_input = ModelInput(args)
    model_input.make_dataset(args.data_path)
    classifier_list = model_input.classifier.keys()

    io_module = IOModule(root_outdir="./result")
    model = GSCV(io_module, model_input)

    clf_result = {}
    clf_feature = pd.DataFrame()
    clf_param = []
    train_results = {}

    for classifier_name in classifier_list:
        train_result = model.train(classifier_name, model_input)
        train_results[classifier_name] = train_result
        performance = Performance(io_module=io_module, classifier_name=classifier_name)
        result, best_param, feature = performance.get_performance(
            classifier_name=classifier_name,
            feature_name=model.dataset.feature_name,
            train_result=train_result,
        )
        clf_result.update(result)
        clf_feature = pd.concat([clf_feature, feature], axis=1)
        clf_param.append(best_param)

    tree_result = pd.DataFrame(clf_result)
    tree_feature = clf_feature
    tree_param = pd.DataFrame(clf_param)

    io_module.to_excelfile("performanceresult_featureimportance_bestparameter.xlsx", tree_result)
    io_module.add_excel_sheet(
        "performanceresult_featureimportance_bestparameter.xlsx",
        sheet_name=f"{args.outcome}_result",
        data=tree_feature,
    )
    io_module.add_excel_sheet(
        "performanceresult_featureimportance_bestparameter.xlsx",
        sheet_name=f"{args.outcome}_param",
        data=tree_param,
    )
    delong_pvalue_result = cal_delong_test(
        classifier_list=classifier_list, train_results=train_results
    )
    ## result save (delong pvalue result를 excel로 저장)
    io_module.to_excelfile(filename="delong_pvalue_result.xlsx", data=delong_pvalue_result)

    # tree_shhs.save_performance_result()

    # 통계 + plotting (독립적으로 실행 가능)
    # tree_shhs.plot_classifier()
    # tree_shhs.plot_km()
    # tree_shhs.plot_cox_func()
