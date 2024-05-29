import argparse
from input.model_input import ModelInput
from model.gscv import GSCV
from output.io_module import IOModule
from postprocessing.algorithm import cal_delong_test
from postprocessing.model_performance import Performance

def get_args():
    """인자값들 저장해두는 Namespace"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="vital", type=str)  # required=True, choices = ['vital','any_cvd']
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
    
    io_module = IOModule(root_outdir="./result", classifier_list=model_input.classifier.keys())
    model = GSCV(io_module)

    for classifier_name in model_input.classifier.keys():
        model.train(classifier_name, model_input)
        cal_delong_test(args, model_input.classifier.keys(), model.train_result)

    Performance(train_result=model.train_result, io_module=io_module)
    
    
    #tree_shhs.save_performance_result()

    # 통계 + plotting (독립적으로 실행 가능)
    # tree_shhs.plot_classifier()
    # tree_shhs.plot_km()
    # tree_shhs.plot_cox_func()