import joblib
import os
import pandas as pd


class IOModule:
    def __init__(self, root_outdir=".", classifier_list: object= object):
        self.root_outdir = root_outdir
        self.excel_writers: dict[str, pd.ExcelWriter] = {}
        self.set_outdir(
            root_dir = self.root_outdir, classifier_list=classifier_list
            )

    #!TODO root_outdir check 및 폴더 생성
    def set_outdir(self, root_dir, classifier_list):
        """ out directory 만들기 (save)

        Args:
            root_dir (str): output root directory
            classifier_list (str): classifier name (e.g. xgboost, lgbm, catboost)
        """
        dir_names = []
        for clf_name in classifier_list:
            dir_name = os.path.join(root_dir, clf_name)
            os.makedirs(dir_name, exist_ok=True)
            dir_names.append(dir_name)
    
    #!TODO export된 filename, file을 type별? 관리    

    def from_picklefile(self, filename, classifier_name=""):
        out_path = self.get_outpath(filename, classifier_name)
        return joblib.load(out_path)

    def to_picklefile(self, filename, data, classifier_name=""):  
        out_path = self.get_outpath(filename, classifier_name)
        joblib.dump(data, out_path)

    def to_excelfile(self, filename, dataframe: pd.DataFrame = pd.DataFrame(), classifier_name=""):
        out_path = self.get_outpath(filename, classifier_name)
        writer = pd.ExcelWriter(out_path)
        dataframe.to_excel(writer)
        
        #!TODO excel writer file name별로 저장 수정하기
        self.excel_writers[filename] = writer

    def add_excel_sheet(self, excel_filename, sheet_name, dataframe: pd.DataFrame):
        writer = self.excel_writers[excel_filename]
        dataframe.to_excel(writer, sheet_name)

    def get_outpath(self, file_name, classifier_name=""):
        """최종 output path 받아오기 (save&load)
        Args:
            file_name (str): 저장 하고자 하는 file 이름의 앞부분
            classifier_name (str, optional): classifier name (e.g. xgboost, lgbm, catboost). Defaults to "".
        """
        return os.path.join(self.root_outdir, classifier_name, file_name)