from sklearn.model_selection import StratifiedKFold
import pandas as pd
from src.input.dataset import MyDataset

import json
from catboost import CatBoostClassifier as catb
from lightgbm import LGBMClassifier as lgb
from xgboost import XGBClassifier as xgb
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rf


class ModelInput:
    def __init__(self, args):
        self.dataset: MyDataset
        self.cv = StratifiedKFold(5, shuffle = True, random_state = 10)
        self.outcome = args.outcome
        
        self.drop_list: list[str] 
        self.hyperparameter: dict[str, object] 
        self.classifier: dict[str, object] 

        self.target_func = {
            "vital": self.select_mortality_ahi_subject,
            "any_cvd": self.select_cvd_ahi_subject
        }
        
        self.init_config(args)        

    def init_config(self, args):
        #!TODO args에서 model input/config을 위해 추가적으로 필요한 모든것 여기에 작성
        # load droplist 
        with open(args.drop_list, "r") as f:
            self.drop_list = json.load(f)
        
        # load hyperparameter 
        with open(args.hyper_params, "r") as f:
            self.hyperparameter = json.load(f)
        
        # self.classifier[clf_name]에 classifier 함수 할당
        for clf_name in self.hyperparameter.keys():
            try:
                self.classifier[clf_name] = eval(f"{clf_name}()")
            except NameError:
                raise Exception(f"You should import module which includes {clf_name} classifier")
        

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
    
    def extract_features(self, total_data: pd.DataFrame):
        """Target outcome에 따라 grouping된 feature 값을 반환하는 함수 (dataset)
        + grouping_data 에 concated_group 결과 저장
        + outcome에 target outcome 저장
        Args:
            total_data: 전체 feature data (DataFrame)
        """
        group_0, group_1 = self.target_func[self.outcome](total_data)
        concated_group = pd.concat([group_0, group_1])
        # grouping data / outcome 저장
        return concated_group
        
    def make_dataset(self, data_path):
        """
        data path로부터 excel read 후 grouping data에 따라 x, y, feature name 저장 (MyDataset)
        fold별로 dataset 저장 ()

        Args:
            data_path (str): data path
        """
        total_data = pd.read_excel(data_path)
        grouping_data = self.extract_features(total_data)
        feature_name = grouping_data.drop(self.drop_list, axis = 1).columns.to_list() 
        x = grouping_data.drop(self.drop_list, axis = 1).to_numpy()
        y = grouping_data[f'{self.outcome}'].to_numpy()
        
        self.dataset = MyDataset(
            feature_name = feature_name,
            x = x,
            y = y            
        )
        self.dataset.make_folded(self.cv)
        
        