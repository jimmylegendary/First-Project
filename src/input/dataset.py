import numpy as np

class MyDataset:
    def __init__(self, feature_name: list, x: np.ndarray, y: np.ndarray):
        self.feature_name = feature_name
        self.x = x
        self.y = y
        self.folded_dataset = []
    
    def make_folded(self, cv):
        self.folded_dataset = []

       # cross validation에 따라 train, test split
        for train_index, test_index in cv.split(self.x, self.y):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]            
            dataset = {
                'train' : {
                    'x' : x_train,
                    'y' : y_train
                },
                'test' : {
                    'x' : x_test,
                    'y' : y_test
                }
            }
            # fold별 train, test data 저장
            self.folded_dataset.append(dataset)
        
