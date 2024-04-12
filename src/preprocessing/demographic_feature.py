import os
import glob
import numpy as np
import mne
from scipy.signal import butter, lfilter

import sys
sys.path.extend(['D:\\Hyunji\\Research_D\\Sleep\\Quality prediction\\code\\src'])
import pandas as pd
from pyhrv import tools as tools
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd

import torch
import biosppy
import csv
import pickle

from hrvanalysis import *
import matplotlib.pyplot as plt
from itertools import chain
from collections import defaultdict

import argparse

class demographic_feature(object):
    def __init__(self, args):
        self.args =args

    def demo_preprocessing(demo, cvd_event):
        
        ## sleep quality data
        demo_data = pd.concat([demo[['nsrrid','age_s1']], demo[['LTDP10','SHLG10','REST10']].sum(axis=1)],axis=1)
        demo_data.columns=['nsrrid','age','sleep_quality']
        demo_data.reset_index(drop=True, inplace=True)
        demo_id = demo_data['nsrrid'].round().astype(int)
        demo_id = demo_id.tolist() 

        ## cvd & vital & age & gender data
        cvd_ = cvd_event[['any_cvd','any_chd']].sum(axis=1)
        vital = cvd_event[['vital']]
        pre_cvd = cvd_event[['prev_chf','prev_revpro','prev_mi','prev_mip','prev_stk','prev_ang']].sum(axis=1)
        cvd_data = pd.concat([cvd_event['nsrrid'], cvd_, pre_cvd, vital, cvd_event[['age_s1', 'gender']]], axis=1)
        cvd_data.columns=['nsrrid','any_cvd', 'pre_cvd', 'vital', 'age', 'gender']
        cvd_id = cvd_data['nsrrid'].round().astype(int)
        cvd_id = cvd_id.tolist()

        # id of subjective sleep quality
        demo_id_t = [demo_id.index(value) for i, value in enumerate(cvd_id) if value in demo_id]
        demo_data=demo_data.iloc[demo_id_t,:]
        demo_data.reset_index(drop=True, inplace=True)

        cvd_id_t = [i for i, value in enumerate(cvd_id) if value in demo_id]
        cvd_data = cvd_data.iloc[cvd_id_t,:]
        cvd_data.reset_index(drop=True, inplace=True)

        data = pd.merge(demo_data, cvd_data, how='left')

        return data



if __name__ == '__main__':

    arguments = get_args()   
    pkl_files = glob.glob(os.path.join(arguments.npz_path,'*.pkl'))

    for index,pkl_file in enumerate(pkl_files):
        try:
            HRV_shhs= hrv_shhs(arguments)
            hrv_result = HRV_shhs.get_hrv(pkl_file)
            objectvie_sleep_result = HRV_shhs.get_objective_sleep(pkl_file)


        except:   pass    
