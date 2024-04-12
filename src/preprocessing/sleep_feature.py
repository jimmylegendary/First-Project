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
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--npz_path', default='E:\\sleep\\data\\shhs_dataset\\npz_new', type=str)
    parser.add_argument('--demo_path', default='D:\\Hyunji\\Research\\Sleep\\neurotx\\data\\shhs\\shhs1-dataset-0.15.0.csv'
                        , type=str)
    parser.add_argument('--cvd_path', default='D:\\Hyunji\\Research\\Sleep\\neurotx\\data\\shhs\\shhs-cvd-summary-dataset-0.15.0.csv'
                        , type=str)
    parser.add_argument('--out_path', default='D:\\Hyunji\\Research\\Thesis\\research\\sleep\\data\\output'
                        , type=str)
    return parser.parse_args()

class sleep_feature(object):
    def __init__(self, args):
        self.args =args


    def get_objective_sleep(self, args):
     
        args = self.args
        # preprocessing of demographic data
        demo = pd.read_csv(args.demo_path)
        cvd_event = pd.read_csv(args.cvd_path)
        demo_data = self.demo_preprocessing(demo, cvd_event) 
        
        # load data
        dir = pkl_file
        name = int(pkl_file.split('\\')[-1].split('.')[0])
        content = pickle.load(open(dir, 'rb'))
        ecg = content['signal']['ECG']
        ecg_f = ecg.reshape(-1)
        annot = content['stage']
      
        # sleep feature extraction
        sleep_feature_, trans_Stage = self.sleep_feature(annot) 
        good_sleep = self.good_sleep_(sleep_feature_)
        sleep_feature_.update(good_sleep)
        sleep_feature_temp={str(key):[value] for key, value in sleep_feature_.items()}

        return sleep_feature_temp
                   

    def stage_transition(annot):
        annot_temp = np.append(annot, 0)
        index = [i for i, v in enumerate(annot) if v!=annot_temp[i+1]]
        value = [v for i, v in enumerate(annot) if v!=annot_temp[i+1]]
        trans_Stage = {v:value[i] for i, v in enumerate(index)}

        # sleep latency
        wake_index=index[0] if value[0]==0 else 0 
        sleep_latency = wake_index*30/60

        # wake after sleep onset
        start, end = index[0], index[-1]
        waso = list(annot[start+1:end]).count(0)*30/60

        # Awakening
        wake_indicies = [i for i, v in enumerate(value) if v==0]
        awake_count = [i for i in wake_indicies if index[i]-index[i-1]>10]
        awakening = len(awake_count)

        return sleep_latency, waso, awakening, trans_Stage 
    

    def sleep_feature(self, annot):

        # restaging annotation
        for i, temp_annot in enumerate(annot):
            if (temp_annot==4)|(temp_annot==5):
                annot[i] -= 1
            if temp_annot==9:
                annot[i]= 0   
        
        # count sleep stage
        stage, counts = np.unique(annot,return_counts=True)
        stage = list(stage)
        total_time = len(annot)
        total_sleep_time = sum(counts[1:5])

        # sleep stage ratio parameter
        #sleep_ratio={v:counts[i]/total_time for i, v in enumerate(stage)}
        sleep_ratio= {index:counts[index]/total_time for index in list(range(2,5))}

        # stage transition parameter
        sleep_latency, waso, awakening, trans_Stage =  self.stage_transition(annot)  
        sleep_efficiency = total_sleep_time/total_time

        feature ={'sleep_latency':sleep_latency, 'awakening':awakening, 'sleep_efficiency':sleep_efficiency}
        feature.update(sleep_ratio)
        
        return feature, trans_Stage

    def good_sleep_(feature):
        criteria={}
        sleep_latency_c = 1 if feature['sleep_latency']<15 else 0
        awakening_c = 1 if feature['awakening']<3 else 0
        #waso_c = 1 if feature['waso']<20 else 0
        sleep_efficiency_c = 1 if feature['sleep_efficiency']>0.85 else 0

        '''
        if 1 in feature:
            n1_c = 1 if feature[1]<0.05 else 0 
        else:
            n1_c=1
        '''
        if 3 in feature:
            n3_c = 1 if 0.16<feature[3]<0.2 else 0
        else:
            n3_c=0

        sum_all = sum([sleep_latency_c, awakening_c, sleep_efficiency_c, n3_c])
        criteria ={'sleep_latency_c':sleep_latency_c, 'awakening_c':awakening_c, 'sleep_efficiency_c':sleep_efficiency_c,
        'n3_c':n3_c, 'sum': sum_all}

        return criteria


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
