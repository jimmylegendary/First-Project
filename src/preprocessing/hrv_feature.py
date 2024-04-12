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

import biosppy
import csv
import pickle

from hrvanalysis import *
import matplotlib.pyplot as plt
from itertools import chain
from collections import defaultdict

import argparse

class hrv_feature(object):
    def __init__(self, args):
        self.args =args


    def get_hrv(self, pkl_file):
        
        # load data
        dir = pkl_file
        name = int(pkl_file.split('\\')[-1].split('.')[0])
        content = pickle.load(open(dir, 'rb'))
        ecg = content['signal']['ECG']
        ecg_f = ecg.reshape(-1)
        annot = content['stage']

        # bandpass filtering
        lowcut = 0.5
        highcut = 45
        ecg_f = self.butter_bandpass_filter(ecg_f, lowcut, highcut, 125, order=5)
        
        #signal split
        ecg_splited = self.five_min_split(ecg_f, annot)     
        if ecg_splited[2]==[] or ecg_splited[3]==[] or ecg_splited[4]==[]:
            print('insufficient ecg:{}'.format(name))

        # hrv feature
        hrv = [[],[],[]]
        ans = {}
        hrv_average = [[], [], []]
        result =[]

        for stage_index in list(range(2,5)):  
            # extracting hrv
            for stage_set in ecg_splited[stage_index]: 
                hrv[stage_index-2].append(self.hrv_feature(stage_set))
            
            # hrv averaging by epoch
            hrv_temp = hrv[stage_index-2]
            if hrv_temp == []:
                ans = {}
                continue
            else:
                
                keys = list(hrv_temp[0])
                                
                for epoch in range(len(hrv_temp)):
                    for feature in keys:
                        if feature not in ans:
                            if (hrv_temp[epoch].get(feature) == 'inf')|(hrv_temp[epoch].get(feature) == None):
                                continue
                            ans[feature] = hrv_temp[epoch].get(feature)
                            continue
                        else:
                            ans[feature] += hrv_temp[epoch].get(feature)
                ans = {feature+'_'+str(stage_index):[ans[feature]/len(hrv_temp)] for feature in ans}
                hrv_average[stage_index-2] = ans
                ans = {}
            

            # save averaged_hrv with subject name        
            result = dict(result, **hrv_average[stage_index-2])

            return result
                
                  

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a =butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y
    


    def nn_intervals(signal):
        rpeak = biosppy.signals.ecg.ecg(signal, sampling_rate = 125, show=False)[2]
        nni = tools.nn_intervals(rpeak)
        return nni


    def five_min_split(self, ecg, annot): 
        out = [ [],
                [],
                [],
                [],
                []]
        in_len = len(annot)

        # epoch(30)*10=300 (5 minutes)
        cutoff = in_len-10
        i = 0
        sw = 0
        while i<cutoff:
            j=1
            while i+j<cutoff:
                if i+j==cutoff-1:
                    sw = 1
                if annot[i] == annot[i+j] :              
                    j+=1
                else:
                    if j>=10:
                            
                        if (annot[i]==4)|(annot[i]==5):
                            annot[i] -= 1

                        if annot[i]==9:
                            annot[i]= 0   

                        out[annot[i]] += [list(range(i,i+j))]
                    i+=j                
                    break
            if sw == 1:
                break

        signal = [ [],
                    [],
                    [],
                    [],
                    []]

                            
        for k in range(5):
            for stage_set in out[k]:
                start = stage_set[0]*125*30
                end = stage_set[-1]*125*30
                signal[k].append(ecg[start:end])

        return signal


    def hrv_feature(self, signal):

        rpeaks = biosppy.signals.ecg.ecg(signal, sampling_rate = 125, show=False)[2]
        rpeak = [v*1000/125 for i, v in enumerate(rpeaks)]
        n = tools.nn_intervals(rpeak)
        nn = [v for i, v in enumerate(n) if v>100 and v<2000]

        # time domain feature
        nni_features = dict(td.nni_parameters(nni=nn))
        hr_features = dict(td.hr_parameters(nni=nn))
        rmssd_features = dict(td.rmssd(nni=nn))
        nn50_features = dict(td.nn50(nni=nn))
        frequency_domain_features = fd.welch_psd(nni=nn, show=False, show_param=False, legend=False)
        plt.close()
        frequency_feature_tp = dict(frequency_domain_features.as_dict())
        frequency_feature = {'fft_peak_vlf': frequency_feature_tp['fft_peak'][0], 'fft_peak_lf': frequency_feature_tp['fft_peak'][1], 'fft_peak_hf': frequency_feature_tp['fft_peak'][2],
        'fft_abs_vlf': frequency_feature_tp['fft_abs'][0], 'fft_abs_lf': frequency_feature_tp['fft_abs'][1], 'fft_abs_hf': frequency_feature_tp['fft_abs'][2],
        'fft_rel_vlf': frequency_feature_tp['fft_rel'][0], 'fft_rel_lf': frequency_feature_tp['fft_rel'][1], 'fft_rel_hf': frequency_feature_tp['fft_rel'][2],
        'fft_norm_lf': frequency_feature_tp['fft_norm'][0], 'fft_norm_hf': frequency_feature_tp['fft_norm'][1],
        'fft_ratio':frequency_feature_tp['fft_ratio'], 'fft_total':frequency_feature_tp['fft_total']}

        hrv_results = dict(**nni_features, **hr_features, **rmssd_features, **nn50_features,**frequency_feature)
        plt.close()
        return hrv_results

