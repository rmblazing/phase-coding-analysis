#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle as pkl
import scipy.stats
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
import sklearn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import permutation_test_score
from sklearn.feature_selection import mutual_info_classif

# This is a set of classes/methods for analyzing pattern stimulation data


class data_prepper:
    def __init__(self, hdf5_path, trigger_file_path, resp_file_path, expt_metadata_path, expt_type = [], laser_trigger_file_path = []):
        self.hdf5_path = hdf5_path
        self.trigger_file_path = trigger_file_path
        self.resp_file_path = resp_file_path
        self.expt_metadata_path = expt_metadata_path
        if laser_trigger_file_path:
            self.laser_trigger_file_path = laser_trigger_file_path
            self.laser_trigger_array = get_events(self.laser_trigger_file_path)
        with open(self.expt_metadata_path, 'rb') as f:
            self.expt_metadata = pkl.load(f)
        self.trigger_array = get_events(self.trigger_file_path)
        self.resp_array = get_events(self.resp_file_path)
        spiketimes_allcells = []
        for file in self.hdf5_path:
            tsecs_file, spike_index, clusternumber = get_spiketimes_s(file)
            spiketimes_allcells.append(tsecs_file)
        self.tsecs = [cell for experiment in spiketimes_allcells for cell in experiment]
        self.tsecs = np.array(self.tsecs)
        if expt_type == 'size':
            self.frame_on_time_s, self.frame_on_idx = get_frameontimes(self.trigger_array, self.expt_metadata)
        elif expt_type == 'sequence':
            self.frame_on_time_s, self.frame_on_idx = get_frameontime_sequence(self.trigger_array, self.expt_metadata)
        else:
            sys.exit("must define an experiment type - either 'size' or 'sequence'") 
        self.fs = 30000 

    def get_spikes(self):
        return self.tsecs
        
    def get_metadata(self):
        return self.expt_metadata
    
    def get_spiketimes(self):
        return self.frame_on_time_s
    
    def get_spikeidx(self):
        return self.frame_on_idx
    
    def get_resp_array(self):
        return self.resp_array
    
    def get_size_index_array(self):
        return get_size_indices(self.expt_metadata)
    
    def get_trigger_array(self):
        return self.trigger_array
    
    def get_laser_trigger_array(self):
        return self.laser_trigger_array
    
    def get_frame_trial_indices(self):
        unique_frame_idx = np.unique(self.expt_metadata['frame_idx'])
        frame_trial_indices = []
        for idx in unique_frame_idx: 
            frame_trial_indices.append(np.where(self.expt_metadata['frame_idx'] == idx)[0])
        return frame_trial_indices
    
    def get_frame_trial_indices_opto(self):
        unique_frame_idx = np.unique(self.expt_metadata['frame_idx'])
        laser_off_frame_trial_indices = []
        laser_on_frame_trial_indices = []
        for idx in unique_frame_idx: 
            # 2 is index for laser off trials
            laser_off_frame_trial_indices.append(np.where((self.expt_metadata['frame_idx'] == idx) & (self.expt_metadata['laser_idx']==2))[0])
            # 3 is index for laser on trials 
            laser_on_frame_trial_indices.append(np.where((self.expt_metadata['frame_idx'] == idx) & (self.expt_metadata['laser_idx']==3))[0])
        frame_trial_indices = np.vstack((laser_off_frame_trial_indices, laser_on_frame_trial_indices))
        return frame_trial_indices

    
class breath_stats: 
    def __init__(self, resp_array, frame_on_time_idx, expt_metadata, frame_trial_indices, interval = [-2.54, 2.54], fs = 30000): 
        self.resp_array = resp_array
        self.frame_on_time_idx = frame_on_time_idx
        self.expt_metadata = expt_metadata
        self.frame_trial_indices = np.array(frame_trial_indices).astype(int)
        self.resp_aligned = []
        self.interval = interval
        self.fs = fs
        for stimulus in self.frame_on_time_idx: 
            self.resp_aligned.append(resp_array[stimulus+int(interval[0]*fs): stimulus + int(interval[1]*fs)]/1000) #millivolt to volt conversion
        self.resp_aligned = np.array(self.resp_aligned)
        
    def get_resp_aligned(self):
        return self.resp_aligned
    
    def align_inh_by_trial_type(self, input_array):
        array_by_trial = []
        array_by_trialtype = []
        for trial_type in self.frame_trial_indices: 
            for entry in trial_type:
                array_by_trial.append(input_array[entry])
            array_by_trialtype.append(array_by_trial)
            array_by_trial = []
        return array_by_trialtype
        
    def get_pre_post_breaths(self):
        pre_breath_indices = []
        post_breath_indices = []
        for trial_idx, trial in enumerate(self.resp_aligned):
            pre_breath_trig_diff, _, post_breath_trig_diff, _, _, _ = mean_neg_cross(trial)
            pre_breath_indices.append(pre_breath_trig_diff) 
            post_breath_indices.append(post_breath_trig_diff) 
        pre_breath_idx = 0 - np.array(pre_breath_indices)
        pre_breath_time = pre_breath_idx/self.fs
        post_breath_idx = 0 - np.array(post_breath_indices)
        post_breath_time = (post_breath_idx/self.fs) - (500/self.fs)
        breath_times = {'pre': pre_breath_time, 'post': post_breath_time, 'pre_idx':pre_breath_idx, 'post_idx': post_breath_idx,  'pre_breath_trig_diff': pre_breath_trig_diff, 'post_breath_trig_diff': post_breath_trig_diff}
        return breath_times
    
    def get_trial_aligned_breaths(self): 
        trial_aligned_breath_idxs = []
        trial_aligned_breath_idxs_trigger_aligned = []
        for trial_idx, trial in enumerate(self.resp_aligned):
            _, _, _, _, neg_cross_idxs, neg_cross_idxs_aligned = mean_neg_cross(trial)
            trial_aligned_breath_idxs.append(neg_cross_idxs)
            trial_aligned_breath_idxs_trigger_aligned.append(neg_cross_idxs_aligned/self.fs)
        return trial_aligned_breath_idxs, trial_aligned_breath_idxs_trigger_aligned
    
    def get_inhalation_mean_std(self, plot = False, xlim = [-2.54, 2.54]):
        mean_breath_trace = np.mean(self.resp_aligned,0) 
        std_breath_trace = np.std(self.resp_aligned,0)
        if plot == True: 
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = np.arange(self.interval[0], self.interval[1], (self.interval[1] - self.interval[0])/((self.interval[1] - self.interval[0])*self.fs))
            plt.plot(x, mean_breath_trace)
            ax.fill_between(x, mean_breath_trace - std_breath_trace, mean_breath_trace + std_breath_trace, alpha = .2, color = 'b')
            plt.xlim(xlim) 
            breath_trace = {'mean':mean_breath_trace, 'std':std_breath_trace, 'x':x}
            return breath_trace
   
    def get_inhalation_mean_std_bysize(self, plot = False, xlim = [-2.54, 2.54]):
        spotnum_indices, pixels_by_spotnum = get_size_indices(expt_metadata)
        resp_by_spotnum = []
        respmean_by_spotnum = []
        respstd_by_spotnum = []
        size_indices = []
        for idx, spotnum in enumerate(spotnum_indices):
            for num in spotnum: 
                size_indices.append(np.where(np.array(self.expt_metadata['frame_idx']) == num))
            size_indices = np.ravel(size_indices)
            resp_by_spotnum.append(self.resp_aligned[size_indices])
            respmean_by_spotnum.append(np.mean(resp_by_spotnum[idx],0))
            respstd_by_spotnum.append(np.std(resp_by_spotnum[idx],0))
            size_indices = []               
        timepoints = np.arange(interval[0], interval[1], (interval[1]-interval[0])/len(respmean_by_spotnum[0]))
        return respmean_by_spotnum, respstd_by_spotnum, timepoints 


class response_stats: 
    def __init__(self, trial_type_raster, expt_metadata, pre_breath_time_shaped, post_breath_time_shaped, mod_time = False, mod_breath_time_1 = 0, mod_breath_time_2 = 0, pattern_type = 'sequence'): 
        self.pattern_type = pattern_type
        self.trial_type_raster = trial_type_raster
        self.expt_metadata = expt_metadata
        self.ncells = len(trial_type_raster)
        self.nstim = len(trial_type_raster[0])
        self.nstim_trials = len(trial_type_raster[0][0])
        self.pre_inh_spike_count = np.zeros((self.ncells, self.nstim, self.nstim_trials))
        self.post_inh_spike_count = np.zeros((self.ncells, self.nstim, self.nstim_trials))
        self.pvals = np.zeros((self.ncells, self.nstim))
        self.auc = np.zeros((self.ncells, self.nstim))
        if mod_time == False:
            for cell_idx, cell in enumerate(trial_type_raster):
                for trial_type_idx, trial_type in enumerate(cell):
                    for trial_idx, trial in enumerate(trial_type):
                        self.pre_inh_spike_count[cell_idx, trial_type_idx, trial_idx] = len((trial[(trial >= pre_breath_time_shaped[trial_type_idx][trial_idx]) & (trial < 0)]))
                        self.post_inh_spike_count[cell_idx, trial_type_idx, trial_idx] = len(trial[(trial >= 0) & (trial <= post_breath_time_shaped[trial_type_idx][trial_idx])])
        if mod_time == True:      
            for cell_idx, cell in enumerate(trial_type_raster):
                for trial_type_idx, trial_type in enumerate(cell):
                    for trial_idx, trial in enumerate(trial_type):
                        self.pre_inh_spike_count[cell_idx, trial_type_idx, trial_idx] = len((trial[(trial >= pre_breath_time_shaped[trial_type_idx][trial_idx]) & (trial < pre_breath_time_shaped[trial_type_idx][trial_idx] + mod_breath_time_2)]))
                        self.post_inh_spike_count[cell_idx, trial_type_idx, trial_idx] = len(trial[(trial >= mod_breath_time_1) & (trial <= mod_breath_time_2)])
                        
    def get_resp_aligned_spike_counts(self):
        spike_counts = {'pre':self.pre_inh_spike_count, 'post':self.post_inh_spike_count}
        return spike_counts
    
    def get_zscored_counts(self):
        z_scored_dist = np.empty((self.ncells, self.nstim, self.nstim_trials, self.nstim_trials+1))
        zscored_spike_counts = np.empty((self.ncells, self.nstim, self.nstim_trials))
        for cell_idx, cell in enumerate(self.trial_type_raster):
            for trial_type_idx, trial_type in enumerate(cell):
                for trial_idx, trial in enumerate(trial_type):
                    z_scored_dist[cell_idx, trial_type_idx, trial_idx, 0:self.nstim_trials] = self.pre_inh_spike_count[cell_idx, trial_type_idx]
                    z_scored_dist[cell_idx, trial_type_idx, trial_idx, self.nstim_trials] = self.post_inh_spike_count[cell_idx, trial_type_idx, trial_idx]
                    z_scored_dist[cell_idx, trial_type_idx, trial_idx] = scipy.stats.zscore(z_scored_dist[cell_idx, trial_type_idx, trial_idx])
                    if np.isnan(z_scored_dist[cell_idx, trial_type_idx, trial_idx, self.nstim_trials]):
                        zscored_spike_counts[cell_idx, trial_type_idx, trial_idx] = 0
                    else: 
                        zscored_spike_counts[cell_idx, trial_type_idx, trial_idx] = z_scored_dist[cell_idx, trial_type_idx, trial_idx, self.nstim_trials] 
                    
        return zscored_spike_counts
                    
    def get_resp_aligned_binned_spike_counts(self, bin_size_s, nbins):
        self.nbins = nbins
        self.binned_spike_counts = np.zeros((self.ncells, self.nstim, self.nstim_trials, self.nbins+1))
        self.binned_spike_counts_z = np.zeros((self.ncells, self.nstim, self.nstim_trials, self.nbins+1))
        self.zscored_post_spike_counts = np.zeros((self.ncells, self.nstim, self.nstim_trials))
        self.bin_size = bin_size_s
        for cell_idx, cell in enumerate(self.trial_type_raster):
            for trial_type_idx, trial_type in enumerate(cell):
                for trial_idx, trial in enumerate(trial_type):
                    for bin_count in range(self.nbins):
                        self.binned_spike_counts[cell_idx, trial_type_idx, trial_idx, bin_count] = len((trial[(trial >= (-bin_count*(self.bin_size+1))) & (trial < (-bin_count*(self.bin_size)))]))
                    self.binned_spike_counts[cell_idx, trial_type_idx, trial_idx, bin_count+1] = len((trial[(trial >= 0) & (trial < bin_size_s)]))
                    self.binned_spike_counts_z[cell_idx, trial_type_idx, trial_idx] = scipy.stats.zscore(self.binned_spike_counts[cell_idx, trial_type_idx, trial_idx])
                    if np.isnan(self.binned_spike_counts_z[cell_idx, trial_type_idx, trial_idx, bin_count + 1]):
                        self.zscored_post_spike_counts[cell_idx, trial_type_idx, trial_idx] = 0
                    else: 
                        self.zscored_post_spike_counts[cell_idx, trial_type_idx, trial_idx] = self.binned_spike_counts_z[cell_idx, trial_type_idx, trial_idx, bin_count + 1]
                           
        return self.zscored_post_spike_counts
    
    def do_rank_sum_test(self):
        for cell in range(self.ncells):
            for stimulus in range(self.nstim):
                try:
            # had to get a little creative here to get the ranksums value R used to calculate the P value. R can be derived from the U statistic in the mann-whitneyU test. 
                    statistic, p_value = scipy.stats.ranksums(self.post_inh_spike_count[cell][stimulus], self.pre_inh_spike_count[cell][stimulus])  
                    stat, p = scipy.stats.mannwhitneyu(self.post_inh_spike_count[cell][stimulus], self.pre_inh_spike_count[cell][stimulus], alternative = "two-sided")
                    R = stat + (((len(self.post_inh_spike_count[cell][stimulus])*(len(self.post_inh_spike_count[cell][stimulus])+1))/2))
                    auROC = (R - len(self.post_inh_spike_count[cell][stimulus])*(len(self.post_inh_spike_count[cell][stimulus])+1)/2)/(len(self.post_inh_spike_count[cell][stimulus])*len(self.post_inh_spike_count[cell][stimulus]));
                    self.pvals[cell,stimulus] = p_value
                    self.auc[cell,stimulus] = auROC
                except: 
                    self.pvals[cell,stimulus] = np.nan
                    self.auc[cell,stimulus] = np.nan
        return self.pvals, self.auc 
    
    def do_rank_sum_test_toblank(self, blankresp):
        for cell in range(self.ncells):
            for stimulus in range(self.nstim):
                try:
            # had to get a little creative here to get the ranksums value R used to calculate the P value. R can be derived from the U statistic in the mann-whitneyU test. 
                    statistic, p_value = scipy.stats.ranksums(self.post_inh_spike_count[cell][stimulus], blankresp[cell])  
                    stat, p = scipy.stats.mannwhitneyu(self.post_inh_spike_count[cell][stimulus], blankresp[cell], alternative = "two-sided")
                    R = stat + (((len(self.post_inh_spike_count[cell][stimulus])*(len(self.post_inh_spike_count[cell][stimulus])+1))/2))
                    auROC = (R - len(self.post_inh_spike_count[cell][stimulus])*(len(self.post_inh_spike_count[cell][stimulus])+1)/2)/(len(self.post_inh_spike_count[cell][stimulus])*len(self.post_inh_spike_count[cell][stimulus]));
                    self.pvals[cell,stimulus] = p_value
                    self.auc[cell,stimulus] = auROC
                except: 
                    self.pvals[cell,stimulus] = np.nan
                    self.auc[cell,stimulus] = np.nan
        return self.pvals, self.auc 
    
    def get_response_props(self, size_indices, cutoff = .05):
        nsizes = len(size_indices)
        if self.pattern_type == 'size':
            trials_per_size = len(size_indices[0])
        if self.pattern_type == 'sequence':
            trials_per_size = len(size_indices)
        p_thresh = []
        auc_activated = []
        auc_suppressed = []
        for cell in self.pvals: 
            p_thresh.append(cell < cutoff)
        for cell in self.auc:
            auc_activated.append(cell > .5)
            auc_suppressed.append(cell < .5)

        response_props = np.zeros((self.ncells, self.nstim))
        sig_activated = []
        sig_suppressed = []
        for idx, cell in enumerate(p_thresh):
            for entry in range(len(cell)):
                if (p_thresh[idx][entry] == True) & (auc_activated[idx][entry] == True):
                    response_props[idx,entry] = 1
                elif (p_thresh[idx][entry] == True) & (auc_suppressed[idx][entry] == True):
                    response_props[idx,entry] = -1
                else:
                    response_props[idx, entry] = 0
        if self.pattern_type == 'size':
            response_props = np.squeeze(response_props)
            prop_suppressed = np.zeros((nsizes, trials_per_size))
            prop_activated = np.zeros((nsizes, trials_per_size))
            prop_nonresp = np.zeros((nsizes, trials_per_size))
            for nsize, size in enumerate(size_indices):
                for num, idx in enumerate(size):
                    prop_suppressed[nsize, num] = (len(np.where(response_props[:,idx] == -1)[0])/len(response_props[:,idx]))*100
                    prop_activated[nsize, num] = (len(np.where(response_props[:,idx] == 1)[0])/len(response_props[:,idx]))*100
                    prop_nonresp[nsize, num] = (len(np.where(response_props[:,idx] == 0)[0])/len(response_props[:,idx]))*100
        if self.pattern_type == 'sequence':
            prop_suppressed = np.zeros((nsizes))
            prop_activated = np.zeros((nsizes))
            prop_nonresp = np.zeros((nsizes))
            for nsize, size in enumerate(size_indices):
                prop_suppressed[nsize] = (len(np.where(response_props[:,size] == -1)[0])/len(response_props[:,size]))*100
                prop_activated[nsize] = (len(np.where(response_props[:,size] == 1)[0])/len(response_props[:,size]))*100
                prop_nonresp[nsize] = (len(np.where(response_props[:,size] == 0)[0])/len(response_props[:,size]))*100
        response_proportions = {'activated':prop_activated, 'suppressed':prop_suppressed, 'non_responding':prop_nonresp, 'all_responses':response_props}
        return response_proportions
    
    def get_tuning(self, all_response_props, size_indices):
        tuning_by_size_activated = []
        tuning_by_size_suppressed = []
        for size_idx in size_indices:
            activated_tuning = np.zeros(len(all_response_props[:, size_idx]))
            suppressed_tuning = np.zeros(len(all_response_props[:, size_idx]))
            for idx, cell in enumerate(all_response_props[:, size_idx]):
                activated_tuning[idx] = np.sum(cell == 1)
                suppressed_tuning[idx] = np.sum(cell == -1)

            frame_idx = np.arange(len(size_idx))
            frame_counts_activated = np.zeros(len(size_idx))
            frame_counts_suppressed = np.zeros(len(size_idx))
            for frame in frame_idx:
                prop_counts_activated = (len(np.where(activated_tuning.astype(int) == frame+1)[0])/len(activated_tuning))*100
                prop_counts_suppressed = (len(np.where(suppressed_tuning.astype(int) == frame+1)[0])/len(suppressed_tuning))*100
                frame_counts_activated[frame] = prop_counts_activated
                frame_counts_suppressed[frame] = prop_counts_suppressed
            tuning_by_size_activated.append(frame_counts_activated)
            tuning_by_size_suppressed.append(frame_counts_suppressed)
            tuning_by_size = {'activated':tuning_by_size_activated, 'suppressed':tuning_by_size_suppressed}
        return tuning_by_size

                             
class correlations: 
    def __init__(self, spike_counts):
        self.spike_counts = spike_counts
        self.nstim = len(spike_counts[0])
        self.nstim_trials = len(spike_counts[0][0])

    def get_corr_matrix(self, set_ranges = False, stim_range_vals = [0,20], trial_range_vals = np.arange(20)):
        if set_ranges == True:
            stim_range = stim_range_vals
            stim_range_add = stim_range[1] + 1 
            trial_range = trial_range_vals
            nstim = (stim_range_vals[1]-stim_range_vals[0])+1
            nstim_trials = len(trial_range_vals)
        if set_ranges == False:
            stim_range = [0, self.nstim]
            stim_range_add = stim_range[1]
            trial_range = np.arange(self.nstim_trials)  
            nstim = self.nstim
            nstim_trials = self.nstim_trials
        mat_width = nstim*nstim_trials
        tbt_corr_mat = np.zeros((mat_width,mat_width))
        count_stim_1 = int(0)
        stim_corr_mean = np.empty((nstim,nstim))
        stim_corr_allmeans = []
        for stim1_idx in range(stim_range[0],stim_range_add):
            count_stim_2 = int(0)
            for stim2_idx in range(stim_range[0], stim_range_add):
                vec1 = self.spike_counts[0:,stim1_idx,trial_range].T
                vec2 = self.spike_counts[0:,stim2_idx,trial_range].T
                stim_corr = np.corrcoef(vec1, vec2)
                if stim1_idx == stim2_idx:
                    h = np.arange(nstim_trials, nstim_trials*2)
                    i = np.arange(0, nstim_trials)
                    stim_corr[h,i] = np.nan
                stim_corr = stim_corr[nstim_trials:(nstim_trials*2), 0:nstim_trials]
                stim_corr_mean[count_stim_1, count_stim_2] = np.nanmean(stim_corr)
                tbt_corr_mat[0+(nstim_trials*count_stim_1):nstim_trials+(nstim_trials*count_stim_1), 0 + (nstim_trials*count_stim_2):nstim_trials+(nstim_trials*count_stim_2)] = stim_corr
                count_stim_2 += 1   
            count_stim_1 += 1
            stim_corr_allmeans.append(stim_corr_mean)
        corr_matrix = {'trial_by_trial':tbt_corr_mat, 'mean_correlations':stim_corr_mean}
        #corr_matrix = {'trial_by_trial' = corr_mat, 'stimulus_average' = 
        return corr_matrix     
    
    def get_cosine_similarity_matrix(self, set_ranges = False, stim_range_vals = [0,20], trial_range_vals = np.arange(20)):
        if set_ranges == True:
            stim_range = stim_range_vals
            stim_range_add = stim_range[1] + 1 
            trial_range = trial_range_vals
            nstim = (stim_range_vals[1]-stim_range_vals[0])+1
            nstim_trials = len(trial_range_vals)
        if set_ranges == False:
            stim_range = [0, self.nstim]
            stim_range_add = stim_range[1]
            trial_range = np.arange(self.nstim_trials)  
            nstim = self.nstim
            nstim_trials = self.nstim_trials
        mat_width = nstim*nstim_trials
        tbt_sim_mat = np.zeros((mat_width,mat_width))
        count_stim_1 = int(0)
        stim_sim_mean = np.empty((nstim,nstim))
        stim_sim_allmeans = []
        for stim1_idx in range(stim_range[0],stim_range_add):
            count_stim_2 = int(0)
            for stim2_idx in range(stim_range[0], stim_range_add):
                vec1 = self.spike_counts[0:,stim1_idx,trial_range].T
                vec2 = self.spike_counts[0:,stim2_idx,trial_range].T
                stim_sim = sklearn.metrics.pairwise.cosine_similarity(vec1,vec2)
                if stim1_idx == stim2_idx:
                    stim_sim[stim1_idx,stim2_idx] = np.nan
                stim_sim_mean[count_stim_1, count_stim_2] = np.nanmean(stim_sim)
                tbt_sim_mat[0+(nstim_trials*count_stim_1):nstim_trials+(nstim_trials*count_stim_1), 0 + (nstim_trials*count_stim_2):nstim_trials+(nstim_trials*count_stim_2)] = stim_sim
                count_stim_2 += 1   
            count_stim_1 += 1
            stim_sim_allmeans.append(stim_sim_mean)
        cosine_similarity_matrix = {'trial_by_trial':tbt_sim_mat, 'mean_similarity':stim_sim_mean}
        return cosine_similarity_matrix
        
        
        
class trial_aligner: 
    def __init__(self, tsecs, frame_on_time_s, expt_metadata, input_trial_indices = [], trials = 'all', interval = [-.54, 2.54], window_std = 10, binsize = .002, single_spot_stim = False):
        self.tsecs = tsecs
        self.frame_on_time_s = frame_on_time_s
        self.expt_metadata = expt_metadata
        self.interval = interval
        self.window_std = window_std
        if np.array(input_trial_indices).any(): 
            self.input_trial_indices = input_trial_indices
        if single_spot_stim == True:
            all_trials = []
            for block in self.expt_metadata['pixels']:
                all_trials.append(block[0:-1])
            trials = [trial for block in all_trials for trial in block]
            self.expt_metadata['frame_idx'] = trials
            print(len(self.expt_metadata['frame_idx']))
        self.unique_frame_num = len(set(self.expt_metadata['frame_idx']))
        self.cell_aligned_raster = get_cell_aligned_raster(self.tsecs, self.frame_on_time_s, self.interval)
        frame_idx = self.expt_metadata['frame_idx']
        unique_frame_idx = np.unique(frame_idx)
        try:
            l = self.expt_metadata['frame_idx'].tolist()
        except:
            l = self.expt_metadata['frame_idx']
        ul = set(self.expt_metadata['frame_idx'])
        result = sorted([(x, l.count(x)) for x in ul], key=lambda y: y[1])
        self.frame_trial_indices = np.zeros((self.unique_frame_num, result[0][1]))
        for idx_num, idx in enumerate(unique_frame_idx): 
            self.frame_trial_indices[idx_num] = np.where(frame_idx == idx)[0]
        if trials != 'all':
            indices = self.frame_trial_indices[:, trials] 
        else: 
            indices = self.frame_trial_indices
        if np.array(input_trial_indices).any():
            indices = np.array(self.input_trial_indices)
        self.trial_type_raster, self.trial_type_hist, self.trial_type_psth, self.PSTH_timepoints = get_raster_psth(self.cell_aligned_raster, indices, interval = interval, window_std = window_std, binsize = binsize)
    
    def get_cell_aligned_raster(self):
        return self.cell_aligned_raster
    
    def get_frame_trial_indices(self):
        return self.frame_trial_indices
  
    def get_trial_type_raster(self):
        return self.trial_type_raster
    
    def get_trial_type_hist(self):
        return self.trial_type_hist
    
    def get_smoothed_trial_type_hist(self):
        return self.trial_type_smoothed_hist
    
    def get_trial_type_psth(self):
        return self.trial_type_psth
    
    def get_psth_timepoints(self):
        return self.PSTH_timepoints
    
    def get_frame_mean_psth(self):
        frame_mean_psth = np.mean((np.mean(self.trial_type_psth,0)),1)
        if len(frame_mean_psth == self.unique_frame_num):
            return frame_mean_psth
        else:
            print('lenth of array does not align with unique frame number!') 
        
    def get_population_mean_psth(self):
        population_mean_psth = np.mean(np.mean((np.mean(self.trial_type_psth,0)),1),0)
        return population_mean_psth
    
    def get_spotnum_aligned_psth(self): 
        spotnum_indices, _ = get_size_indices(self.expt_metadata)
        psth_by_spotnum = []
        mean_psth_frames_spotnumsorted = []
        mean_psth_by_spotnum = []
        se_psth_by_spotnum = []
        trial_type_psth = np.array(self.trial_type_psth)
        for idx, spotnum in enumerate(spotnum_indices):
            psth_by_spotnum.append(trial_type_psth[0:,spotnum,0:])
            mean_psth_frames_spotnumsorted.append(np.mean(np.mean(np.mean(psth_by_spotnum,1),2),0))
            mean_psth_by_spotnum.append(np.mean(mean_psth_frames_spotnumsorted[idx],0))
            se_psth_by_spotnum.append(np.std(mean_psth_frames_spotnumsorted[idx],0)/np.sqrt(len(mean_psth_frames_spotnumsorted[idx])))
        return mean_psth_by_spotnum, se_psth_by_spotnum, mean_psth_frames_spotnumsorted
    
    def get_sequence_aligned_psth(self, trial_indices): 
        psth_by_sequence = []
        mean_psth_frames_sequencesorted = []
        mean_psth_by_sequence = []
        se_psth_by_sequence = []
        trial_type_psth = np.array(self.trial_type_psth)
        for idx, sequence in enumerate(trial_indices):
            psth_by_sequence.append(trial_type_psth[0:,sequence,0:])
            mean_psth_frames_sequencesorted.append(np.mean(np.mean(np.mean(psth_by_sequence,1),2),0))
            mean_psth_by_sequence.append(np.mean(mean_psth_frames_sequencesorted[idx],0))
            se_psth_by_sequence.append(np.std(mean_psth_frames_sequencesorted[idx],0)/np.sqrt(len(mean_psth_frames_sequencesorted[idx])))
            psth_by_sequence = []  
        return mean_psth_by_sequence, se_psth_by_sequence, mean_psth_frames_sequencesorted
    
class population_analyses:
    def __init__(self, spike_counts_post, ptn_indices):
        self.spike_counts_post = spike_counts_post
        self.ncells, self.nstim, self.trials_per_stim = spike_counts_post.shape
        self.total_trials = self.nstim*self.trials_per_stim
        self.spike_counts_reshaped, self.labels, self.stim_idxs = spike_shaper_2d(spike_counts_post)
        self.ptn_indices = ptn_indices
        
    def PCA_spike_counts(self, scale_data = False, plot = False, n_components = 3):
        pca = PCA(n_components=n_components)
        if scale_data == True:
            scaled_spike_counts = StandardScaler().fit_transform(self.spike_counts_reshaped)
            principalComponents = pca.fit_transform(scaled_spike_counts)
        else:
            principalComponents = pca.fit_transform(self.spike_counts_reshaped)       
        if plot == True:
            fig = plt.figure(figsize = (20,20))
            for num, idx in enumerate(self.ptn_indices):
                ag = fig.add_subplot(1,len(self.ptn_indices),num+1, projection = '3d')
                #ag = fig.add_subplot(1,len(size_indices),num+1)
                for stim_idx, stim in enumerate(self.stim_idxs[idx]):
                    ag.scatter(principalComponents[stim,0], principalComponents[stim,1], principalComponents[stim,2], 'o')
                    #plt.plot(principalComponents[stim,0], principalComponents[stim,1],'o', color = colors[stim_idx])
                    ag.set_xlim([-10,10])
                    ag.set_ylim([-10,10])
                    ag.set_zlim([-10,10])
                    ratio = 1
                    xleft, xright = ag.get_xlim()
                    ybottom, ytop = ag.get_ylim()
                    ag.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
                    right_side = ag. spines["right"]
                    right_side. set_visible(False)
                    top_side = ag. spines["top"]
                    top_side. set_visible(False)

                    plt.subplots_adjust(left=0.125,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.8, 
                                hspace=0.35)
        return principalComponents, self.stim_idxs
    
    def avg_PCA_spike_counts(self, scale_data = False, plot = False, n_components = 3):
        mean_spike_counts = np.mean(self.spike_counts_post,2).T
        pca = PCA(n_components=n_components)
        if scale_data == True:
            scaled_spike_counts = StandardScaler().fit_transform(mean_spike_counts)
            principalComponents = pca.fit_transform(scaled_spike_counts)
        else:
            principalComponents = pca.fit_transform(mean_spike_counts) 
        return principalComponents, self.stim_idxs
    
    def svm_decoder(self, nreps = 200, train_size = .9, test_size = .10):
        label_idx = []
        label_by_sequence = []
        for ptn in self.ptn_indices:
            for idx in ptn:
                label_idx.append(np.where(self.labels == idx))
            label_by_sequence.append(np.ravel(label_idx))
            label_idx = []
        scores = np.empty((len(self.ptn_indices),nreps))
        for ptn_idx, ptn in enumerate(self.ptn_indices):
            for rep in range(nreps):
                labels = np.array(self.labels)
                X_train, X_test, y_train, y_test = train_test_split(self.spike_counts_reshaped[label_by_sequence[ptn_idx]], labels[label_by_sequence[ptn_idx]], train_size = train_size, test_size=test_size) # 70% training and 30% test
                clf = svm.SVC(kernel='linear') # Linear Kernel
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                scores[ptn_idx, rep] = metrics.accuracy_score(y_test, y_pred)
        return scores
    
    def get_labels(self):
        _, labels, _ = spike_shaper_2d(self.spike_counts_post)
        spikes, _, _ = spike_shaper_2d(self.spike_counts_post)
        return spikes, labels

class anova_tuning:
    def __init__(self,sequence_indices, spike_counts):
        self.sequence_indices = sequence_indices
        self.spike_counts = spike_counts
        self.all_tuning = np.empty((np.array(self.sequence_indices).shape[0],self.spike_counts.shape[0]))
        self.labels = np.repeat(np.arange(len(self.sequence_indices[0])),self.spike_counts.shape[2])
        for idx, pattern in enumerate(self.sequence_indices):
            test_counts = self.spike_counts[:,pattern,:]
            for cell_idx, cell in enumerate(test_counts):     
                self.all_tuning[idx, cell_idx] = find_temp_tuning(test_counts[cell_idx,:,:].ravel(),self.labels)
    
    def get_tuning(self):
        return self.all_tuning        
    
    def get_percentage_tuned(self):
        percent_tuned = np.empty(self.all_tuning.shape[0])
        for ptn_num, ptn in enumerate(self.all_tuning):
            percent_tuned[ptn_num] = (len(ptn[~np.isnan(ptn)])/len(ptn))*100
        return percent_tuned
    
    def get_tuning_frequency(self):
        pref_time_frequency = np.empty((np.array(self.sequence_indices).shape[0],len(self.sequence_indices[0])))
        for ptn_num, ptn in enumerate(self.all_tuning):
            index_list = list(np.unique(self.labels))
            occurrence = [len(np.where(ptn==item)[0]) for item in index_list]
            frequency = (occurrence/np.sum(occurrence))*100
            pref_time_frequency[ptn_num] = frequency
        return pref_time_frequency     
    
    def get_tuned_cell_selectivity(self, trial_type_psth, PSTH_timepoints, time_window = .3):
        timepoints = (PSTH_timepoints > 0) & (PSTH_timepoints<.3)
        all_tuned_select_by_ptn = []
        for ptn_idx, ptn in enumerate(self.sequence_indices):
            tuned_cell_idx = ~np.isnan(self.all_tuning[ptn_idx])
            ptn_psth = trial_type_psth[:,ptn,:,:]
            ptn_psth_tuned = ptn_psth[tuned_cell_idx,:,:,:]
            mean_ptn_psth = np.mean(ptn_psth_tuned[:,:,:,timepoints],(2,3))
            max_val_sub_mat = np.repeat(np.max(mean_ptn_psth,1),len(ptn)-1).reshape(mean_ptn_psth.shape[0],len(ptn)-1)
            all_tuned_select_by_ptn.append(np.mean((max_val_sub_mat -  np.sort(mean_ptn_psth,1)[:,0:len(ptn)-1])/(max_val_sub_mat +  np.sort(mean_ptn_psth,1)[:,0:len(ptn)-1]),1))
        all_selectivity = [ptn for all_ptns in all_tuned_select_by_ptn for ptn in all_ptns]
        return all_tuned_select_by_ptn, all_selectivity
    
        
class decoder: 
    def __init__(self, spike_counts, stim_indices, times, stim_type = 'time', experiment_type = 'jitter', use_PCs = 'false', generalize = 'false', generalize_set = [], n_components = 3, shuffle_labels = False):
        self.spikes = spike_counts['post']
        self.stim_indices = stim_indices
        self.indiv_corr = []
        self.indiv_acc = []
        self.fit_rsq = []
        self.popt_all = []
        self.fit_resid = []
        self.eucdist_times = []
        self.eucdist_acc = []
        self.indiv_acc_unsort = []
        self.mutual_info_all = []
        self.corr_vals_all = []
        scores = []
        corr_times = []
        corr_val = []
        corr_times_indiv = []
        diag_indiv = []
        accuracy = []
        eucdist_times = []
        all_times = times
        for idx0, indices in enumerate(self.stim_indices): 
            corr_times_indiv = []
            diag_indiv = []
            eucdist_times = []
            eucdist_spatial_patterns = []
            mutual_info = []
            corr_vals = []
            if experiment_type == 'order':
                times = np.array(all_times)[[indices]]
            else:
                times = np.array(all_times)
            for idx1, time in enumerate(indices):
                for idx2, time2 in enumerate(indices): 
                    if time<time2:
                        spikes = self.spikes[:,[time,time2],:]
                        spike_counts_reshaped, labels, stim_idxs = spike_shaper_2d(spikes)
                        if shuffle_labels == True:
                            random.shuffle(labels)
                        if use_PCs == 'true':
                            pca = PCA(n_components=n_components)
                            scaled_spike_counts = StandardScaler().fit_transform(spike_counts_reshaped)
                            x_train_PCA = pca.fit_transform(scaled_spike_counts)
                            X_train = x_train_PCA
                            y_train = labels
                        else: 
                            X_train = spike_counts_reshaped
                            y_train = labels
                        svm_clf = SVC(kernel = 'linear')
                        svm_clf.fit(X_train, y_train)
                        loo = LeaveOneOut()
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
                        if generalize == 'false':
                            y_train_pred = cross_val_predict(svm_clf, X_train, y_train, cv = loo)
                            conf_mx = confusion_matrix(y_train, y_train_pred)
                        else:
                            spikes_test = spike_counts['post'][:,[generalize_set[idx0][idx1],generalize_set[idx0][idx2]],:]
                            spike_counts_reshaped_test, labels_test, stim_idxs = spike_shaper_2d(spikes_test)
                            X_test = spike_counts_reshaped_test
                            y_test = labels_test
                            y_train_pred = svm_clf.predict(X_test)
                            conf_mx = confusion_matrix(y_test, y_train_pred)
                        conf_mx = conf_mx/self.spikes.shape[2]
                        corr_times.append(np.corrcoef(times[idx1], times[idx2])[0,1])
                        eucdist_times.append(np.linalg.norm(times[idx1]-times[idx2]))
                        accuracy.append(np.mean(np.diag(conf_mx)))
                        corr_times_indiv.append(np.corrcoef(times[idx1], times[idx2])[0,1])
                        diag_indiv.append(np.mean(np.diag(conf_mx)))
                        mutual_info_1 = mutual_info_classif(X_train, y_train)    
                        mutual_info.append(mutual_info_1)
                        mean_counts = np.mean(self.spikes,2)
                        corr_mean = np.corrcoef(mean_counts[:,time],mean_counts[:,time2])
                        corr_vals.append(corr_mean[0,1])
            sort_idx = np.argsort(corr_times_indiv)
            eucdist_sort = np.argsort(eucdist_times)
            corr_times_indiv = np.array(corr_times_indiv)
            diag_indiv = np.array(diag_indiv)
            self.indiv_corr.append(corr_times_indiv[sort_idx])
            self.indiv_acc.append(diag_indiv[sort_idx])
            self.indiv_acc_unsort.append(diag_indiv)
            self.eucdist_times.append(np.array(eucdist_times)[eucdist_sort])
            self.eucdist_acc.append(diag_indiv[eucdist_sort])
            self.mutual_info_all.append(np.array(mutual_info)[sort_idx])
            self.corr_vals_all.append(np.array(corr_vals)[sort_idx])
            if stim_type == 'time':
                popt, pcov = curve_fit(logifunc, 1-corr_times_indiv[sort_idx], diag_indiv[sort_idx])
                residuals = diag_indiv[sort_idx] - logifunc(1-corr_times_indiv[sort_idx], *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((diag_indiv[sort_idx]-np.mean(diag_indiv[sort_idx]))**2)
                r_squared = 1 - (ss_res / ss_tot)
                self.fit_rsq.append(r_squared)
                self.popt_all.append(popt)
                self.fit_resid.append(residuals)
            else:
                pass
    
    def binary_time_decode(self):
        return self.indiv_acc
    
    def return_temporal_correlations(self):
        return self.indiv_corr

    def return_binary_time_decode_unsort(self):
        return self.indiv_acc_unsort
    
    def binary_time_decode_eucdist(self):
        return self.eucdist_times, self.eucdist_acc
    
    def return_corrtime_acc_fits(self):
        return self.popt_all
    
    def return_corrtime_acc_rsq(self):
        return self.fit_rsq
    
    def return_corrtime_acc_resid(self):
        return self.fit_resid
    
    def return_mutual_information(self):
        return self.mutual_info_all
    
    def return_corr_vals(self):
        return self.corr_vals_all
        
        
class metadata_analysis:
    def __init__(self, expt_metadata):
        self.expt_metadata = expt_metadata
        
    def plot_times_raster(self, times_indices = []):
        times = self.expt_metadata['frame_timing_idx']
        times_color = []
        all_times = []
        for time in times:
            for point in time:
                times_color.append([point])
            all_times.append(times_color)
            times_color = []

        NUM_COLORS = 48

        cm = plt.get_cmap('gist_rainbow')
        colorlist = []
        for i in range(NUM_COLORS):
            if i%2 == 0:
                colorlist.append(cm(i//3*3.0/NUM_COLORS))

        i = 0
        for t in all_times:
            plt.eventplot(t, colors = colorlist, lineoffsets=[i]*24, linewidth = 2)
            i += 1
        ax = plt.gca()
        ax.set_yticks([0,1,2,3,4,5])
        ax.set_yticklabels(['1','2','3','4','5','6'])
        ax.set_xticks([0,50,100,150])
        plt.xlim(-1,150)
        plt.ylabel('sequence #')
        plt.xlabel('time (ms)')

        

        
def find_temp_tuning(spikes, labels, alpha = .05):
    # for a given neuron, determine whether it is temporally tuned and compute its preferred temporal pattern. 
    
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import numpy as np
    
    dat = {'spikes': spikes, 'labels': np.uint16(labels).ravel()}
    dat = pd.DataFrame(data = dat)
    dat['labels'] = dat['labels'].astype('category')
    
    # first ANOVA
    mod = ols('spikes ~ labels', data = dat).fit()
    anv = sm.stats.anova_lm(mod)
    pval = anv['PR(>F)'][0]
    
    # if passed, compute tuning index
    if pval < alpha:
        spikemeans = dat.groupby(by = 'labels').mean()
        rvals = spikemeans.values
        cvals = np.uint16(spikemeans.index.values)
        rpref = np.max(rvals)
        pref = np.argmax(rvals)
        cpref = cvals[pref]
        key = np.unique(labels)
        max_tun = key[pref]
    else: 
        max_tun = np.nan
        
    return max_tun
        
def twoptn_rank_sum_test(p1,p2):
    ncells = p1.shape[0]
    auc = np.empty(ncells)
    pvals = np.empty(ncells)
    for cell in range(ncells):
        try:
    # had to get a little creative here to get the ranksums value R used to calculate the P value. R can be derived from the U statistic in the mann-whitneyU test. 
            statistic, p_value = scipy.stats.ranksums(p1[cell],p2[cell])  
            stat, p = scipy.stats.mannwhitneyu(p1[cell], p2[cell], alternative = "two-sided")
            R = stat + (((len(p1[cell])*(len(p1[cell])+1))/2))
            auROC = (R - len(p1[cell])*(len(p1[cell])+1)/2)/(len(p1[cell])*len(p1[cell]));
            pvals[cell] = p_value
            auc[cell] = auROC
        except: 
                pvals[cell] = np.nan
                auc[cell] = np.nan
    return pvals, auc     
    
def logifunc(x,l,c,k):
    return 1 / (1 + c*np.exp(-k*x))

def logifunc_l_free(x,l,c,k):
    return l / (1 + c*np.exp(-k*x))
    
    
### shape spike counts (ncells x nstim x ntrials) into 2d (ncells x ntrials) 
def spike_shaper_2d(spike_counts):
    ncells, nstim, trials_per_stim = spike_counts.shape
    total_trials = nstim*trials_per_stim
    spike_counts_reshaped = np.empty((total_trials,ncells))
    stim_idx = []
    stim_idxs = []
    labels = np.zeros(spike_counts_reshaped.shape[0])
    count = 0
    label = 0
    for stim in range(nstim):
        for trial in range(trials_per_stim):
            spike_counts_reshaped[count,:] = spike_counts[0:,stim,trial]
            stim_idx.append(count)
            labels[count] = (label)
            count += 1
        stim_idxs.append(stim_idx)
        stim_idx = []
        label+=1
    stim_idxs = np.array(stim_idxs)
    return spike_counts_reshaped, labels, stim_idxs
    
def mean_neg_cross(x):
    N = 500
    xsmooth = np.convolve(x, np.ones((N,))/N, mode='valid')
    mean_trace = np.mean(x)/1.1
    neg_cross_idxs = []
    for idx, val in enumerate(xsmooth[0:-1]): 
        if (val > mean_trace) & (xsmooth[idx+1] < mean_trace):
            neg_cross_idxs.append(idx)
    trigger_idx = len(x)/2
    diff_trigger_idx = np.abs(np.array(neg_cross_idxs)-trigger_idx)
    trigger_point = np.argmin(diff_trigger_idx)
    pre_breath_idx = neg_cross_idxs[trigger_point-1]+N
    post_breath_idx = neg_cross_idxs[trigger_point+1]+N
    pre_breath_trig_diff = trigger_idx - pre_breath_idx
    post_breath_trig_diff = trigger_idx - post_breath_idx
    neg_cross_idxs_aligned = np.array(neg_cross_idxs) - trigger_idx
    inh_trigger_idx = np.argmin(abs(neg_cross_idxs_aligned))
    neg_cross_idxs_aligned[inh_trigger_idx] = 0
    return pre_breath_trig_diff, pre_breath_idx, post_breath_trig_diff, post_breath_idx, neg_cross_idxs, neg_cross_idxs_aligned  

def get_size_indices(expt_metadata):
    frame_ID_idx = expt_metadata['frame_ID_idx']
    spot_numbers = expt_metadata['spot_size']
    spotnum_indices = []
    indices = []
    spotnum_pix = []
    pixels_by_spotnum = []
    for spotnum in spot_numbers: 
        for idx, frame in enumerate(frame_ID_idx): 
            if len(frame) == spotnum:
                indices.append(idx)
                spotnum_pix.append(frame_ID_idx[idx])
        spotnum_indices.append(indices)
        pixels_by_spotnum.append(spotnum_pix)
        indices = []
        spotnum_pix = []
    return spotnum_indices, pixels_by_spotnum
        

def get_cell_aligned_raster(tsecs, frame_on_time_s, interval = [-1, 2]): #interval was [-.54, 2.54]
    trial_align_raster = []
    cell_align_raster = []
    for cell in tsecs:
        for trial in frame_on_time_s:
            trial_align = cell - trial
            trial_align_raster.append(trial_align[(interval[0] < trial_align) & (interval[1] > trial_align)])
        cell_align_raster.append(trial_align_raster)
        trial_align_raster = []
    return cell_align_raster
        
def get_raster_psth(cell_align_raster, frame_trial_indices, interval = [-1, 2], window_std = 10, binsize = .002):
    x = np.arange(interval[0],interval[1],binsize) #2 ms bins, 3 seconds of data
    histlength = len(np.histogram(cell_align_raster[0][0], range = (interval[0], interval[1]), bins = int((abs(interval[0])+interval[1])/binsize))[0])
    cellraster_by_trialtype = []
    cellraster_trialtype = []
    trial_type_raster = []
    trial_type_psth = np.empty((len(cell_align_raster), len(frame_trial_indices), len(frame_trial_indices[0]), histlength))
    trial_type_hist = np.empty((len(cell_align_raster), len(frame_trial_indices), len(frame_trial_indices[0]), histlength))
    trial_type_raster = []
    for cell_idx, cell in enumerate(cell_align_raster):
        for trial_type_idx, trial_type in enumerate(frame_trial_indices.astype(int)): 
            for entry_idx, entry in enumerate(trial_type):
                cellraster_by_trialtype.append(cell[entry])
                cellhist = np.histogram(cell[entry], range = (interval[0], interval[1]), bins = int((abs(interval[0])+interval[1])/binsize))[0]
                trial_type_hist[cell_idx, trial_type_idx, entry_idx, 0:] = cellhist
                inst_rate = cellhist*((abs(interval[0])+interval[1]/binsize)/(abs(interval[0])+interval[1]))
                inst_rate = scipy.ndimage.gaussian_filter1d(inst_rate, (window_std/1000)/.002) 
                trial_type_psth[cell_idx, trial_type_idx, entry_idx, 0:] = inst_rate
            cellraster_trialtype.append(cellraster_by_trialtype)
            cellraster_by_trialtype = []
        trial_type_raster.append(cellraster_trialtype)
        cellraster_trialtype = []
    PSTH_timepoints= np.arange(interval[0],interval[1],(abs(interval[0]) + interval[1])/((abs(interval[0])+interval[1])/binsize))
    return trial_type_raster, trial_type_hist, trial_type_psth, PSTH_timepoints


def get_raster_psth_old(cell_align_raster, frame_trial_indices, interval = [-.54, 2.54], window_std = 10, binsize = .002): # old version used before 8/30/24
    window_size = np.arange(-1000,1001,1)
    window_std = window_std
    window = scipy.stats.norm.pdf(window_size, 0, window_std)
    window /= window.sum()
    window = window[750:1250]
    histlength = len(np.histogram(cell_align_raster[0][0], range = (interval[0], interval[1]), bins = int((abs(interval[0])+interval[1])/binsize))[0])
    cellraster_by_trialtype = []
    cellraster_trialtype = []
    trial_type_raster = []
    trial_type_psth = np.empty((len(cell_align_raster), len(frame_trial_indices), len(frame_trial_indices[0]), histlength))
    trial_type_hist = np.empty((len(cell_align_raster), len(frame_trial_indices), len(frame_trial_indices[0]), histlength))
    trial_type_smoothed_hist = np.empty((len(cell_align_raster), len(frame_trial_indices), len(frame_trial_indices[0]), histlength))
    trial_type_raster = []
    for cell_idx, cell in enumerate(cell_align_raster):
        for trial_type_idx, trial_type in enumerate(frame_trial_indices.astype(int)): 
            for entry_idx, entry in enumerate(trial_type):
                cellraster_by_trialtype.append(cell[entry])
                cellhist = np.histogram(cell[entry], range = (interval[0], interval[1]), bins = int((abs(interval[0])+interval[1])/binsize))[0]
                trial_type_hist[cell_idx, trial_type_idx, entry_idx, 0:] = cellhist
                inst_rate = cellhist*((abs(interval[0])+interval[1]/binsize)/(abs(interval[0])+interval[1]))
                inst_rate = np.convolve(window, inst_rate, mode='same')
                convolve_hist = np.convolve(window, cellhist, mode='same')
                trial_type_psth[cell_idx, trial_type_idx, entry_idx, 0:] = inst_rate
                trial_type_smoothed_hist[cell_idx, trial_type_idx, entry_idx, 0:] = convolve_hist
            cellraster_trialtype.append(cellraster_by_trialtype)
            cellraster_by_trialtype = []
        trial_type_raster.append(cellraster_trialtype)
        cellraster_trialtype = []
    PSTH_timepoints= np.arange(interval[0],interval[1],(abs(interval[0]) + interval[1])/((abs(interval[0])+interval[1])/binsize))
    return trial_type_raster, trial_type_hist, trial_type_psth, trial_type_smoothed_hist, PSTH_timepoints


# def get_raster_psth(cell_align_raster, frame_trial_indices, interval = [-.54, 2.54], window_std = 10):
#     window_size = np.arange(-1000,1001,1)
#     window_std = 10
#     window = scipy.stats.norm.pdf(window_size, 0, window_std)
#     window /= window.sum()
#     window = window[750:1250]
#     cellraster_by_trialtype = []
#     cellraster_trialtype = []
#     trial_type_raster = []
#     cellhist_by_trialtype = []
#     cellhist_trialtype = []
#     trial_type_hist = []
#     cellpsth_by_trialtype = []
#     cellpsth_trialtype = []
#     trial_type_psth = []
#     for cell in cell_align_raster:
#         for trial_type in frame_trial_indices: 
#             for entry in trial_type:
#                 cellraster_by_trialtype.append(cell[entry])
#                 cellhist = np.histogram(cell[entry], range = (interval[0], interval[1]), bins = int((abs(interval[0])+interval[1])/.002))[0]
#                 cellhist_by_trialtype.append(cellhist)
#                 inst_rate = cellhist*((abs(interval[0])+interval[1]/.002)/(abs(interval[0])+interval[1]))
#                 inst_rate = np.convolve(window, inst_rate, mode='same')
#                 cellpsth_by_trialtype.append(inst_rate)
#             cellraster_trialtype.append(cellraster_by_trialtype)
#             cellraster_by_trialtype = []
#             cellhist_trialtype.append(cellhist_by_trialtype)
#             cellhist_by_trialtype = []
#             cellpsth_trialtype.append(cellpsth_by_trialtype)
#             cellpsth_by_trialtype = []
#         trial_type_raster.append(cellraster_trialtype)
#         cellraster_trialtype = []
#         trial_type_hist.append(cellhist_trialtype)
#         cellhist_trialtype = []
#         trial_type_psth.append(cellpsth_trialtype)
#         cellpsth_trialtype = []
#     PSTH_timepoints= np.arange(interval[0],interval[1],(abs(interval[0]) + interval[1])/((abs(interval[0])+interval[1])/.002))
#     return trial_type_raster, trial_type_hist, trial_type_psth, PSTH_timepoints

# fit an exponential curve to psth decay. 
def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

# Reads the HDF5 file output by Spyking Circus and returns spike time and cluster index information 
def get_spiketimes_s(spike_file):
    units = h5py.File(spike_file, 'r')
    spiketimes = units['spiketimes']
    dataset_names = [n for n in spiketimes.keys()];
    tsecs = []
    clusternumber = []
    spike_index = []
    for number, unit in enumerate(dataset_names):
        tsecs.append(np.array(spiketimes[unit][:,:])/30000);
        spike_index.append(np.array(spiketimes[unit][:,:]));
        clusternumber.append(int(re.findall('\d+', unit)[0])) #finds the digits in the name 
    return tsecs, spike_index, clusternumber

def get_frameontime_sequence(trigger_array, expt_metadata, fs = 30000):
    fs = 30000
    trigger_array[trigger_array > 20000] = 30000
    trigger_array[trigger_array <= 20000] = 0
    test = np.where(np.diff(trigger_array) == 30000)
    diff_array = np.diff(trigger_array)
    trig_times = np.where(diff_array == 30000)[0]
    frame_change_timediff = np.diff(trig_times)
    frame_change_indices = np.unique(np.diff(trig_times))
    frame_on = np.where(frame_change_timediff >5000) #allow for a 1 microsecond jitter
    frame_on_time_s = trig_times[frame_on[0]+1]/fs
    frame_on_time_idx = trig_times[frame_on[0]+1]                 
    nframes = len(expt_metadata['frame_idx'])
    frame_on_time_s = np.insert(frame_on_time_s,0, trig_times[0]/fs)
    frame_on_time_idx = np.insert(frame_on_time_idx,0, trig_times[0])
    if nframes == len(frame_on_time_s):
        print('frames good')
        return frame_on_time_s, frame_on_time_idx
    else:
        frame_diff = np.diff(frame_on_time_s)
        frame_on_time_s = np.delete(frame_on_time_s,np.where(frame_diff<3)[0],0)
        frame_on_time_idx = np.delete(frame_on_time_idx,np.where(frame_diff<3)[0],0)
        #frame_on_time_s = np.insert(frame_on_time_s,0, trig_times[0]/fs)
        #frame_on_time_idx = np.insert(frame_on_time_idx,0, trig_times[0])
        print('Metadata frames and trigger frames do not align! Check triggers!')
        return frame_on_time_s, frame_on_time_idx
 

def get_frameontimes(trigger_array, expt_metadata, fs = 30000):
    trigger_array[trigger_array > 20000] = 30000
    trigger_array[trigger_array <= 20000] = 0
    trigger_diff = np.unique(np.diff(trigger_array))
    diff_array = np.diff(trigger_array)
    trig_times = np.where(diff_array == trigger_diff[0])[0]
    frame_change_timediff = np.diff(trig_times)
    frame_change_indices = np.unique(np.diff(trig_times))
    frame_on = np.where(frame_change_timediff <= frame_change_indices[0] + 30) #allow for a 1 microsecond jitter
    frame_on_time_s = trig_times[frame_on]/fs
    frame_on_time_idx = trig_times[frame_on]                    
    nframes = len(expt_metadata['frame_idx'])
    if nframes == len(frame_on_time_s):
        return frame_on_time_s, frame_on_time_idx
    else:
        print('Metadata frames and trigger frames do not align! Check triggers!')

# Allows you to load the events files -two triggers + breath- as numpy arrays 
def get_events(event_file):
    events_file = open(event_file, 'rb')
    events = events_file.read()
    event_array = np.fromstring(events, dtype='int16')
    events_file.close()
    return event_array

# Gets trigger on time in seconds from the loaded mightex trigger array 
def get_frameontime_s(mtrigger_array):
    mtrigger_array[mtrigger_array < 20000] = 0
    mtrigger_array[mtrigger_array >= 20000] = 30000
    trigger_diff = np.unique(np.diff(mtrigger_array))
    diff_array = np.diff(mtrigger_array)
    trig_times = np.where(diff_array == trigger_diff[0])[0]
    frame_change_timediff = np.diff(trig_times)
    frame_change_indices = np.unique(np.diff(trig_times))
    frame_on = np.where((frame_change_timediff == frame_change_indices[0]) | (frame_change_timediff == frame_change_indices[1]))
    frame_on_time_s = trig_times[frame_on]/30000
    return frame_on_time_s

# Makes an n_spot dimensional array with each spot on the camera
def makeDMDArrayPolygon(mask, spot_size, pix_scale = 1.92, cam_h = int(2200), cam_w = int(2688)):
    spot_size = int(round(spot_size/pix_scale))
    ret,thresh = cv2.threshold(mask, 254,255,cv2.THRESH_BINARY)
    thresh = thresh.astype('uint8')
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = contours[0]
    x_bbox,y_bbox,w_bbox,h_bbox = cv2.boundingRect(cnt)
    real_bbox = [x_bbox,y_bbox,w_bbox+x_bbox, h_bbox+y_bbox]
    y_pixels_length = real_bbox[-1] - real_bbox[1]
    x_pixels_length = real_bbox[-2] - real_bbox[0]
    y_pixels_length = y_pixels_length - (y_pixels_length%spot_size)
    x_pixels_length = x_pixels_length - (x_pixels_length%spot_size)
    y_pixels_start = real_bbox[1]
    x_pixels_start = real_bbox[0]
    grid_indices_x = []
    grid_indices_y = []
    spot_nums = [int(x_pixels_length/spot_size), int(y_pixels_length/spot_size)]
    for count in range(0,spot_nums[0]+1):
        grid_indices_x.append(x_pixels_start + count*spot_size) 
    for count in range(0,spot_nums[1]+1):
        grid_indices_y.append(y_pixels_start + count*spot_size) 
    cam_array = []
    pixmap = np.zeros((cam_h, cam_w))
    fullmap = np.zeros((cam_h, cam_w))
    for xidx in range(0, len(grid_indices_x)-1):
        for yidx in range(0, len(grid_indices_y)-1):
            if (len(np.unique(mask[grid_indices_y[yidx]:grid_indices_y[yidx+1],grid_indices_x[xidx]:grid_indices_x[xidx+1]])) == 1) and (np.unique(mask[grid_indices_y[yidx]:grid_indices_y[yidx+1],grid_indices_x[xidx]:grid_indices_x[xidx+1]])[-1] == 255.0):
                pixmap[grid_indices_y[yidx]:grid_indices_y[yidx+1],grid_indices_x[xidx]:grid_indices_x[xidx+1]]= 255
                fullmap[grid_indices_y[yidx]:grid_indices_y[yidx+1],grid_indices_x[xidx]:grid_indices_x[xidx+1]]=255
                cam_array.append(pixmap)
            pixmap = np.zeros((int(2200), int(2688)))
    return cam_array

def axis_fixer(ratio = 1, size = 15, axis_width = 1.822):
    plt.rcParams.update({'font.size': size})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams.update({'font.family':'arial'})
    ag = plt.gca()
    right_side = ag. spines["right"]
    right_side. set_visible(False)
    top_side = ag. spines["top"]
    top_side. set_visible(False)
    ratio = ratio
    xleft, xright = ag.get_xlim()
    ybottom, ytop = ag.get_ylim()
    for axis in ['bottom','left']:
        ag.spines[axis].set_linewidth(axis_width) # was .4964
    ag.tick_params(width=axis_width)
    ag.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.subplots_adjust(left=0.125,
                bottom=0.1, 
                right=0.9, 
                top=0.9, 
                wspace=0.8, 
                hspace=0.35)
    
    
def counts(cell_align_raster, mod_breath_time_1 = 0, mod_breath_time_2 = .3): 
    ncells = len(cell_align_raster)
    ntrials = len(cell_align_raster[0])
    pre_inh_spike_count = np.zeros((ncells, ntrials))
    post_inh_spike_count = np.zeros((ncells, ntrials))
    pvals = np.zeros((ncells))
    auc = np.zeros((ncells))
    for cell_idx, cell in enumerate(cell_align_raster):
        for trial_idx, trial in enumerate(cell):
                post_inh_spike_count[cell_idx, trial_idx] = len(trial[(trial >= mod_breath_time_1) & (trial <= mod_breath_time_2)])
    return post_inh_spike_count


