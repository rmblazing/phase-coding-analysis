#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy
import matplotlib.gridspec as gridspec
from scipy import signal
import scipy.stats
from PIL import Image
import PIL.ImageOps    
import os
from python_pattern_stim_analysis import PGanalysis
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import importlib
import h5py
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import warnings
import random

class extract_data:
    def __init__(self, datafile, mtrigger_file, resp_file, metadata_file, laser_trigger_file, optostim = False):
        
        prep_data = PGanalysis.data_prepper(datafile, mtrigger_file, resp_file, metadata_file, laser_trigger_file_path = laser_trigger_file, expt_type = 'sequence')

        tsecs = prep_data.get_spikes()
        frame_on_time_s = prep_data.get_spiketimes()
        frame_on_time_idx = prep_data.get_spikeidx()
        self.expt_metadata = prep_data.get_metadata()
        resp_array = prep_data.get_resp_array()
        #change indexing to opto indexing
        if optostim == True:
            frame_trial_indices = prep_data.get_frame_trial_indices_opto()
        else:
            frame_trial_indices = prep_data.get_frame_trial_indices()
        align_trials = PGanalysis.trial_aligner(tsecs, frame_on_time_s, self.expt_metadata, frame_trial_indices)
        cell_align_raster = align_trials.get_cell_aligned_raster()
        #frame_trial_indices = align_trials.get_frame_trial_indices()
        self.trial_type_psth = align_trials.get_trial_type_psth()
        self.trial_type_raster = align_trials.get_trial_type_raster()
        self.PSTH_timepoints = align_trials.get_psth_timepoints()

        breath_times_obj = PGanalysis.breath_stats(resp_array, frame_on_time_idx, self.expt_metadata, frame_trial_indices, interval = [-5, 5])
        breath_times_pre_false = np.zeros(len(frame_on_time_s))-.5
        breath_times_post_false = np.zeros(len(frame_on_time_s))+.5
        self.pre_breath_time_shaped = breath_times_obj.align_inh_by_trial_type(breath_times_pre_false)
        self.post_breath_time_shaped = breath_times_obj.align_inh_by_trial_type(breath_times_post_false)
        
    def save_expt_data(self, save_file_path, sequence_timing_indices, sequence_timing_indices_controls = [], blank_control_index = 32, window_start = -.05, window_end = .3): 

        corr_stats = PGanalysis.response_stats(self.trial_type_raster, self.expt_metadata, self.pre_breath_time_shaped, self.post_breath_time_shaped, mod_time = True, mod_breath_time_2 = window_end, pattern_type = 'sequence')
        spike_counts = corr_stats.get_resp_aligned_spike_counts()
        blank = spike_counts['post'][:,blank_control_index]
        pvals, auroc = corr_stats.do_rank_sum_test_toblank(blank)
        prop_responsive = corr_stats.get_response_props([sequence_timing_indices])
        all_response_props = prop_responsive['all_responses']
        prop_activated = np.zeros(all_response_props.shape)
        prop_activated[all_response_props>0] = 1
        prop_suppressed = np.zeros(all_response_props.shape)
        prop_suppressed[all_response_props<0] = 1

        p = (self.PSTH_timepoints>window_start) & (self.PSTH_timepoints<window_end)
        p_baseline = ((self.PSTH_timepoints>(-window_end+window_start)) & (self.PSTH_timepoints<0))
        psth_on = np.mean(self.trial_type_psth[:,:,:,p],2)
        psth_baseline = np.mean(self.trial_type_psth[:,:,:,p_baseline],2)

        resp_cells = np.where(np.sum(prop_activated[:,sequence_timing_indices],1)>0)[0]

        if np.array(sequence_timing_indices_controls).any():
            psth_on = psth_on[:,sequence_timing_indices_controls,:]
            psth_baseline = psth_baseline[:,sequence_timing_indices_controls,:]
            psth_on_activated = psth_on[resp_cells,:,:]
            psth_on_activated_baseline = psth_baseline[resp_cells,:,:]
            prop_activated = prop_activated[:,sequence_timing_indices_controls]
            prop_suppressed = prop_suppressed[:,sequence_timing_indices_controls]
            trial_type_raster = np.array(self.trial_type_raster)[:,sequence_timing_indices_controls,:]
            trial_type_psth = self.trial_type_psth[:,sequence_timing_indices_controls,:,:]
            print('spot pair data extracted')
            print(sequence_timing_indices_controls.shape)
            print(prop_activated.shape)
            print(prop_suppressed.shape)
        else: 
            psth_on = psth_on[:,sequence_timing_indices,:]
            psth_on_activated = psth_on[resp_cells,:,:]
            psth_on_activated_baseline = psth_baseline[resp_cells,:,:]
            trial_type_raster = np.array(self.trial_type_raster)[:,sequence_timing_indices,:]
            trial_type_psth = self.trial_type_psth[:,sequence_timing_indices,:,:]
            
        rec_data = {'pvals':pvals, 'auroc':auroc, 'spike_counts':spike_counts['post'], 'psth_all': trial_type_psth, 'psth_on':psth_on, 'psth_on_activated':psth_on_activated, 'psth_on_activated_baseline':psth_on_activated_baseline, 'resp_cell_indices':resp_cells, 'trial_type_psth':trial_type_psth, 'PSTH_timepoints':self.PSTH_timepoints, 'trial_type_raster':trial_type_raster, 'expt_metadata':self.expt_metadata, 'prop_activated':prop_activated, 'prop_suppressed':prop_suppressed}

        np.save(save_file_path, [rec_data])

        print('data extraction complete, data file saved')

