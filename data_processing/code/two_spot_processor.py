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
import scipy.stats as st
import PGanalysis


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
        
''' two-spot plotting functions'''

def plot_individual_spot_tuning(cell, baseline_sub_psth_by_trial, n_trials, n_latencies_single_spot, filt_width = 5, font_size = 15, ylim=[-10,20], y_step = 10):
    '''
    plot the tuning curves for individual spots as a function of stimulation latency relative to inhalation.
    
    parameters
    ----------
    cell: index of cell to plot
    baseline_sub_psth_by_trial: the trial by trial baseline subtracted psth matrix. Should be n_cells x n_stim x n_trials
    n_trials: number of trials per stimulus
    n_latencies_single_spot: the number of single spot stimulation latencies (should be 15 unique latencies)
    
    outputs
    -------
    plots the tuning curves for spots A and B as a function of latency relative to inhalation for a single cell
    '''
    
    # spot A indices: where spot A was stimulated alone at different intervals relative to inhalation
    spot_A_indices = np.arange(30,45) # hard-coded. 30-45 are spot A alone indices. 
    spot_A_psths = baseline_sub_psth_by_trial[cell,spot_A_indices,:]
    
    # the single spot intervals
    x_single_spot = np.arange(0,75,5)
   
    # initialize a matrix to store smoothed phase-tuning curves for spot A
    mov_mean_spotA = np.empty((n_trials, n_latencies_single_spot))
    for trial in range(n_trials):
        y = spot_A_psths[:,trial]
        mov_mean_y = np.convolve(y, np.ones(filt_width)/filt_width, mode='same')
        mov_mean_spotA[trial,:] = mov_mean_y

    # calculate the 95% confidence interval for change in firing rate for each trial 
    mean_stim_respA = np.mean(mov_mean_spotA,0)
    CI_A = np.empty(n_latencies_single_spot)
    for stim in range(n_latencies_single_spot):
        t = st.t.interval(alpha=0.95, df=len(mov_mean_spotA[:,stim])-1, loc=np.mean(mov_mean_spotA[:,stim]), scale=st.sem(mov_mean_spotA[:,stim])) 
        CI_A[stim] = (np.abs(mean_stim_respA[stim]-t[0])).T
    
    # plot the tuning curves for spot A  
    plt.plot(x_single_spot, np.mean(mov_mean_spotA,0), '-o', color = 'navy', linewidth = .5)
    plt.errorbar(x_single_spot, np.mean(mov_mean_spotA,0), yerr = CI_A, color = 'b', linewidth = .5)

    # spot B indices: where spot B was stimulated alone at different intervals relative to inhalation
    spot_B_indices = np.arange(45,60) # hard-coded. 45-60 are spot B alone indices. 
    spot_B_psths = baseline_sub_psth_by_trial[cell,spot_B_indices,:]
   
    # initialize a matrix to store smoothed phase-tuning curves for spot B
    mov_mean_spotB = np.empty((n_trials, n_latencies_single_spot))
    for trial in range(n_trials):
        y = spot_B_psths[:,trial]
        mov_mean_y = np.convolve(y, np.ones(filt_width)/filt_width, mode='same')
        mov_mean_spotB[trial,:] = mov_mean_y

    # calculate the 95% confidence interval for change in firing rate for each trial 
    mean_stim_respB = np.mean(mov_mean_spotB,0)
    CI_B = np.empty(n_latencies_single_spot)
    for stim in range(n_latencies_single_spot):
        t = st.t.interval(alpha=0.95, df=len(mov_mean_spotB[:,stim])-1, loc=np.mean(mov_mean_spotB[:,stim]), scale=st.sem(mov_mean_spotB[:,stim])) 
        CI_B[stim] = (np.abs(mean_stim_respB[stim]-t[0])).T
    
    # plot the tuning curves for spot B 
    plt.plot(x_single_spot, np.mean(mov_mean_spotB,0), '-o', color = 'firebrick', linewidth = .5)
    plt.errorbar(x_single_spot, np.mean(mov_mean_spotB,0), yerr = CI_B, color = 'firebrick', linewidth = .5)
  
    # format plot according to parameters 
    plt.ylim(ylim[0],ylim[1])
    yticks = np.arange(ylim[0],ylim[1]+y_step, y_step)
    plt.yticks(yticks)
    plt.xticks([0,35,70])
    plt.ylabel('$\Delta$ firing rate (Hz)')
    plt.xlabel('Time from \n inhalation (ms)')
    PGanalysis.axis_fixer(ratio = 1, size = font_size)
    

def plot_real_and_predicted_curves(cell, baseline_sub_psth_by_trial, two_spot_indices, single_spot_indices, filt_width = 5, font_size = 15, ylim = [-10,20], y_step = 10, return_outputs = False, plot = True):
    '''
    plot the tuning curves for individual spots as a function of stimulation latency relative to inhalation.
    
    parameters
    ----------
    cell: index of cell to plot
    baseline_sub_psth_by_trial: the trial by trial baseline subtracted psth matrix. Should be n_cells x n_stim x n_trials
    two_spot_indices: the indices of the two-spot trials, sorted from A-->B to B-->A
    single_spot_indices: indices of the single spot trials sorted from A-->B
    
    outputs
    -------
    plots the observed tuning curve for spots A and B presented together, as well as the curve predicted from the sum of the two spots for a single cell
    '''
    
    x_two_spot = np.arange(-70,75,5)
    filt_width = 5
    
    # Here, we will create the expected tuning curves from summing the responses to A and B at each position in the two spot stimulation paradigm. 
    
    # index 30 is where spot A is presented alone at the onset of inhalation. 
    A_alone_inhalation = 30
    
    # index 45 is where spot B is presented alone at the onset of inhalation.
    B_alone_inhalation = 45
    
    # now, make an array of indices for spot A and B for each delta t condition
    first_spot_indices = np.concatenate((np.zeros(14)+A_alone_inhalation, np.zeros(15)+B_alone_inhalation)).astype(int)

    # now, add the responses to A and B at the start of inhalation to the responses at each delta t condition 
    single_spot_responses_move = baseline_sub_psth_by_trial[cell,:]
    
    # flip the single spot indices so B is paired with A and A with B
    single_spot_responses_move = single_spot_responses_move[np.flipud(single_spot_indices),:]
    
    single_spot_responses_static = baseline_sub_psth_by_trial[cell,:,:]
    single_spot_responses_static = single_spot_responses_static[first_spot_indices,:]
    
    # sum the responses for the moving (delta t) and static (presented at inhalation onset) spots
    predicted_two_spot_responses = single_spot_responses_move+single_spot_responses_static

    # get the observed two spot responses for the cell
    observed_two_spot_responses = baseline_sub_psth_by_trial[cell,:,:]
    observed_two_spot_responses = observed_two_spot_responses[two_spot_indices,:]
    
    # initialize arrays to store observed and predicted curves 
    n_trials = baseline_sub_psth_by_trial.shape[2]           
    all_mov_mean_y_observed = np.empty((n_trials,len(two_spot_indices)))
    all_mov_mean_y_predicted = np.empty((n_trials,len(two_spot_indices)))

    # for each trial, compute and smooth the tuning curve                                     
    for trial in range(n_trials):
        y = observed_two_spot_responses[:,trial]
        y_predicted = predicted_two_spot_responses[:,trial]
        mov_mean_y = np.convolve(y, np.ones(filt_width)/filt_width, mode='same')
        mov_mean_y_predicted = np.convolve(y_predicted, np.ones(filt_width)/filt_width, mode='same')

        all_mov_mean_y_observed[trial,:] = mov_mean_y
        all_mov_mean_y_predicted[trial,:] = mov_mean_y_predicted

    # get the mean tuning curve across trials for observed and predicted responses                                     
    mean_tuning_observed = np.mean(all_mov_mean_y_observed,0)
    mean_tuning_predicted = np.mean(all_mov_mean_y_predicted,0)
     
    # calculate the 95% confidence interval for tuning across trials for each delta t condition                                   
    CI = np.empty(len(two_spot_indices))
    CI_predicted = np.empty(len(two_spot_indices))
    for stim in range(len(two_spot_indices)):
        # get the confidence interval for the observed curves 
        data = all_mov_mean_y_observed[:,stim]
        t = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
        CI[stim] = np.abs(mean_tuning_observed[stim]-t[0])
                                        
        # get the confidence interval for the predicted curves                  
        data_predicted = all_mov_mean_y_predicted[:,stim]
        t_predicted = st.t.interval(alpha=0.95, df=len(data_predicted)-1, loc=np.mean(data_predicted), scale=st.sem(data_predicted)) 
        CI_predicted[stim] = np.abs(mean_tuning_predicted[stim]-t_predicted[0])

    if plot == True:                                    
        # plot the observed tuning curve                                    
        plt.plot(x_two_spot, mean_tuning_observed, '-ko', linewidth = .5)
        plt.errorbar(x_two_spot, mean_tuning_observed, yerr = CI, color = 'k', linewidth = .5)

        # plot the predicted tuning curve 
        plt.plot(x_two_spot, mean_tuning_predicted, '-mo', linewidth = .5)
        plt.errorbar(x_two_spot, mean_tuning_predicted, yerr = CI_predicted, color = 'purple', linewidth = .5)

        # format plot according to parameters
        plt.ylim(ylim[0],ylim[1])
        plt.xticks([-70,-35,0,35,70])
        yticks = np.arange(ylim[0],ylim[1]+y_step, y_step)
        plt.yticks(yticks)
        plt.ylabel('$\Delta$ firing rate (Hz)')
        plt.xlabel('$t_{a}-t_{b}$ (ms)')
        plt.title(str(cell))
        PGanalysis.axis_fixer(ratio = 1, size = font_size)
    
    if return_outputs == True:
        return mean_tuning_observed, mean_tuning_predicted, CI, CI_predicted