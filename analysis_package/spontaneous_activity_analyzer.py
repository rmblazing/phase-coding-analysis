#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import kruskal
import pickle as pkl
import random
import PGanalysis
import phase_analysis


class Spontaneous_activity_analysis:
    def __init__(self, datafile_path, instantaneous_phase_path, mtrigger_array_path, phase_stats_path, phase_tuning_path, bins = np.arange(0,np.radians(360)+np.radians(18), np.radians(18)), fs = 30000, expt_type = []):
        
        self.bins = bins
        
        # load the spiking data 
        self.tsecs = phase_analysis.get_tsecs(datafile_path)
        
        # load the hilbert-filtered instantaneous phase data 
        self.instantaneous_phase_corrected = np.load(instantaneous_phase_path, allow_pickle = True)
        
        # load the mightex trigger events 
        self.mtrigger_array = PGanalysis.get_events(mtrigger_array_path)
            
        # load the phase stats data 
        self.phase_stats = np.load(phase_stats_path, allow_pickle = True)[0]
        
        # load the path to the phase tuning data 
        self.phase_tuning = np.load(phase_tuning_path, allow_pickle = True)[0]
        
        # Because stimulation is on for most of the experiment, we want to get the spontaneous data from the end of the experiment.
        # Most experiments have 10-30 minutes of activity recorded without stimulation. 
        frame_on_time_s = phase_analysis.get_frame_on_time_s(self.mtrigger_array)
        self.last_frame_idx = (frame_on_time_s[-1]*fs).astype(int)
        expt_len = len(self.mtrigger_array)
        
        # for naris occlusion experiments, we need to get spontaneous data recorded in the middle of the session. 
        if expt_type == 'contra_occlusion':
            # get the last trial recorded in the contra occlusion condition
            last_contra_trial = int(len(frame_on_time_s)/2)-1
            # get the next 10 minutes of data after this trial for instantaneous phase, triggers, and spikes
            self.instantaneous_phase_corrected = self.instantaneous_phase_corrected[0:int(frame_on_time_s[last_contra_trial]*fs) + fs*60*10]
            self.mtrigger_array = self.mtrigger_array[0:int(frame_on_time_s[last_contra_trial]*fs) + fs*60*10]
            new_tsecs = []
            for cell in self.tsecs:
                new_tsecs.append(cell[cell<(frame_on_time_s[last_contra_trial] + 60*10)])
            self.tsecs = new_tsecs
            frame_on_time_s = phase_analysis.get_frame_on_time_s(self.mtrigger_array)
            self.last_frame_idx = (frame_on_time_s[last_contra_trial]*fs).astype(int)
            expt_len = len(self.mtrigger_array)
            print('spontaneous activity recorded for ' + str(np.round((expt_len-self.last_frame_idx)/fs/60,2)) + ' minutes')

        else:                    
            print('spontaneous activity recorded for ' + str(np.round((expt_len-self.last_frame_idx)/fs/60,2)) + ' minutes')
    
    ''' Calculate a bootstrapped distribution to determine whether cells are phase tuned. Use the respiration-shuffling method described in Fukunaga et al. 2012'''
    def get_phase_locking_bootstrap(self, bins, n_shuffs = 100, fs = 30000):
        # get each individual cycle and separate out into a list
        inhalations = phase_analysis.get_inhalation_indices(self.instantaneous_phase_corrected)
        shuff_inhalations = [self.instantaneous_phase_corrected[0:inhalations[0]]]
        for idx in range(len(inhalations)-1):
                shuff_inhalations.append(self.instantaneous_phase_corrected[inhalations[idx]:inhalations[idx+1]])
        shuff_inhalations.append(self.instantaneous_phase_corrected[inhalations[idx+1]:])
        # shuffle the list of cycles to get out an array with shuffled inhalations. This will be used to calculate the phase-locking confidence interval (Method used in Fukunaga, 2012)
        n_cells = len(self.tsecs)
        hist_counts = len(bins)-1
        phase_lock_boot_dist = np.empty((n_cells, n_shuffs, hist_counts))
        shuf_array = np.empty_like(self.instantaneous_phase_corrected)
        for shuffle in range(n_shuffs):
            # set the seed for reproducibility 
            random.seed(shuffle)
            # shuffle the respiration list 
            random.shuffle(shuff_inhalations)
            i = 0
            # recreate the shuffled respiration array
            for resp in shuff_inhalations:
                shuf_array[i:len(resp)+i] = resp
                i += len(resp)
            # find the phase of spikes aligned to the shuffled respirations 
            for cell_n, cell in enumerate(self.tsecs): 
                analyze_idx = (cell*fs).astype(int)
                analyze_idx = analyze_idx[analyze_idx>self.last_frame_idx]
                shuff_hist = np.histogram(shuf_array[analyze_idx], bins = bins)[0]
                phase_lock_boot_dist[cell_n, shuffle, :] = shuff_hist
        return phase_lock_boot_dist
    
    ''' Perform statistical test to identify significantly phase-locked cells. For each bin, compute the 95% confidence interval of bin height using the bootstrapped shuffled distribution. Cells are defined as phase locked if at least 5% of spikes exceed (either positively or negatively) the confidence intervals, and the CI is exceeded on at least two consecutive bins.'''
    def get_phase_locked_cells(self, phase_lock_boot_dist, bins, fs = 30000):
        n_cells = len(self.tsecs)
        tuned = np.zeros(n_cells)
        total_percent_spikes_exceeding = np.zeros(n_cells)
        # radius high is the upper bound of the 95% CI
        all_radius_low = np.empty((n_cells,len(bins)-1))
        # radius low in the lower bound of the 95% CI
        all_radius_high = np.empty((n_cells,len(bins)-1))
        all_phase_hist = np.empty((n_cells,len(bins)-1))
        for cell in range(n_cells):
            for bin_ in range(len(bins)-1):
                # get the observation in the 2.5th and 97.5th percentile
                all_radius_low[cell, bin_] = np.percentile(phase_lock_boot_dist[cell,:,bin_],2.5)
                all_radius_high[cell, bin_] = np.percentile(phase_lock_boot_dist[cell,:,bin_],97.5)
            # get the spike times occuring during the spontaneous analysis period 
            analyze_idx = (self.tsecs[cell]*fs).astype(int)
            analyze_idx = analyze_idx[analyze_idx>self.last_frame_idx]
            # bin the spike times by respiration phase
            hist = np.histogram(self.instantaneous_phase_corrected[analyze_idx], bins = bins)[0]
            all_phase_hist[cell,:] = hist
            radius_high = all_radius_high[cell,:]
            radius_low = all_radius_low[cell,:]
            # calculate the total percentage of spikes that exceed the 95% confidence interval in either the negative or positive direction
            percent_exceed = (np.sum(hist[np.where(hist>radius_high)[0]]-radius_high[np.where(hist>radius_high)[0]])/np.sum(hist))*100
            percent_min = np.sum(radius_low[np.where(hist<radius_low)[0]]-hist[np.where(hist<radius_low)[0]])/np.sum(hist)*100
            total_diff = percent_exceed+percent_min
            # additionally, ensure that at least 2 consecutive bins exceed the CI
            if 1 in np.concatenate(((np.diff(np.where(hist>radius_high)[0])),(np.diff(np.where(hist<radius_low)[0])))):
                consecutive_bins = 1
            else:
                consecutive_bins = 0 
            # if >5% of spikes exceed the confidence intervals across bins, and there are two or more consecutive bins that exceed the CI, cound the cell as tuned. 
            if (total_diff > 5) & (consecutive_bins == 1):
                tuned[cell] = 1
            # store the total percentage of spikes exceeding the confidence interval as a metric of tuning strength
            total_percent_spikes_exceeding[cell] = percent_exceed + percent_min
        # return the upper and lower bounds of the CI for each cell, the binary tuning array, and the array containing percentage of spikes exceeding the CI. 
        return all_radius_low, all_radius_high, all_phase_hist, tuned, total_percent_spikes_exceeding

    def get_instantaneous_phase_corrected(self):
        return self.instantaneous_phase_corrected
    
    def get_inhalation_indices(self):
        inhalation_idxs = phase_analysis.get_inhalation_indices(self.instantaneous_phase_corrected)
        return inhalation_idxs
    
    
    @staticmethod
    def get_resp_scaling_factor(instantaneous_phase_corrected, last_frame_idx, bins):
        
        # to compute the scaling factor, get the proportion of timepoints in the respiration cycle falling within each bin. 
        scaling_factor = np.histogram(instantaneous_phase_corrected[last_frame_idx:],bins = bins)[0]/np.sum(np.histogram(instantaneous_phase_corrected[last_frame_idx:],bins = bins)[0])
        
        return scaling_factor
        
    
    ''' to calculate the spontaneous phase tuning, for each cell we will compute a histogram of spikes across the sniff cycle (20 bins). We will then normalize this histogram by the average amount of time spent in each bin. This corrects for the non-sinusoidal shape of the respiration cycle, and bias for spikes to fall in exhalation time bins.'''
    def get_spontaneous_tuning(self, fs = 30000):
        
        # to compute the scaling factor, get the proportion of timepoints in the respiration cycle falling within each bin. 
        scaling_factor = Spontaneous_activity_analysis.get_resp_scaling_factor(self.instantaneous_phase_corrected, self.last_frame_idx, self.bins)
        
        # get the number of respirations occuring over the analysis period (from the last frame on time to end of experiment)
        n_resps = len(phase_analysis.get_inhalation_indices(self.instantaneous_phase_corrected[self.last_frame_idx:]))
        
        # this function gets the mean duration of each respiration, then multiplies by the proportion of time spent in each phase bin across the analysis period.
        # this gives a mean estimate of the duration of each phase bin. 
        duration_in_resp_bin = Spontaneous_activity_analysis.get_time_in_phase_bins(self.instantaneous_phase_corrected, self.last_frame_idx, self.bins)
        
        # to compute the scaled spontaneous tuning curves, divide the phase histogram generated for each cell by the scaling factor for each bin, then normalize to the max bin.
        nbins = len(self.bins)-1
        ncells = self.phase_tuning.shape[2]
        resp_tuning_scaled = np.empty((ncells, nbins))
        resp_tuning_per_inh_scaled = np.empty((ncells, nbins))
        for cell in range(len(self.tsecs)):
            analyze_idx = (self.tsecs[cell]*fs).astype(int)
            analyze_idx = analyze_idx[analyze_idx>self.last_frame_idx]
            test_hist = np.histogram(self.instantaneous_phase_corrected[analyze_idx].ravel(), bins = self.bins)[0]
            resp_tuning_scaled[cell,:] = ((test_hist/scaling_factor/(np.max(test_hist/scaling_factor))))
            # divide the histogram of total binned spikes by the number of respirations to get spikes per phase bin. 
            # multiply by (1/resp bin duration) to get spikes/s in bin. 
            resp_tuning_per_inh_scaled[cell,:] = ((test_hist/n_resps)*(1/duration_in_resp_bin))
            
        return resp_tuning_scaled, resp_tuning_per_inh_scaled, duration_in_resp_bin
    
    ''' for each cell, we want to find the significant tuning curves in order to understand how phase tuning relates to spontaneous tuning. 
    we predict that the phase tuning curve will generally lead the spontaneous tuning curve.'''
    def get_sig_tuning_curves(self):
        
        # stimulation phase preference is a n_spots x n_cells matrix containing the preferred phase for each significantly activated tuned cell. Entries for non-significantly activated/tuned cells are nans. 
        stimulation_phase_preference = self.phase_stats['stimulation_phase_preference']
        
        # get a list of all of the spot indices that contain significant tuning curves (significantly activated by at least one bin and significantly tuned) for each cell    
        activated_tuned_indices = []
        significant_phase_tuning_curves = []
        mean_significant_tuning_curves = []
        for cell_idx, cell in enumerate(stimulation_phase_preference.T):
            tuned_indices = np.where(~np.isnan(cell))[0]
            activated_tuned_indices.append(np.where(~np.isnan(cell))[0])
            if tuned_indices.any():
                cell_tuning_curves = self.phase_tuning[:,:,cell_idx]
                significant_phase_tuning_curves.append(cell_tuning_curves[tuned_indices,:])
                mean_significant_tuning_curves.append(np.mean(cell_tuning_curves[tuned_indices,:],0))
            else:
                significant_phase_tuning_curves.append(np.nan)
                mean_significant_tuning_curves.append(np.nan)
        
        return significant_phase_tuning_curves, mean_significant_tuning_curves, activated_tuned_indices
    
    ''' Caclulate the mean amount of time spent in each phase bin over the course of the experiment '''
    @staticmethod
    def get_time_in_phase_bins(instantaneous_phase_corrected, last_frame_idx, bins):
        
        resp_duration, inh_duration, exh_duration = phase_analysis.get_resp_stats(instantaneous_phase_corrected)

        scaling_factor = Spontaneous_activity_analysis.get_resp_scaling_factor(instantaneous_phase_corrected, last_frame_idx, bins)
    
        duration_in_resp_bin = scaling_factor*np.mean(resp_duration)
        
        return duration_in_resp_bin
    
    def get_ncells(self):
        
        ncells = self.phase_tuning.shape[2]
        
        return ncells