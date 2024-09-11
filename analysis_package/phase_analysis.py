#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import kruskal
import random
import PGanalysis

def dict_key_concatenator(dict_list, key, axis = 1):
    '''
    parameters
    ------
    dict_list: list of dictionaries where you would like to concatenate across like keys
    key: string indicating key that you would like to conatenate across
    
    outputs
    -------
    concatenated_key: numpy array containing the concatenated key variables.'''
    
    if type(dict_list[0][key]) != dict:
        concat_var = dict_list[0][key]
        concat_var_shape = concat_var.shape
        for entry in range(1,len(dict_list)):
            concat_var = np.concatenate((concat_var, dict_list[entry][key]), axis = len(concat_var_shape)-1)
        return concat_var
    
    if type(dict_list[0][key]) == dict:
        key2 = dict_list[0][key].keys()
        key2 = [keys for keys in key2][0]
        concat_var = dict_list[0][key][key2]
        concat_var_shape = concat_var.shape
        for entry in range(1,len(dict_list)):
            concat_var = np.concatenate((concat_var, dict_list[entry][key][key2]), axis = len(concat_var_shape)-1)
        return concat_var

'''pull out the spike times (in seconds) from the HDF5 file. Output is a list of spiketimes for each cell'''
# datafile is a string with the location of the spyking circus HDF5 file
def get_tsecs(datafile):
    spike_files = datafile
    spiketimes_allcells = []
    for file in spike_files:
        tsecs_file, spike_index, clusternumber = PGanalysis.get_spiketimes_s(file)
        spiketimes_allcells.append(tsecs_file)
    tsecs = [cell for experiment in spiketimes_allcells for cell in experiment]
    return tsecs 

'''get the trigger index of each trial sorted by spot. Output is an n_spots x n_trials_per_spot numpy array'''
def get_trial_type_indices(expt_metadata):
    all_trials = []
    #the 'pixels' field of expt_metadata contains the 'pixel' of the grid that was stimulated for each trial.
    #in this case, a pixel is the equivalent of a spot (not a single pixel on the dmd or camera). Apologies for the confusing naming. 
    #concatenating and unraveling the order of spot stimulation. 
    for block in expt_metadata['pixels']:
        all_trials.append(block[0:])
    trials = [trial for block in all_trials for trial in block]
    spot_ID = trials
    unique_spot_IDs = np.unique(spot_ID)
    trial_type_indices = []
    # finding all indices in the experiment when each spot was triggered. 
    for ID in unique_spot_IDs:
        trial_type_index = np.where(spot_ID == ID)
        trial_type_indices.append(trial_type_index[0])
    # output numpy array is n_spots x n_trials per spot. 
    return np.array(trial_type_indices)

''' identify frame on times from trigger array'''
def get_frame_on_time_s(mtrigger_array):
    mtrigger_array[mtrigger_array < 20000] = 0
    mtrigger_array[mtrigger_array >= 20000] = 30000
    trigger_diff = np.unique(np.diff(mtrigger_array))
    diff_array = np.diff(mtrigger_array)
    trig_times = np.where(diff_array == trigger_diff[0])[0]
    frame_change_timediff = np.diff(trig_times)
    frame_change_indices = np.unique(np.diff(trig_times))
    frame_on = np.where((frame_change_timediff <= frame_change_indices[2]))
    frame_on_time_s = trig_times[frame_on]/30000
    return frame_on_time_s

''' get the indices of trials occuring in each respiration bin'''
def get_bin_indices(trial_type_indices, expt_metadata, frame_on_time_s, instantaneous_phase_corrected, fs = 30000, bin_width = 18):
    bin_indices_all = np.empty(np.array(trial_type_indices).shape)
    n_stim = len(np.unique(expt_metadata['trial_indices']))
    for stim in range(n_stim):
        #trial type indices is the trials sorted by stimulus ID. Find the index of all trials with the same ID. 
        trial_type_frame_on_idx = (frame_on_time_s[trial_type_indices[stim]]*fs)# calculate time shift by looking at peak in the PSTH in response to stimulation +(.1*fs)

        #get the instantaneous phase at trial start for each trial 
        trial_phase = instantaneous_phase_corrected[trial_type_frame_on_idx.astype(int)]

        #bin the phases into 20 bins (equivalent to 18 degrees)
        bins = np.arange(0,np.radians(360)+np.radians(bin_width), np.radians(bin_width))
        phase_hist = np.histogram(trial_phase, bins = bins)

        #get the bin index of each trials (this groups trials into 20 bins depending on phase of presentation)
        bin_indices = np.digitize(trial_phase, phase_hist[1], right = False)

        #include the edge trial if necessary 
        bin_indices[bin_indices>(len(phase_hist[0]))] = len(phase_hist[0])

        bin_indices_all[stim,:] = bin_indices
    return bin_indices_all, bins
    
    
'''get mean and standard deviation of respiration rate'''
def get_resp_stats(instantaneous_phase_corrected, start_coord = 0, end_coord = -1, fs = 30000):
    # define analysis window
    instantaneous_phase_corrected_awake = instantaneous_phase_corrected[start_coord:end_coord]
    # find all inhalations
    threshold = 1
    # find threshhold crossings 
    threshold_crossings = np.diff(instantaneous_phase_corrected_awake> threshold, prepend=False)
    # find upward crossings 
    upward_crossings = (np.roll(instantaneous_phase_corrected_awake,-1) < threshold)
    # find negative-going threshold crossings 
    threshold_idx_inhalations = np.argwhere(threshold_crossings & upward_crossings)[:,0]
    # find all inhalation-exhalation transitions
    threshold = np.pi
    # find threshhold crossings 
    threshold_crossings = np.diff(instantaneous_phase_corrected_awake > threshold, prepend=False)
    # find upward crossings 
    upward_crossings = (np.roll(instantaneous_phase_corrected_awake,-1) > threshold)
    # find negative-going threshold crossings 
    threshold_idx_cross = np.argwhere(threshold_crossings & upward_crossings)[:,0]
    # remove noise by selecting the first point at which respiration crosses threshold 
    phase_aligned_indices = np.empty_like(threshold_idx_inhalations)
    for resp_idx, resp in enumerate(threshold_idx_inhalations[0:-2]):
        locate_phase = threshold_idx_cross - resp
        if len(np.where(locate_phase>0)[0])>0:
            min_index = np.where(locate_phase>0)[0][0]
            phase_aligned_indices[resp_idx] = threshold_idx_cross[min_index]
    erroneous_inhalations = np.diff(phase_aligned_indices) == 0
    erroneous_inhalations = np.insert(erroneous_inhalations,0,0)
    #remove erroneously labeled inhalations from arrays
    threshold_idx_inhalations = threshold_idx_inhalations[~erroneous_inhalations]
    phase_aligned_indices = phase_aligned_indices[~erroneous_inhalations]
    # get difference between inhalations to find resp duration 
    respiration_duration = np.diff(threshold_idx_inhalations)/fs
    #get duration of inhalation to start of exhalation:
    inhalation_duration = (phase_aligned_indices - threshold_idx_inhalations)/fs
    #get exhalation duration 
    exhalation_duration = (threshold_idx_inhalations[1:] - phase_aligned_indices[0:-1])/fs
    return respiration_duration, inhalation_duration, exhalation_duration

'''get index of all inhalations'''
def get_inhalation_indices(instantaneous_phase_corrected, start_coord = 0, end_coord = -1, fs = 30000):
    # define analysis window
    instantaneous_phase_corrected_awake = instantaneous_phase_corrected[start_coord:end_coord]
    # find all inhalations
    threshold = 1
    # find threshhold crossings 
    threshold_crossings = np.diff(instantaneous_phase_corrected_awake> threshold, prepend=False)
    # find upward crossings 
    upward_crossings = (np.roll(instantaneous_phase_corrected_awake,-1) < threshold)
    # find negative-going threshold crossings 
    threshold_idx_inhalations = np.argwhere(threshold_crossings & upward_crossings)[:,0]
    # find all inhalation-exhalation transitions
    threshold = np.pi
    # find threshhold crossings 
    threshold_crossings = np.diff(instantaneous_phase_corrected_awake > threshold, prepend=False)
    # find upward crossings 
    upward_crossings = (np.roll(instantaneous_phase_corrected_awake,-1) > threshold)
    # find negative-going threshold crossings 
    threshold_idx_cross = np.argwhere(threshold_crossings & upward_crossings)[:,0]
    # remove noise by selecting the first point at which respiration crosses threshold 
    phase_aligned_indices = np.empty_like(threshold_idx_inhalations)
    for resp_idx, resp in enumerate(threshold_idx_inhalations[0:-2]):
        locate_phase = threshold_idx_cross - resp
        if len(np.where(locate_phase>0)[0])>0:
            min_index = np.where(locate_phase>0)[0][0]
            phase_aligned_indices[resp_idx] = threshold_idx_cross[min_index]
    erroneous_inhalations = np.diff(phase_aligned_indices) == 0
    erroneous_inhalations = np.insert(erroneous_inhalations,0,0)
    #remove erroneously labeled inhalations from arrays
    threshold_idx_inhalations = threshold_idx_inhalations[~erroneous_inhalations]
    return threshold_idx_inhalations

'''get the respiration phase-tuning curve for each stimulus'''
def get_phase_tuning_by_stim(cell, bins, trial_type_psth_array, bin_indices_all, shuff_bin_ID = False, analysis_window = [0,.3], baseline_window = [-.3,0], plot = False, norm = True, ylim = [0,10], xlim = [0,360], color = 'k'):
    all_psth_phase_mean_norm = np.empty((trial_type_psth_array.shape[1],len(bins)-1))
    x = np.arange(-1,2,.002)
    for stim in range(trial_type_psth_array.shape[1]):
        psth_phase_sorted = []
        psth_phase_mean_sorted = []
        
        if shuff_bin_ID == False:  
            for index in np.unique(bin_indices_all[stim]):
                # for each index, get the psth for each trial, then compute the mean psth across trials
                sort_index = np.where(bin_indices_all[stim] == index)[0]
                psth_phase_sorted.append(trial_type_psth_array[cell,stim,sort_index,:])
                psth_phase_mean_sorted.append(np.mean(trial_type_psth_array[cell,stim,sort_index,:],0))
        
        # in normal case, the tuning curves are the firing rates for each successive bin. We can see how similar the tuning is across
        # spots using this metric. As a comparison, we can shuffle the bins across each curve to get rid of any correlations between 
        # curves for different spots. This is implemented below. 
        if shuff_bin_ID == True: 
            # to get a different shuffle order for each stimulus, set the seed. 
            random.seed(stim)
            # shuffle the order of bins in the tuning curve. 
            indices_list = np.unique(bin_indices_all[stim])
            random.shuffle(indices_list)
            for index in indices_list:
                # get the trials corresponding to each bin. This result is the same as non-shuffled.
                sort_index = np.where(bin_indices_all[stim] == index)[0]

                # for each index, get the psth for each trial, then compute the mean psth across trials
                psth_phase_sorted.append(trial_type_psth_array[cell,stim,sort_index,:])
                psth_phase_mean_sorted.append(np.mean(trial_type_psth_array[cell,stim,sort_index,:],0))

        #analyze the response in a 300ms window (this is equivalent to the ISI)
        analyze_indices = np.where((x>analysis_window[0]) & (x<analysis_window[1]))

        #get the baseline firing rate 
        analyze_baseline = np.where((x>baseline_window[0]) & (x<baseline_window[1]))

        #get the baseline-subtracted change in firing rate
        psth_phase_mean_sorted_select = np.array(psth_phase_mean_sorted)[:,analyze_indices]
        psth_phase_mean_sorted_select_baseline = np.array(psth_phase_mean_sorted)[:,analyze_baseline]
        psth_phase_means = np.mean(psth_phase_mean_sorted_select.squeeze(),1) # - psth_phase_mean_sorted_select_baseline.squeeze(),1)

        #plot the baseline-subtracted change in firing rate normalized to the max across bins. Compare to the respiration-phase histogram of the cell. 
        if norm == True:
            all_psth_phase_mean_norm[stim,:] = psth_phase_means/np.max(psth_phase_means)
        if norm == False:
            all_psth_phase_mean_norm[stim,:] = psth_phase_means
            
    if plot == True: 
        plt.figure(figsize = (15,5))
        for stim_idx, stim_resp in enumerate(all_psth_phase_mean_norm[1:,:]):
            plt.subplot(2,5,stim_idx+1)
            plt.plot(np.degrees(bins[0:-1]), stim_resp, 'k')
            plt.plot(np.degrees(bins[0:-1]), all_psth_phase_mean_norm[0,:], 'b')
            plt.xticks([0,180,360])
            if norm == True:
                plt.ylim(-2,1)
                plt.yticks([-2,-1,0,1])
            plt.xlabel('phase')
            plt.ylabel('FR. (Hz)')
#             if stim_idx == 0:
#                 plt.title('blank')
#                 PGanalysis.axis_fixer(ratio = 1)
#             else:
            plt.title('spot ' + str(stim_idx+1))
            plt.ylim(ylim[0],ylim[1])
            PGanalysis.axis_fixer(ratio = 1) 
    return all_psth_phase_mean_norm

'''get the preferred respiration phase as a function of stimulus and bin'''
def get_pref_resp_phase_by_stim(cell, bins, trial_type_psth_array, frame_on_time_s, trial_type_indices, bin_indices_all, instantaneous_phase_corrected, analysis_window = [0,.3], baseline_window = [-.3,0], fs = 30000):
    all_psth_phase_mean_norm = np.empty((trial_type_psth_array.shape[1],len(bins)-1))
    stim_resp_pref_phase = np.empty((trial_type_psth_array.shape[1],len(bins)-1))
    x = np.arange(-1,2,.002)
    #analyze the response in a 300ms window (this is equivalent to the ISI)
    analyze_indices = np.where((x>analysis_window[0]) & (x<analysis_window[1]))[0]
    #get a baseline in the 300ms before stim on
    baseline_indices = np.where((x>baseline_window[0]) & (x<baseline_window[1]))[0]
    time_window = x[analyze_indices]
    all_stim_resp_bins_max_rate_idx = []
    trial_type_peak_times = []
    trial_type_time_to_50 = []
    all_psth_phase_mean_sorted = []
    trial_type_frame_idx_sorted_all = []
    for stim in range(trial_type_psth_array.shape[1]):
        psth_phase_sorted = []
        psth_phase_mean_sorted = []
        trial_type_frame_on_idx = (frame_on_time_s[trial_type_indices[stim]]*fs)
        trial_type_frame_on_idx_sorted = []
        all_index_peak_times = []
        all_index_time_to_50 = []
        for index_num, index in enumerate(np.unique(bin_indices_all[stim])):
            # for each index, get the psth for each trial, then compute the mean psth across trials
            sort_index = np.where(bin_indices_all[stim] == index)[0]
            # sort psths into each bin 
            psth_phase_sorted.append(trial_type_psth_array[cell,stim,sort_index,:])
            # get the mean psth across sorted trials for this stimulus, should end up with a n_binsx1500 array 
            psth_phase_mean_sorted.append(np.mean(trial_type_psth_array[cell,stim,sort_index,:],0))
            # get the time of the peak psth in each bin 
            peak_times = np.argmax(np.array(psth_phase_mean_sorted)[index_num,analyze_indices])
            # find the time to 50% of max response
            stimulus_evoked_activity = np.array(psth_phase_mean_sorted)[index_num,analyze_indices]
            baseline_activity = np.array(psth_phase_mean_sorted)[index_num,baseline_indices]
            baseline_subtracted_activity = stimulus_evoked_activity - np.mean(baseline_activity)
            if np.max(baseline_subtracted_activity)>=0:
                time_to_50 = np.min(np.where((baseline_subtracted_activity) >= (np.max(baseline_subtracted_activity)/2))[0])
                all_index_time_to_50.append(time_window[time_to_50]*1000)
                # here, to get the response phase what I am doing is taking each trial index and adding the mean time to 50% of max. This is probably not the best way to do this. Also note that we need to use circular statistics to calculate the mean instantaneous phase. Otherwise on later bins, the response phase artifically become the average of both late phases and early phases, resulting in a mean of ~180. 
                trial_type_frame_on_idx_sorted.append(trial_type_frame_on_idx[sort_index.astype(int)]+(time_window[time_to_50]*fs).astype(int))
            
            # get phase at half max response
                stim_resp_pref_phase[stim, index_num] = np.degrees(scipy.stats.circmean(instantaneous_phase_corrected[trial_type_frame_on_idx_sorted[index_num].astype(int)]))
                
            else:
                time_to_50 = np.nan
                all_index_time_to_50.append(np.nan)
                trial_type_frame_on_idx_sorted.append(np.nan)
                stim_resp_pref_phase[stim, index_num] = np.nan
                
            #find the frame on index of the peak time. Should be able to compute peak response phase from this. 
            #trial_type_frame_on_idx_sorted.append(trial_type_frame_on_idx[sort_index.astype(int)]+(time_window[peak_times]*fs).astype(int))
            
            all_index_peak_times.append(time_window[peak_times]*1000)
            
            #stim_resp_pref_phase[stim, index_num] = np.degrees(np.mean(instantaneous_phase_corrected[trial_type_frame_on_idx_sorted[index_num].astype(int)]))
            
        trial_type_peak_times.append(all_index_peak_times)
        trial_type_time_to_50.append(all_index_time_to_50)
        all_psth_phase_mean_sorted.append(psth_phase_mean_sorted)
        trial_type_frame_idx_sorted_all.append(trial_type_frame_on_idx_sorted)
        #list that is 11 stim x 20 bins x n number of trials in bins 
    return stim_resp_pref_phase, np.array(trial_type_peak_times), np.array(all_psth_phase_mean_sorted), np.array(trial_type_time_to_50)

''' for each trial, get the number of spikes fired in the 300ms after stimulation onset.'''
def get_trial_type_spike_counts(trial_type_raster_array, trial_type_psth_array,x = np.arange(-1,2,.002), max_trial = []):
    trial_type_spike_counts = np.empty(trial_type_psth_array.shape[0:3])
    trial_type_spike_rate = np.empty(trial_type_psth_array.shape[0:3])
    for cell_n, cell in enumerate(trial_type_raster_array):
        for stim_n, stim in enumerate(cell):
            for trial_n, trial in enumerate(stim):
                analyze_indices = np.where((x>0)&(x<.3))[0]
                if max_trial:
                    trial_type_spike_counts[cell_n, stim_n, trial_n] = len(np.where((trial>0) & (trial<.3))[0][0:max_trial])
                else:
                    trial_type_spike_counts[cell_n, stim_n, trial_n] = len(np.where((trial>0) & (trial<.3))[0])
                psth = trial_type_psth_array[cell_n, stim_n, trial_n,:]
                trial_type_spike_rate[cell_n, stim_n, trial_n] = np.mean(psth[analyze_indices])
    return trial_type_spike_counts, trial_type_spike_rate

''' bin spike counts for each spot type into phases'''
def bin_spike_counts_by_phase(trial_type_spike_counts, bin_indices_all, max_trial = []):
    trial_type_phase_binned_spike_counts = []
    stim_phase_binned_spike_counts = []
    for stim_n, stim in enumerate(bin_indices_all):
        stim_phase_binned_counts = []
        for idx in np.unique(stim):
            if max_trial:
                stim_phase_binned_counts.append(trial_type_spike_counts[:,stim_n, np.where(stim == idx)[0][0:max_trial]])
            else: 
                stim_phase_binned_counts.append(trial_type_spike_counts[:,stim_n, np.where(stim == idx)[0]])
        trial_type_phase_binned_spike_counts.append(stim_phase_binned_counts)
    return trial_type_phase_binned_spike_counts

'''Perform kruskal-wallis test (non-parametric ANOVA) to assess whether cells are tuned'''
def kruskal_wallis_phase_tuning(trial_type_phase_binned_spike_counts, p_thresh = .01):
    n_stim = len(trial_type_phase_binned_spike_counts)
    n_bins = len(trial_type_phase_binned_spike_counts[0])
    n_cells = len(trial_type_phase_binned_spike_counts[0][0])
    cell_bin_list = []
    labels = []
    kruskal_pval = np.empty((n_stim,n_cells))
    kruskal_pval_thresh = np.zeros((n_stim,n_cells))
    for stim in range(n_stim):
        for cell in range(n_cells):
            cell_bin_list = []
            for bin_num in range(n_bins):
                cell_bin_list.append(trial_type_phase_binned_spike_counts[stim][bin_num][cell])
            stat, pval = kruskal(*cell_bin_list)
            kruskal_pval[stim,cell] = pval
            if pval < p_thresh:
                kruskal_pval_thresh[stim,cell] = 1
    return kruskal_pval_thresh

''' Determine whether cells are significantly activated or suppressed at each bin using a rank sum test (mann whitney U) and auroc'''
def rank_sum_test(trial_type_phase_binned_spike_counts, p_thresh = .0025):
    n_stim = len(trial_type_phase_binned_spike_counts)
    n_bins = len(trial_type_phase_binned_spike_counts[0])
    n_cells = len(trial_type_phase_binned_spike_counts[0][0])
    activated = []
    stim_activated = []
    stim_suppressed = []
    for stim in range(n_stim):
        activated = []
        emp_cells_activated = np.zeros((n_bins,n_cells))
        emp_cells_suppressed = np.zeros((n_bins,n_cells))
        for bin_num in range(n_bins):
            # comparing the spike counts in the target stim x bin combination to the spike counts in the equivalent bin for the blank stimulus.
            # blank stimulus index is always zero.
            p1 = trial_type_phase_binned_spike_counts[stim][bin_num]
            p2 = trial_type_phase_binned_spike_counts[0][bin_num]
            # computing the p-value and the auroc 
            pvals,auroc = PGanalysis.twoptn_rank_sum_test(p1,p2)
            # auroc > .5 and p < p_thresh, cell is significantly activated at this stim x bin combo
            emp_cells_activated[bin_num, (np.where((pvals<p_thresh) & (auroc>.5))[0])] = 1
            # auroc < .5 and p < p_thresh, cell is significantly suppressed at this stim x bin combo 
            emp_cells_suppressed[bin_num, (np.where((pvals<p_thresh) & (auroc<.5))[0])] = 1
        stim_activated.append(emp_cells_activated)
        stim_suppressed.append(emp_cells_suppressed)
    stim_activated = np.array(stim_activated)
    stim_suppressed = np.array(stim_suppressed)
    responsive_cells = {'activated':stim_activated, 'suppressed':stim_suppressed}
    return responsive_cells

''' For each cell, get the correlation between tuning curves for activating and suppressing spots'''
def get_tuning_curve_correlations(trial_type_psth_array, bins, bin_indices_all, activated_cell_bins, suppressed_cell_bins, kruskal_pval_thresh, shuff = False):
    resp_corr_pairs = []
    resp_sup_pairs = []
    all_resp_corr_pairs = []
    all_resp_sup_pairs = []
    activated_cell_spot_correlation_mat = []
    all_activating_spot_IDs = []
    n_cells = trial_type_psth_array.shape[0]
    if shuff == False:
        shuff_bins = False
    elif shuff == True:
        shuff_bins = True
    for cell in range(n_cells):
        # get the normalized tuning curves for all spots
        all_psth_phase_mean_norm= get_phase_tuning_by_stim(cell, bins, trial_type_psth_array, bin_indices_all, shuff_bin_ID = shuff_bins, norm = True)
        #get the indices of the response-eliciting spots (both activating and suppressing)
        resp_spots = np.where(np.sum(activated_cell_bins[:,:,cell],1)>0)[0]
        resp_spots = resp_spots[kruskal_pval_thresh[resp_spots,cell]>0]
        sup_spots = np.where(np.sum(suppressed_cell_bins[:,:,cell],1)>0)[0]
        sup_spots = sup_spots[kruskal_pval_thresh[sup_spots,cell]>0]
        if resp_spots.any():
            if len(resp_spots)>1:
                # get the tuning curve correlation matrix for all activating spots 
                resp_corr_pairs = np.corrcoef(all_psth_phase_mean_norm[resp_spots,:])
                # set the top half of the matrix to nan (this is to get rid of duplicates and self-correlation)
                resp_corr_pairs[np.arange(resp_corr_pairs.shape[0])[:,None] <= np.arange(resp_corr_pairs.shape[1])] = np.nan
                all_resp_corr_pairs.append(resp_corr_pairs[~np.isnan(resp_corr_pairs)])
                # save the full matrix for each cell 
                activated_cell_spot_correlation_mat.append(resp_corr_pairs)
                # save the identity of spots. Later we will use this to plot tuning curve correlation as a function of spot distance. 
                all_activating_spot_IDs.append(resp_spots)
        if sup_spots.any():
            if len(sup_spots)>1:
                # get the tuning curve correlation matrix for all suppressing spots 
                resp_sup_pairs = np.corrcoef(all_psth_phase_mean_norm[sup_spots,:])
                # set the top half of the matrix to nan (this is to get rid of duplicates and self-correlation)
                resp_sup_pairs[np.arange(resp_sup_pairs.shape[0])[:,None] <= np.arange(resp_sup_pairs.shape[1])] = np.nan
                all_resp_sup_pairs.append(resp_sup_pairs[~np.isnan(resp_sup_pairs)])
    all_resp_corr_pairs = [corr for resp in all_resp_corr_pairs for corr in resp]
    all_resp_sup_pairs = [corr for resp in all_resp_sup_pairs for corr in resp]
    resp_corr_pairs = {'activated_cells': all_resp_corr_pairs, 'suppressed_cells':all_resp_sup_pairs, 'activated_cell_spot_correlation_mat':activated_cell_spot_correlation_mat, 'activating_spot_IDs':all_activating_spot_IDs}
    return resp_corr_pairs

def unravel_list(in_list):
    unraveled_list = [l for entry in in_list for l in entry]
    return unraveled_list

''' class to pull out phase tuning response stats'''
class Phase_response_stats:
    def __init__(self, cell, bins, resp_spots, stim_resp_pref_phase, peak_times, all_psth_phase_mean_sorted, time_to_50, all_psth_phase_mean_norm, stim_activated):
        
        self.n_spots = stim_activated.shape[0] #number of spots stimulated in experiment
        
        self.n_bins = stim_activated.shape[1] #number of degree bins that respiration cycle is divided into 
        
        self.n_cells = stim_activated.shape[2] #number of cells recorded in experiment
        
        self.cell = cell 
        
        self.bins = bins
        
        # resp_spots: the spots that elicit significantly tuned responses (KW test p<.01) and have at least one significantly activated bin (Rank-sum test p<.0025)
        self.resp_spots = resp_spots
        
        # resp_pref_phase: the preferred RESPONSE phase of the cell for each spot x phase bin combination (instantaneous phase at peak of mean PSTH for stimulation in each phase bin).
        self.stim_resp_pref_phase = stim_resp_pref_phase
        
        # peak_times: the latency to the peak of the mean PSTH for each spot x phase bin combination
        self.peak_times = peak_times
        
        # all_psth_phase_mean_sorted: the n_spots x n_bins x n_timepoints mean PSTH for each phase bin
        self.all_psth_phase_mean_sorted = all_psth_phase_mean_sorted
       
        # time_to_50: the latency to 50% of the peak of the mean PSTH for each spot x phase bin combination
        self.time_to_50 = time_to_50
        
        # all_psth_phase_mean_norm: the mean firing rate for each spot x phase bin combination in the first 300ms after stimulation
        self.all_psth_phase_mean_norm = all_psth_phase_mean_norm
        
        # stim_activated: n_spots x n_bins x n_stim array indicating signficantly activating spot x bin pairs. 1 is significantly activated, 0 is not. 
        self.stim_activated = stim_activated
        
        # get the bin activation indices of for the spots that evoke both significantly activated and significantly tuned responses. 
        self.stim_tuned = stim_activated[resp_spots,:,:]
    
    '''get the preferred response (NOT STIMULATION) phase for each spot for each significantly responsive bin.'''
    '''the preferred response phase tends to be linearly related to the phase bin in which the stimulation occurred.'''
    def return_spot_x_bin_response_pref_resp_phase(self):
        all_stim_resp_pref_phase = np.zeros((self.n_spots,self.n_bins)) + np.nan
        for spot in self.resp_spots:
            #get the significantly activated phase bins for each activating spot
            stim_tuned_byspot = self.stim_activated[spot,:,self.cell] 
            resp_bins = np.where(stim_tuned_byspot)[0]
            for bin_n in resp_bins:
                # get the preferred phase (instantaneous phase at peak of mean PSTH for stimulation in each phase bin)
                all_stim_resp_pref_phase[spot,bin_n] = self.stim_resp_pref_phase[spot,bin_n]
        return all_stim_resp_pref_phase
    
    ''' get the time to 50% of the max PSTH for each spot for each significantly responsive bin.'''
    ''' the time to 50 tends to be (although not always) constant across bins and spots, but varies across cells.'''
    def return_spot_x_bin_time_to_50(self):
        all_stim_rise_times = np.zeros((self.n_spots,self.n_bins)) + np.nan
        for spot in self.resp_spots:
            #get the significantly activated phase bins for each activating spot
            stim_tuned_byspot = self.stim_activated[spot,:,self.cell]
            resp_bins = np.where(stim_tuned_byspot)[0]
            for bin_n in resp_bins:
                # get the time to 50% for each spot x bin combo
                all_stim_rise_times[spot,bin_n] = self.time_to_50[spot,bin_n]
        return all_stim_rise_times
    
    ''' get the time to PSTH peak for each spot for each significantly responsive bin.'''
    def return_spot_x_bin_time_to_peak(self):
        all_stim_peak_times = np.zeros((self.n_spots,self.n_bins)) + np.nan
        for spot in self.resp_spots:
            #get the significantly activated phase bins for each activating spot
            stim_tuned_byspot = self.stim_activated[spot,:,self.cell]
            resp_bins = np.where(stim_tuned_byspot)[0]
            for bin_n in resp_bins:
                # get the time to 50% for each spot x bin combo
                all_stim_peak_times[spot,bin_n] = self.peak_times[spot,bin_n]
        return all_stim_peak_times
    
    '''get the response (normalized to max response bin) of each significantly activated bin for each spot.'''
    '''for significantly activated/tuned cells, get the bin with the max significant response for each spot x cell combination.'''
    def return_spot_x_bin_respond_pref_stim_phase(self):
        all_stim_pref_stim_phase = np.zeros((self.n_spots,self.n_bins)) + np.nan
        all_stim_phase_max = np.zeros((self.n_spots)) + np.nan
        for spot in self.resp_spots:
            #get the significantly activated phase bins for each activating spot
            stim_tuned_byspot = self.stim_activated[spot,:,self.cell]
            resp_bins = np.where(stim_tuned_byspot)[0]
            for bin_n in resp_bins:
                # get the norm to max response for each significant spot x bin combo
                all_stim_pref_stim_phase[spot,bin_n] = self.all_psth_phase_mean_norm[spot,bin_n]
            # get the bin of the max significant response
            all_stim_phase_max[spot] = np.degrees(self.bins)[np.nanargmax(all_stim_pref_stim_phase[spot,:])]
        stimulation_phase_preference = {'all_stim_pref_stim_phase':all_stim_pref_stim_phase, 'max_phase_stim_x_cell':all_stim_phase_max}
        return stimulation_phase_preference


''' plotting functions here'''

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, color = 'b', conf_inf = False, low_percentile = [], high_percentile = []):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor=color, fill=True, color = color, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
        
    if conf_inf == True:
        ax.plot(bins[:-1], low_percentile)
        ax.plot(bins[:-1], high_percentile)

    return ax, n, bins, patches, radius

'''Plot PSTHs for each cell for each phase bin.'''
def plot_phase_binned_PSTHs(cell, stim, trial_type_psth_array, bin_indices_all, bins, x):
    plt.figure(figsize = (50,20))
    for bin_num in range(1,len(bins)):
        cell_stim_psth = trial_type_psth_array[cell,stim]
        phase_binned_psth = []
        for i, trial in enumerate(cell_stim_psth): 
            if i in np.where(bin_indices_all[stim] == bin_num)[0]:
                phase_binned_psth.append(trial)
        plt.subplot(4,5,bin_num)
        plt.plot(x, np.mean(phase_binned_psth,0), linewidth = 5, color = 'k');
        plt.xlim(-.5,.5)
        plt.title(str(np.degrees(bins[bin_num])))
        plt.ylabel('trial')
        plt.xlabel('time since stim (s)')
        plt.ylim(0,30)
        PGanalysis.axis_fixer(ratio = .5, size = 30)
    PGanalysis.axis_fixer(ratio = .5, size = 30)
    
'''Plot rasters for each cell for each phase bin.'''
def plot_phase_binned_rasters(cell, stim, trial_type_raster_array, bin_indices_all, bins, x, color = 'k'):    
    plt.figure(figsize = (50,20))
    for bin_num in range(1,len(bins)):
        cell_stim_raster = trial_type_raster_array[cell][stim]
        phase_binned_raster = []
        for i, trial in enumerate(cell_stim_raster): 
            if i in np.where(bin_indices_all[stim] == bin_num)[0]:
                phase_binned_raster.append(trial)
        plt.subplot(4,5,bin_num)
        plt.eventplot(phase_binned_raster, linewidth = 5, color = color);
        plt.xlim(-.1,.3)
        plt.title(str(np.degrees(bins[bin_num])))
        plt.ylabel('trial')
        plt.xlabel('time since stim (s)')
        PGanalysis.axis_fixer(ratio = .5, size = 30)
    PGanalysis.axis_fixer(ratio = .5, size = 30)
    
    
def plot_phase_binned_rasters(trial_type_raster_array, bin_indices_all, cell, stim, bin_size_degrees = 36, axis_width = 1.822):
    '''
    This function plots the rasters for trials of spot stimululation binned by respiration phase.
    
    parameters
    ------
    cell: index of the cell to plot
    stim: index of the spot responses to plot
    
    outputs
    -------
    plot of rasters aligned to stimulation, binned by respiration phase.'''
        
    plt.figure(figsize = (10,7))
    bins = np.arange(0,np.radians(360)+np.radians(bin_size_degrees), np.radians(bin_size_degrees))
    for bin_num in range(1,len(bins)):
        trial_type_raster_array_sample = trial_type_raster_array[cell][stim]
        phase_binned_raster = []
        for i, trial in enumerate(trial_type_raster_array_sample): 
            if i in np.where(bin_indices_all[stim] == bin_num)[0]:
                phase_binned_raster.append(trial)
        plt.subplot(4,5,bin_num)
        plt.eventplot(phase_binned_raster, color = 'k', linewidths = .3);
        plt.xlim(-.1,.3)
        plt.ylim(0, len(phase_binned_raster))
        plt.yticks([0, len(phase_binned_raster)])
        plt.xticks([-.1, 0, .1, .2, .3], labels = ['-.1', '0', '.1', '.2', '.3'])
        plt.title(str(np.degrees(bins[bin_num]).astype(int)) + '°')
        if (bin_num == 1)|(bin_num == 6):
            plt.ylabel('trial')
        plt.xlabel('time since stim (s)')
        PGanalysis.axis_fixer(ratio = .5, size = 10, axis_width = axis_width)
        
def plot_spot_tuning_curves(tuning, cell, ylim = 30):
    '''
    This function plots the mean tuning curves for each cell spot pair.
    
    parameters
    ------
    tuning: the 
    cell: index of the cell to plot
    
    outputs
    -------
    plot of phase tuning curves for each stimulated spot.'''
    
    phases = np.arange(0,360,18)
    plt.figure(figsize = (10,5))
    for stim in range(1,tuning.shape[0]):
        plt.subplot(2,5,stim)
        plt.plot(phases,tuning[stim,:,cell],'k')
        plt.plot(phases,tuning[0,:,cell],color = [.7,.7,.7])
        plt.ylim(0,ylim)
        plt.xlim(0,360)
        plt.xticks([0,180,360])
        plt.yticks([0,ylim/2,ylim])
        plt.title('spot ' + str(stim))
        plt.xlabel('phase')
        if (stim == 1)|(stim == 6):
            plt.ylabel('FR (Hz)')
        PGanalysis.axis_fixer(ratio = 1, size = 15)