
import numpy as np
import os
import scipy.io
import h5py
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy import signal
import random
import PGanalysis
import phase_analysis



def hilbert_transform_resp(resp_array, butter_order = 1, butter_crit_freq = 8, resp_fs = 2000):
    '''
    Applies the Hilbert transfrom to derive the instantaneous phase of the respiration signal. Shifts the phase so that
    zero corresponds to the start of inhalation.
    
    Parameters
    ----------
    resp_array: input respiration signal. In Shiva's experiments this is recorded at 2 kHz.
    butter_order: order of the butterworth filter
    butter_crit_freq: critical frequency of the butterworth filter 
    resp_fs: sampling rate of the respiration signal
    
    Outputs:
    --------
    instantaneous_phase_corrected: Instantaneous phase of the respiration trace corrected so 0 degrees is the onset of inhalation 
    '''

    #low pass filter the respiration signal 
    sos = signal.butter(butter_order, butter_crit_freq, 'lp', fs=resp_fs, output='sos')
    
    # filter and mean subtract the data
    filtered = signal.sosfilt(sos, resp_array)
    filtered = filtered-np.mean(filtered)
    
    # perform the hilbert transform to get the instantaneous phase from the respiration trace
    analytic_signal = hilbert(filtered)
    instantaneous_phase = np.angle(analytic_signal)
    
    # shift the instantaneous phase so zero occurs at the start of inhalation
    instantaneous_phase_corrected_noshift = np.zeros(instantaneous_phase.shape)
    instantaneous_phase_corrected_noshift[instantaneous_phase<= .5*np.pi] = instantaneous_phase[instantaneous_phase<=.5*np.pi] + .5*np.pi
    instantaneous_phase_corrected_noshift[instantaneous_phase> .5*np.pi] = instantaneous_phase[instantaneous_phase > .5*np.pi] - 1.5*np.pi
    instantaneous_phase_corrected = instantaneous_phase_corrected_noshift + np.pi
    
    return instantaneous_phase_corrected

def upsample_resp(hilbert_resp):
    '''
    Upsamples the hilbert-transformed respiration data to derive the phase of spikes across the experiment.
    
    Parameters
    ----------
    hilbert_resp: hilbert transformed respiration trace
    
    Outputs
    -------
    upsampled_hilbert_resp: the upsampled respiration trace
    
    '''

    # linearly interpolate to upsample respiration data to neural data sampling rate
    x = np.arange(len(hilbert_resp))
    y = hilbert_resp
    interp_fn = scipy.interpolate.interp1d(x, y, kind='linear')
    # get the upsampled data 
    xnew = np.arange(0,len(hilbert_resp)-1,(1/15)) #here, upsample to 30 kHz, same as neural data sampling rate
    upsampled_hilbert_resp = interp_fn(xnew)
    return upsampled_hilbert_resp

def get_all_inhalations(upsampled_hilbert_resp):
    '''
    find the indices of all inhalations occuring across the experiment.
    
    Parameters
    ----------
    upsampled_hilbert_resp: the filtered respiration trace upsampled to 30 kHz
    
    Outputs
    -------
    inhalation_indices: the indices of all inhalations across the experiment
    
    '''
    
    # find all inhalations
    threshold = 1
    # find threshhold crossings 
    threshold_crossings = np.diff(upsampled_hilbert_resp> threshold, prepend=False)
    # find upward crossings 
    upward_crossings = (np.roll(upsampled_hilbert_resp,-1) < threshold)
    # find negative-going threshold crossings 
    threshold_idx_inhalations = np.argwhere(threshold_crossings & upward_crossings)[:,0]
    
    # find all inhalation-exhalation transitions
    threshold = np.pi
    # find threshhold crossings 
    threshold_crossings = np.diff(upsampled_hilbert_resp > threshold, prepend=False)
    # find upward crossings 
    upward_crossings = (np.roll(upsampled_hilbert_resp,-1) > threshold)
    # find negative-going threshold crossings 
    threshold_idx_cross = np.argwhere(threshold_crossings & upward_crossings)[:,0]

    # remove noise by selecting the first point at which respiration crosses threshold 
    phase_aligned_indices = np.empty_like(threshold_idx_inhalations)
    for resp_idx, resp in enumerate(threshold_idx_inhalations):
        locate_phase = threshold_idx_cross - resp
        try:
            min_index = np.where(locate_phase>0)[0][0]
            phase_aligned_indices[resp_idx] = threshold_idx_cross[min_index]
        except:
            pass

    #remove erroneously labeled inhalations from arrays
    erroneous_inhalations = np.diff(phase_aligned_indices) == 0
    erroneous_inhalations = np.insert(erroneous_inhalations,0,0)
    inhalation_indices = threshold_idx_inhalations[~erroneous_inhalations]
    
    return inhalation_indices

def get_pre_post_inhalation_mat(inhalation_idx, threshold_idx_inhalations, upsampled_hilbert_resp, conc = 2, n_pre_inhalations = 10, n_post_inhalations = 10, fs = 30000):
    '''
    Produces a matrix that contains the trials (11 odors x 15 trials) x breaths (pre and post-stimulus onset). These are all of the inhalation times surrounding the stimulus onset. 
    
    Parameters
    ----------
    PREXtimes_file: file containing the time of the first inhalation after each trial derived from Shiva's data. 
    threshold_idx_inhalations: all of the indices of inhalation times across the experiment. 
    
    Outputs
    -------
    inhalation_mat: a matrix of the 10 pre- and 10 post-odor delivery inhalations, plus the odor delivery inhalation for a total of n_pre + n_post + 1 inhalation times 
    trial_type_inhalation_idxs: The indices of the first inhalation after odor delivery for each trial. Should be an n_valves x n_trials matrix. 
    
    '''
    trial_type_inhalation_idxs = np.empty((len(inhalation_idx),inhalation_idx[0].shape[1]))
    for odor_idx, odor in enumerate(inhalation_idx):
        trial_type_inhalation_idxs[odor_idx,:] = (odor[0]*fs).astype(int)
        
    inhalation_idxs_ravel = np.sort(trial_type_inhalation_idxs.ravel())
    
    inhalation_mat = np.empty((len(inhalation_idxs_ravel),n_pre_inhalations + n_post_inhalations + 1))
    
    for trial_n, trial in enumerate(inhalation_idxs_ravel):
        # get all the inhalations that are >= trial onset
        post_inhalations = threshold_idx_inhalations[threshold_idx_inhalations>=trial]
        # get all inhalations occuring before trial onset
        pre_inhalations = threshold_idx_inhalations[threshold_idx_inhalations<trial]
        # first, populate the matrix with the 10 inhalations preceeding trial onset (indices 0:9)
        inhalation_mat[trial_n,0:n_pre_inhalations] = pre_inhalations[n_pre_inhalations*-1:]
        # next, populate the matrix with the inhalation after odor onset, and then the 10 inhalations following this breath (index 10:21)
        inhalation_mat[trial_n,n_pre_inhalations:(n_pre_inhalations + n_post_inhalations + 1)] = post_inhalations[0:n_post_inhalations + 1]
        
    return inhalation_mat, trial_type_inhalation_idxs
    

def get_pre_post_inhalation_mat_KB(inhalation_idx, threshold_idx_inhalations, conc = 2, n_pre_inhalations = 10, n_post_inhalations = 10, fs = 30000):
    '''
    Produces a matrix that contains the trials (11 odors x 15 trials) x breaths (pre and post-stimulus onset). These are all of the inhalation times surrounding the stimulus onset. 

    Parameters
    ----------
    PREXtimes_file: file containing the time of the first inhalation after each trial derived from Shiva's data. 
    threshold_idx_inhalations: all of the indices of inhalation times across the experiment. 

    Outputs
    -------
    inhalation_mat: a matrix of the 10 pre- and 10 post-odor delivery inhalations, plus the odor delivery inhalation for a total of n_pre + n_post + 1 inhalation times 
    trial_type_inhalation_idxs: The indices of the first inhalation after odor delivery for each trial. Should be an n_valves x n_trials matrix. 

    '''
    trial_type_inhalation_idxs = (inhalation_idx*fs).astype(int)

    inhalation_idxs_ravel = np.sort(trial_type_inhalation_idxs.ravel())

    inhalation_mat = np.empty((len(inhalation_idxs_ravel),n_pre_inhalations + n_post_inhalations + 1))

    for trial_n, trial in enumerate(inhalation_idxs_ravel):
        post_inhalations = threshold_idx_inhalations[threshold_idx_inhalations>=trial]
        pre_inhalations = threshold_idx_inhalations[threshold_idx_inhalations<trial]
        inhalation_mat[trial_n,0:n_pre_inhalations] = pre_inhalations[n_pre_inhalations*-1:]
        inhalation_mat[trial_n,n_pre_inhalations:(n_pre_inhalations + n_post_inhalations + 1)] = post_inhalations[0:n_post_inhalations + 1]

    return inhalation_mat, trial_type_inhalation_idxs

def format_spike_rasters(raster_align, conc_to_analyze = 2):
    '''
    Takes the .mat file containing trial-aligned rasters and returns a reformatted nested list of spike times. 
    List dimensions are valves x cells x trials x spikes 
    
    Parameters: 
    -----------
    raster_align_file: file path to the rasteralign .mat file
    conc_to_analyze: index of the concentration to analyze. 0 is lowest, 2 is highest. 
    
    Outputs:
    --------
    dimension_swap_odors: a nested list containing valves x cells x trials x spikes 
    
    '''
    
    all_rasters = [raster_align]
    
    # Do a bit of reformatting (only take specified conc. trials (index 0 is lowest, 2 is highest))
    cell_trials = []
    cell_trials_conc = []
    cell_trials_conc_odor = []
    cell_trials_conc_odor_raster = []
    cell_trials_conc_odor_raster_all = []
    for raster in all_rasters:
        for odor in raster:
            for conc_n, conc in enumerate(odor):
                if conc_n == conc_to_analyze:
                    for cell in conc:
                        for trial in cell:
                            cell_trials.append(trial[0][0])
                        cell_trials_conc.append(cell_trials)
                        cell_trials = []
                    #cell_trials_conc_odor.append(cell_trials_conc)
                    #cell_trials_conc = []
            cell_trials_conc_odor_raster.append(cell_trials_conc)
            cell_trials_conc = []
        cell_trials_conc_odor_raster_all.append(cell_trials_conc_odor_raster)
        cell_trials_conc_odor_raster = []
    
    valve_cells_expt = cell_trials_conc_odor_raster_all
    
    # concatenate across all cells within odor. Want an nodor x ncell x ntrials list
    dimension_swap = []
    dimension_swap_cells = []
    dimension_swap_odors = []
    #valve_cells_expt = np.array(valve_cells_expt)
    for odor in range(len(valve_cells_expt[0])):
        for cell in range(len(valve_cells_expt[0][0])):
            for trial in range(len(valve_cells_expt[0][0][0])):
                dimension_swap.append(valve_cells_expt[0][odor][cell][trial])
            dimension_swap_cells.append(dimension_swap)
            dimension_swap = []
        dimension_swap_odors.append(dimension_swap_cells)
        dimension_swap_cells = []
    
    return dimension_swap_odors

def reformat_time_aligned_rasters(n_odors_n_cells_raster):
    '''
    reformats the rasters output by the format_spike_rasters function into a nested list with n_cells x n_odors x n_trials x n_spikes
    
    Parameters: 
    -----------
    n_odors_n_cells_raster: a nested list containing odors x cells x trials x spikes 
    
    Outputs:
    --------
    formatted_rasters_odor: a nested list containing cells x valves x trials x spikes 
    '''
    
    formatted_rasters = []
    formatted_rasters_trial = []
    formatted_rasters_odor = []
    formatted_rasters_cell = []
    for expt in n_odors_n_cells_raster:
        for cell in range(len(expt[0])):
            for odor in range(len(expt)):
                for trial in range(len(expt[0][0])):
                    formatted_rasters.append(expt[odor][cell][trial])
                formatted_rasters_trial.append(formatted_rasters)
                formatted_rasters = []
            formatted_rasters_odor.append(formatted_rasters_trial)
            formatted_rasters_trial = []
        formatted_rasters_cell.append(formatted_rasters_odor)
        formatted_rasters_odor = []
        
    return formatted_rasters_cell 

def format_spike_rasters_KB(raster_align,valve_idxs, n_trials = 15):
    

    # Do a bit of reformatting
    trials = []
    cell_trials = []
    valve_cells = []
    valve_cells_expt = []
    for valve in raster_align:
        for cell_n, cell in enumerate(valve):
            for trial in cell:
                    try:
                        trials.append(trial[0][0])
                    except:
                        trials = []
            cell_trials.append(trials)
            trials = []
        valve_cells.append(cell_trials)
        cell_trials = []

    keep_valves = []
    for odor in range(len(valve_cells)):
        if odor in valve_idxs: # only some valves are active in these experiments, select active valves. 4 is Mineral oil. 
            keep_valves.append(valve_cells[odor][1:])

    all_first_trials = []
    first_trials = []
    for valve in keep_valves:
        for cell in valve:
            first_trials.append(cell[0:n_trials])
        all_first_trials.append(first_trials)
        first_trials = []
    
    return all_first_trials

def get_spike_counts(aligned_rasters_cat, min_val, max_val): 
    '''
    For each trial, get the number of spikes occuring in a specified time or phase window
    
    Parameters: 
    -----------
    aligned_rasters_cat: A nested list of rasters with dimensions n_cells x n_odors x n_trials
    min_val: the minimum value relative to trial start (0) over which to count spikes. Can be in time or phase coordinates. For time, should be in units of seconds, for phase should be in radians. 
    max_val: the max value relative to trial start (0) over which to count spikes. Can be in time or phase coordinates. For time, should be in units of seconds, for phase should be in radians. 
    
    Outputs:
    --------
    spike_count: a numpy array containing spike counts for each trial with dimensions n_cells x n_odors x n_trials
    
    '''
     
    n_total_cells = aligned_rasters_cat.shape[0]
    n_odors = aligned_rasters_cat.shape[1]
    n_trials = aligned_rasters_cat.shape[2]
    
    # Initialize array 
    spike_counts = np.empty((n_total_cells, n_odors, n_trials))
    
    # count the number of spikes in each trial 
    for cell_n, cell in enumerate(aligned_rasters_cat):
        for odor_n, odor in enumerate(cell):
            for trial_n, trial in enumerate(odor):
                spike_counts[cell_n, odor_n, trial_n] = len(trial[(trial>=min_val) & (trial<=max_val)])
    
    return spike_counts

def get_mean_spike_latencies(aligned_rasters_cat, min_val, max_val, phase_space = True): 
    '''
    For each trial, get the number of spikes occuring in a specified time or phase window
    
    Parameters: 
    -----------
    aligned_rasters_cat: A nested list of rasters with dimensions n_cells x n_odors x n_trials
    min_val: the minimum value relative to trial start (0) over which to average spike times. Can be in time or phase coordinates. For time, should be in units of seconds, for phase should be in radians. 
    max_val: the max value relative to trial start (0) over which to average spike times. Can be in time or phase coordinates. For time, should be in units of seconds, for phase should be in radians. 
    
    Outputs:
    --------
    mean_spike_latencies: a numpy array containing mean spike latencies for each trial with dimensions n_cells x n_odors x n_trials
    
    '''
     
    n_total_cells = aligned_rasters_cat.shape[0]
    n_odors = aligned_rasters_cat.shape[1]
    n_trials = aligned_rasters_cat.shape[2]
    
    # Initialize array 
    mean_spike_latencies = np.empty((n_total_cells, n_odors, n_trials))
    
    # count the number of spikes in each trial 
    for cell_n, cell in enumerate(aligned_rasters_cat):
        for odor_n, odor in enumerate(cell):
            for trial_n, trial in enumerate(odor):
                n_spikes = len(trial[(trial>=min_val) & (trial<=max_val)])
                if n_spikes>0:
                    if phase_space == True:
                        mean_spike_latencies[cell_n, odor_n, trial_n] = scipy.stats.circmean(trial[(trial>=min_val) & (trial<=max_val)])
                    else:
                        mean_spike_latencies[cell_n, odor_n, trial_n] = np.mean(trial[(trial>=min_val) & (trial<=max_val)])
                else:
                    mean_spike_latencies[cell_n, odor_n, trial_n] = np.nan
    
    return mean_spike_latencies

def get_time_aligned_psth(all_trial_rasters, window_std_ms = 5):
    '''
    Compute the time aligned psth by convolving the spike time histogram with a 10ms standard deviation gaussian window. 
    
    Parameters:
    -----------
    all_trial_rasters: list containing cells x valves x trials x spikes 
    
    Outputs: 
    --------
    valve_PSTHs: numpy array containing the smoothed firing rates for each cell x valve x trial.      
    '''
    # Compute the time aligned PSTHs for each odor valve
    valves = np.arange(len(all_trial_rasters))
    x = np.arange(-5,5,.002) #2 ms bins, 10 seconds of data
    valve_PSTHs = np.empty((len(all_trial_rasters[0]), len(all_trial_rasters), len(all_trial_rasters[0][0]), len(x)))
    for cell in range(len(all_trial_rasters[0])):
        for valve_n, valve in enumerate(valves):
            for entry_n, entry in enumerate(all_trial_rasters[valve][cell]):
                    # histogram with 2ms bins in a 10 second window surrounding odor delivery 
                    cellhist = np.histogram(entry, range = (-5, 5), bins = 5000)[0] 
                    # compute the instantaneous firing rate (in spks/s) in each bin 
                    inst_rate = cellhist*(1/.002) 
                    # convolve with a gaussian kernel with standard deviation defined by window_std
                    inst_rate = scipy.ndimage.gaussian_filter1d(inst_rate, (window_std_ms/1000)/.002) 
                    valve_PSTHs[cell, valve_n, entry_n, :] = inst_rate
                    
    return valve_PSTHs

def get_phase_aligned_rasters(all_trial_rasters, upsampled_hilbert_resp, trial_type_inhalation_idxs, inhalation_mat, resp_alignment_index = 10, fs = 30000):
    '''
    Compute the phase-warped rasters for all cell x odor x trial combinations 
    
    Parameters
    ----------
    all_trial_rasters: a nested list of rasters containing valves x cells x trials x spikes 
    upsampled_hilbert_resp: the upsampled (to 30kHz) hilbert-filtered respiration trace 
    all_inhalation_idxs: The indices of each inhalation after odor presentation. Should be an n_valves x n_trials matrix. 
    inhalation_mat: matrix of inhalation times surrounding each trial. Will be an (n_valves x n_trials) x n_resps matrix. 
    resp_alignment_index: the index of the respiration from the inhalation_mat that should be used for alignment. In this case, 10 is first inhalation occuring after odor presentation. 
    
    Outputs
    -------
    phase-warped_rasters: a list with dimensions n_cells x n_valves x n_trials x n_spikes. This contains spike aligned to the first inhalation after odor delivery in phase-coordinates. 
    
    '''
    
    # first, get the indices of each trial and sort by odor valve. 
    inhalation_idxs_ravel = np.sort(trial_type_inhalation_idxs.ravel())
    
    trial_type_indices = np.zeros_like(trial_type_inhalation_idxs)
    for odor_n, odor in enumerate(trial_type_inhalation_idxs):
        for trial_n, trial in enumerate(odor):
            trial_type_indices[odor_n,trial_n] = np.where(inhalation_idxs_ravel == trial)[0]
    
    # Now, arrange the inhalation matrix by the trial type indices. 
    # This should give us an n_valves x n_trials x n_respirations matrix
    inhalation_mat_trial_type = inhalation_mat[trial_type_indices.astype(int),:]
    
    # Here, we will compute the phase-warped rasters for all cells for all valve x trial combinations 
    n_cells = len(all_trial_rasters[0])
    phase_warped_rasters = []
    phase_warped_aligned_trial = []
    phase_warped_aligned_trial_odor = []
    i = 0
    for cell in range(n_cells):
        for conc in range(inhalation_mat_trial_type.shape[0]):
            for trial in range(inhalation_mat_trial_type.shape[1]):
                # For each trial, get the index of the start of the first inhalation after odor delivery (this is the resp alignment index)
                trial_first_inhale_coords = int(inhalation_mat_trial_type[conc,trial,resp_alignment_index])
                # Each of our rasters contains spike times aligned to trial onset. Instead, we want to derive the spike index in the experiment. 
                # To do this, we will multiply the spike times by the sampling rate, then add the index of the trial onset (first inhalation after odor delivery). 
                test_resp = ((all_trial_rasters[conc][cell][trial])*fs)+trial_first_inhale_coords
                # Now, we will derive the phase of each of these spikes from the hilbert transformed respiration trace. 
                phase_warped = upsampled_hilbert_resp[test_resp.astype(int)]
                last_spike = 0
                cycle = np.radians(360)
                new_cyc = 0
                phase_warped_aligned = []
                # Now, we will align the spikes in phase relative to the trial onset.
                # Here, the first inhalation after odor delivery is zero, the second is 6.28, while the inhalation before odor delivery is -6.28. 
                for spike_n, spike in enumerate(test_resp.astype(int)):
                    for resp in range(inhalation_mat_trial_type.shape[2]-1):
                        if (spike > inhalation_mat_trial_type[conc, trial, resp]) & (spike < inhalation_mat_trial_type[conc, trial, resp+1]):
                            phase_warped_aligned.append(phase_warped[spike_n] + (np.radians(360)*resp))
                phase_warped_aligned_trial.append(phase_warped_aligned-(np.radians(360)*resp_alignment_index))
            phase_warped_aligned_trial_odor.append(phase_warped_aligned_trial)
            phase_warped_aligned_trial = []
        phase_warped_rasters.append(phase_warped_aligned_trial_odor)
        phase_warped_aligned_trial_odor = []
    
    return phase_warped_rasters, inhalation_mat_trial_type

def get_phase_warped_PSTHs_old(phase_warped_rasters, inhalation_mat_trial_type, window_std = 5, fs = 30000): 
    '''
    Compute the phase-warped PSTHs from the phase-warped raster plots
    
    Parameters
    ----------
    phase_warped_rasters: a list with dimensions n_cells x n_valves x n_trials x n_spikes. This contains spike aligned to the first inhalation after odor delivery in phase-coordinates. 
 
    
    Outputs
    -------
    phase_warped_PSTH: 
    
    '''
    n_cells = len(phase_warped_rasters)
    n_concs = len(phase_warped_rasters[0])
    n_trials = len(phase_warped_rasters[0][0])

    mean_resp_duration = np.empty((n_concs, n_trials))
    for conc in range(n_concs):
        for trial in range(n_trials):
            mean_resp_duration[conc,trial] = np.mean(np.diff(inhalation_mat_trial_type[conc,trial]))/fs

    import scipy
    window_size = np.arange(-1000,1001,1)
    window = scipy.stats.norm.pdf(window_size, 0, window_std)
    window /= window.sum()
    window = window[750:1250]
    interval = [-60,60]
    binsize = .05
    bins = int((abs(interval[0])+interval[1])/binsize)
    phase_warped_PSTHs = np.zeros((n_cells,n_concs,n_trials,bins))
    for conc in range(n_concs):
        for cell in range(n_cells):
            for trial in range(n_trials):
                raster = phase_warped_rasters[cell][conc][trial]
                cellhist = np.histogram(raster, range = (interval[0], interval[1]), bins = bins)[0]
                inst_rate = cellhist*(1/binsize)
                inst_rate = np.convolve(window, inst_rate, mode='same')
                # This gives us spikes per radian. Now convert to spikes per cycle, then divide by average cycle time in this trial. 
                phase_warped_PSTHs[cell,conc,trial, 0:] = (inst_rate*np.radians(360))/mean_resp_duration[conc,trial]
    
    return phase_warped_PSTHs



def get_phase_warped_PSTHs(phase_warped_rasters, inhalation_mat_trial_type, window_std_ms = 5, fs = 30000): 
    '''
    Compute the phase-warped PSTHs (smoothed with a gaussian kernel) from the phase-warped raster plots
    
    Parameters
    ----------
    phase_warped_rasters: a nested list with dimensions n_cells x n_valves x n_trials x n_spikes. This contains spike aligned to the first inhalation after odor delivery in phase-coordinates. 
    inhalation_mat_trial_type: a matrix of the times of inhalation onset for all respirations in a window surrounding odor onset. This matrix will be used to estimate the firing rate on a trial by trial basis. 
    window_std_ms: the standard deviation (in ms) of the gaussian kernel used to smooth the PSTHs
    fs: neural data sampling rate
    
    Outputs
    -------
    phase_warped_PSTHs: a numpy array containing smoothed phase-warped firing rates with dimensions n_cells x n_valves x n_trials x n_timepoints. 
    
    '''
    n_cells = len(phase_warped_rasters)
    n_odors = len(phase_warped_rasters[0])
    n_trials = len(phase_warped_rasters[0][0])
    
    # for each trial, get the mean respiration duration in seconds
    mean_resp_duration_s = np.empty((n_odors, n_trials))
    for odor in range(n_odors):
        for trial in range(n_trials):
            mean_resp_duration_s[odor,trial] = np.mean(np.diff(inhalation_mat_trial_type[odor,trial]))/fs

    interval = [-60,60]
    binsize = .05 #bin size is .05 radians, 2400 total bins, each bin is 2.86 degrees 
    bins = int((abs(interval[0])+interval[1])/binsize) #2400 bins
    phase_warped_PSTHs = np.zeros((n_cells,n_odors,n_trials,bins))
    for odor in range(n_odors):
        for cell in range(n_cells):
            for trial in range(n_trials):
                raster = phase_warped_rasters[cell][odor][trial] 
                # get a histogram of spikes occuring over analysis window, binned in 2.86 degree bins (.05 radians)
                cellhist = np.histogram(raster, range = (interval[0], interval[1]), bins = bins)[0]
                # get the instantaneous rate in spikes/radian
                inst_rate = cellhist*(1/binsize) 
                # now, using the mean resp duration, we will use a filter (in radians) that estimates a gaussian kernel with an n ms standard deviation 
                sd_radians = (np.radians(360)*(window_std_ms/1000))/mean_resp_duration_s[odor,trial] # for each trial, estimate how many radians are traversed over the window_std period.
                # set filter sigma as sd_radians 
                inst_rate = scipy.ndimage.gaussian_filter1d(inst_rate, sd_radians/binsize) 
                # This gives us spikes per radian. Now convert to spikes per cycle, then divide by average cycle time in this trial. 
                phase_warped_PSTHs[cell,odor,trial, 0:] = (inst_rate*np.radians(360))/mean_resp_duration_s[odor,trial] 
    
    return phase_warped_PSTHs



def get_valve_times(valve_time_file):
    '''
    Parameters
    ----------
    valve_time_file: file path corresponding to all final valve switch times for all valve x conc combinations 
    
    Outputs
    -------
    valve_times: valve times derived from valve time file
    
    '''
    valve_times = scipy.io.loadmat(valve_time_file)['FVSwitchTimes']
    return valve_times

def flatten_valve_times(valve_times):
    '''
    Parameters
    ----------
    valve_times: valve x conc numpy array of valve on times
    
    Outputs
    -------
    flat_valve_times: flattened array of valve times sorted in ascending order
    '''
    all_times = []
    for valve_times in all_valve_times.ravel():
        times = valve_times[0]
        all_times.append(times)
    flat_valve_times = np.sort([time for times in all_times for time in times])
    return flat_valve_times 


def get_resp_scaling_factor(upsampled_hilbert_resp, bins):
    '''
    Compute the proportion of time spent in each respiration bin across the whole experiment
    
    Parameters
    ----------
    upsampled hilbert resp: the upsampled hilbert transformed respiration trace
    bins: the phase bins in radians 
    
    Outputs
    -------
    scaling factor: an array containing the proportion of time spent in each bin
    '''

    # to compute the scaling factor, get the proportion of timepoints in the respiration cycle falling within each bin. 
    scaling_factor = np.histogram(upsampled_hilbert_resp,bins = bins)[0]/np.sum(np.histogram(upsampled_hilbert_resp,bins = bins)[0])

    return scaling_factor

def get_time_in_phase_bins(upsampled_hilbert_resp, bins):
    '''
    Get the average amount of time spend in each respiration bin across the experiment. This can be used to convert spikes/phase bin into spikes/s
    
    Parameters:
    ----------
    upsampled hilbert resp: the upsampled hilbert transformed respiration trace
    bins: the phase bins in radians 
    
    Outputs:
    -------
    duration_in_resp_bin: array containing the average amount of time spent in each phase bin
    '''
    
    # this function finds the average duration of each respiration in the experiment, as well as the average duration of inhalations and exhalations.
    resp_duration, inh_duration, exh_duration = phase_analysis.get_resp_stats(upsampled_hilbert_resp)
    
    # get the proportion of time spent in each phase bin.
    scaling_factor = get_resp_scaling_factor(upsampled_hilbert_resp, bins)

    # multiply each scaling factor by the mean respiration duration to find average time in bin. 
    duration_in_resp_bin = scaling_factor*np.mean(resp_duration)

    return duration_in_resp_bin


def get_spontaneous_tuning(tsecs_spontaneous, upsampled_hilbert_resp, bins, fs = 30000):
    '''
    Get the spontaneous phase tuning of each cells, both scaled (normalized to max) and raw
    
    Parameters:
    -----------
    tsecs_spontaneous: list of spontaneous rasters (not occuring during odor presentation)
    upsampled hilbert resp: the upsampled hilbert transformed respiration trace
    bins: the phase bins in radians 
    
    Outputs:
    --------
    resp_tuning_scaled: the normalized (to max) spontaneous phase locking
    resp_tuning_per_inh_scaled: the spontaneous phase locking of each cell in spikes/s
    time_in_phase_bins: the mean duration of each phase bin across the experiment
    
    '''
    # returns an array containing the proportion of time spent in each phase bin
    scaling_factor = get_resp_scaling_factor(upsampled_hilbert_resp, bins)
        
    # returns an array containing the average duration of each phase bin
    time_in_phase_bins = get_time_in_phase_bins(upsampled_hilbert_resp, bins)
        
    # get the number of respirations occuring over the analysis period (from the last frame on time to end of experiment)
    n_resps = len(phase_analysis.get_inhalation_indices(upsampled_hilbert_resp))

    # to compute the scaled spontaneous tuning curves, divide the phase histogram generated for each cell by the scaling factor for each bin, then normalize to the max bin.
    nbins = len(bins)-1
    ncells = len(tsecs_spontaneous)
    resp_tuning_scaled = np.empty((ncells, nbins))
    resp_tuning_per_inh_scaled = np.empty((ncells, nbins))
    for cell in range(len(tsecs_spontaneous)):
        analyze_idx = (tsecs_spontaneous[cell]*fs).astype(int)
        test_hist = np.histogram(upsampled_hilbert_resp[analyze_idx[analyze_idx<len(upsampled_hilbert_resp)]].ravel(), bins = bins)[0]
        resp_tuning_scaled[cell,:] = ((test_hist/scaling_factor/(np.max(test_hist/scaling_factor))))
        # divide the histogram of total binned spikes by the number of respirations to get spikes per phase bin. 
        # multiply by (1/resp bin duration) to get spikes/s in bin. 
        resp_tuning_per_inh_scaled[cell,:] = ((test_hist/n_resps)*(1/time_in_phase_bins))

    return resp_tuning_scaled, resp_tuning_per_inh_scaled, time_in_phase_bins


def filter_spike_times(spike_times, stim_times, spike_window = 2):
    '''
    Filter out spikes occuring during odor delivery so that only spontaneous spiking remains
    
    Parameters:
    -----------
    
    spike_times: the list of spike times for each cell across the experiment
    stim_times: flattened list of all odor on times throughout the experiment
    spike_window: window after odor on time that should be excised from spike time array 
    
    Outputs: 
    --------
    filtered_spike_times: array of spike times for each cell, with spikes during odor on periods removed

    '''
    filtered_spike_times = []
    for spike_time in spike_times:
        spike_time = spike_time[0]
        if not np.where(((spike_time-stim_times)<spike_window) & (spike_time-stim_times>0))[0]:
            filtered_spike_times.append(spike_time)
    return np.array(filtered_spike_times)

def filter_spike_times_KB(spike_times, stim_times, spike_window = 2, time_limit = []):
    '''
    Filter out spikes occuring during odor delivery so that only spontaneous spiking remains
    
    Parameters:
    -----------
    
    spike_times: the list of spike times for each cell across the experiment
    stim_times: flattened list of all odor on times throughout the experiment
    spike_window: window after odor on time that should be excised from spike time array 
    
    Outputs: 
    --------
    filtered_spike_times: array of spike times for each cell, with spikes during odor on periods removed

    '''
    filtered_spike_times = []
    for spike_time in spike_times:
        if not np.where(((spike_time-stim_times)<spike_window) & (spike_time-stim_times>0))[0]:
            filtered_spike_times.append(spike_time)
    if time_limit:
        filtered_spike_times = np.array(filtered_spike_times)
        filtered_spike_times = filtered_spike_times[filtered_spike_times<time_limit]
    return np.array(filtered_spike_times)

def get_phase_locking_bootstrap(upsampled_hilbert_resp, tsecs_spontaneous, bins, n_shuffs = 100, fs = 30000):
    '''
    Compute the expected distribution of spikes using shuffled respiration data. We will use this null distribution to assess whether cells are significantly phase-locked.  
    
    Parameters:
    -----------
    upsampled_hilbert_resp: the array of Hilbert-transformed respiration data upsampled to 30kHz
    tsecs_spontaneous: the list of spontaneous spike times (spikes during odor-on period filtered out) for each cell 
    bins: array of bin edges ranging from 0 to 6.28 radians
    n_shuffs: number of shuffles to perform when computing the bootstrapped spike-phase distribution for each cell 
    fs: the neural data sampling rate 
    
    Outputs: 
    --------
    phase_locked_boot_dist: a matrix size n_cells x n_shuffles x n_bins containing the spike-phase histograms derived using shuffled respirations. 

    
    '''
    # get each individual cycle and separate out into a list
    inhalations = phase_analysis.get_inhalation_indices(upsampled_hilbert_resp)
    shuff_inhalations = [upsampled_hilbert_resp[0:inhalations[0]]]
    for idx in range(len(inhalations)-1):
        shuff_inhalations.append(upsampled_hilbert_resp[inhalations[idx]:inhalations[idx+1]])
    shuff_inhalations.append(upsampled_hilbert_resp[inhalations[idx+1]:])
    # shuffle the list of cycles to get out an array with shuffled inhalations. This will be used to calculate the phase-locking confidence interval (Method used in Fukunaga, 2012)
    n_cells = len(tsecs_spontaneous)
    hist_counts = len(bins)-1
    phase_lock_boot_dist = np.empty((n_cells, n_shuffs, hist_counts))
    shuf_array = np.empty_like(upsampled_hilbert_resp)
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
        for cell_n, cell in enumerate(tsecs_spontaneous): 
            analyze_idx = (cell*fs).astype(int)
            shuff_hist = np.histogram(shuf_array[analyze_idx[analyze_idx<len(upsampled_hilbert_resp)]], bins = bins)[0]
            phase_lock_boot_dist[cell_n, shuffle, :] = shuff_hist
    return phase_lock_boot_dist


def get_phase_locked_cells(tsecs_spontaneous, upsampled_hilbert_resp, phase_lock_boot_dist, bins, fs = 30000, spikes_exceeding_thresh = 5):
    '''
     Parameters:
    -----------
    tsecs_spontaneous: the list of spontaneous spike times (spikes during odor-on period filtered out) for each cell 
    upsampled_hilbert_resp: the array of Hilbert-transformed respiration data upsampled to 30kHz
    phase_locked_boot_dist: a matrix size n_cells x n_shuffles x n_bins containing the spike-phase histograms derived using shuffled respirations.
    bins: array of bin edges ranging from 0 to 6.28 radians
    fs: the neural data sampling rate
    spikes_exceeding_thresh: cutoff percentage of spikes exceeding the 95% confidence interval needed for a cell to be classified as phase-locked.
    
    Outputs: 
    --------
    all_radius_low: the lower bound of the 95% confidence interval for phase locking computed using shuffled respirations.
    all_radius_high: the upper bound of the 95% confidence interval for phase locking computed using shuffled respirations.
    all_phase_hist: the true phase histogram calculated using spontaneous spikes for each cell.
    tuned: an n_cells array indicating whether a cell is spontaneously phase locked given the criteria of at least 5% of spikes exceeding the 95% C.I. and
           at least 2 consecutive bins exceeding the C.I. in either direction. 
    total_percent_spikes_exceeding: an n_cells array containing the total percentage of spikes exceeding the 95% confidence interval for each cell. 
    
    '''
    n_cells = len(tsecs_spontaneous)
    phase_locked = np.zeros(n_cells)
    total_percent_spikes_exceeding = np.zeros(n_cells)
    # radius low is the lower bound of the 95% CI
    all_radius_low = np.empty((n_cells,len(bins)-1))
    # radius high in the upper bound of the 95% CI
    all_radius_high = np.empty((n_cells,len(bins)-1))
    all_phase_hist = np.empty((n_cells,len(bins)-1))
    for cell in range(n_cells):
        for bin_ in range(len(bins)-1):
            # get the observation in the 2.5th and 97.5th percentile
            all_radius_low[cell, bin_] = np.percentile(phase_lock_boot_dist[cell,:,bin_],2.5)
            all_radius_high[cell, bin_] = np.percentile(phase_lock_boot_dist[cell,:,bin_],97.5)
        # get the spike times occuring during the spontaneous analysis period 
        analyze_idx = (tsecs_spontaneous[cell]*fs).astype(int)
        # bin the spike times by respiration phase
        hist = np.histogram(upsampled_hilbert_resp[analyze_idx[analyze_idx<len(upsampled_hilbert_resp)]], bins = bins)[0]
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
        # if >5% of spikes exceed the confidence intervals across bins, and there are two or more consecutive bins that exceed the CI, count the cell as tuned. 
        if (total_diff > spikes_exceeding_thresh) & (consecutive_bins == 1):
            phase_locked[cell] = 1
        # store the total percentage of spikes exceeding the confidence interval as a metric of tuning strength
        total_percent_spikes_exceeding[cell] = percent_exceed + percent_min
    # return the upper and lower bounds of the CI for each cell, the binary tuning array, and the array containing percentage of spikes exceeding the CI. 
    return all_radius_low, all_radius_high, all_phase_hist, phase_locked, total_percent_spikes_exceeding



def load_KB_spiketimes(spike_file):
    '''
    load the data from the HDF5 files from Kevin Bolding's experiments containing spike times 
    
    Parameters:
    -----------
    spike_file: the .st HDF5 file containing the spyking circus output 
    
    Outputs:
    --------
    all_spiketimes: the spiketimes for each cell throughout the course of the experiment 
    
    '''
    f = h5py.File(spike_file,'r') 
    spiketimes = f['SpikeTimes']
    tsecs = spiketimes['tsec']
    all_spiketimes = []
    for unit in range(len(tsecs[0])):
        st = tsecs[0][unit]
        obj = f[st][:][0]
        all_spiketimes.append(np.array(obj))
    return all_spiketimes 



def get_PSTH_consistency(cells_activated, valve_PSTHs_analyze):
    '''
    For each pair of neurons i and j, we computed the Euclidian distance dif between the PETHs of neuron i in the first condition and neuron j in the second condition. To account for
    differences in firing rates each PETH was normalized between 0 and 1.The PETH consistency measure of neuron i was defined to be the percentage of all other neurons j for which dii < dij.

    Parameters 
    ----------
    cells_activated: an n_cells x n_valves matrix. Value set to 1 if cell is significantly activated with respect to mineral oil response. 
    valve_PSTHs_analyze: the trial averaged PSTHs for each cell over the response window period
    
    Outputs
    -------
    similarity_mats: a list of n_cells x n_cells binary matrices. 1 indicates that the dii<dij. 0 indicates that dii>dij. 
    
    '''
    similarity_mats = []
    distance_mats = []
    all_distances = []
    neuron_distance = []
    test_similarity_mat = []
    n_valves = valve_PSTHs_analyze.shape[1]
    # loop through each pair of odors 
    for odor1 in range(1,n_valves):
        for odor2 in range(1,n_valves):
            # find the indices of cells that respond significantly to both odors
            multi_tuned_idx = np.where(np.sum(cells_activated[:,[odor1,odor2]],1)>1)[0]
            similarity_mat = np.zeros((len(multi_tuned_idx), len(multi_tuned_idx)))
            distance_matrix = np.zeros((len(multi_tuned_idx), len(multi_tuned_idx)))
            # Now loop through the significantly responsive cells for all odor pairs 
            neuron_PSTH_distance = []
            for n1, cell1 in enumerate(valve_PSTHs_analyze[multi_tuned_idx,odor1,:]):
                #First, find the distance between the same cell's response to odor1 and odor2
                cell1_norm_odor1 = cell1/np.max(cell1)
                cell1_norm_odor2 = valve_PSTHs_analyze[multi_tuned_idx[n1], odor2,:]/np.max(valve_PSTHs_analyze[multi_tuned_idx[n1], odor2,:])
                self_distance = np.linalg.norm((cell1_norm_odor1-cell1_norm_odor2))
                # Now, loop through the rest of the cells and calculate the distance between cell 1's response to odor 1 and cell 2's response to odor 2. 
                for n2, cell2 in enumerate(valve_PSTHs_analyze[multi_tuned_idx,odor2,:]):
                    cell1_norm = cell1/np.max(cell1)
                    cell2_norm = cell2/np.max(cell2)
                    PSTH_distance = np.linalg.norm((cell1_norm-cell2_norm))
                    if (self_distance<PSTH_distance):
                        similarity_mat[n1,n2] = 1
                    if n1 == n2:
                        similarity_mat[n1,n2] = np.nan
            # Append the matrices for all odor pairs 
            if odor1 != odor2: 
                similarity_mats.append(similarity_mat)
            distance_mats.append(distance_matrix)
            
    return similarity_mats


def get_PSTH_consistency_shuff(cells_activated, valve_PSTHs_analyze):
    '''
    Compute PSTH similarity as defined above, but shuffle the indices between cells. 
    
    Parameters 
    ----------
    cells_activated: an n_cells x n_valves matrix. Value set to 1 if cell is significantly activated with respect to mineral oil response. 
    valve_PSTHs_analyze: the trial averaged PSTHs for each cell over the response window period
    
    Outputs
    -------
    similarity_mats: a list of n_cells x n_cells binary matrices. 1 indicates that the dii<dij. 0 indicates that dii>dij. 
    
    '''
    similarity_mats = []
    all_distances = []
    neuron_distance = []
    n_valves = valve_PSTHs_analyze.shape[1]
    # loop through each pair of odors 
    for odor1 in range(1,n_valves):
        for odor2 in range(1,n_valves):
            # find the indices of cells that respond significantly to both odors
            multi_tuned_idx = np.where(np.sum(cells_activated[:,[odor1,odor2]],1)>1)[0]
            multi_tuned_idx_rand = np.copy(multi_tuned_idx)
            np.random.shuffle(multi_tuned_idx_rand)
            similarity_mat = np.zeros((len(multi_tuned_idx), len(multi_tuned_idx)))
            # Now loop through the significantly responsive cells for all odor pairs 
            neuron_PSTH_distance = []
            for n1, cell1 in enumerate(valve_PSTHs_analyze[multi_tuned_idx,odor1,:]):
                #First, find the distance between the same cell's response to odor1 and odor2. Except in this case, indices are shuffled so cells are different. 
                cell1_norm_odor1 = cell1/np.max(cell1)
                cell1_norm_odor2 = valve_PSTHs_analyze[multi_tuned_idx_rand[n1], odor2,:]/np.max(valve_PSTHs_analyze[multi_tuned_idx_rand[n1], odor2,:])
                self_distance = np.linalg.norm((cell1_norm_odor1-cell1_norm_odor2))
                # Now, loop through the rest of the cells and calculate the distance between responses to odor 1 and odor 2. 
                for n2, cell2 in enumerate(valve_PSTHs_analyze[multi_tuned_idx_rand,odor2,:]):
                    cell1_norm = cell1/np.max(cell1)
                    cell2_norm = cell2/np.max(cell2)
                    PSTH_distance = np.linalg.norm((cell1_norm-cell2_norm))
                    if (self_distance<PSTH_distance):
                        similarity_mat[n1,n2] = 1
                    if n1 == n2:
                        similarity_mat[n1,n2] = np.nan
            # Append the matrices for all odor pairs 
            if odor1 != odor2: 
                similarity_mats.append(similarity_mat)
            
    return similarity_mats


def get_PSTH_consistency_spont(cells_activated, valve_PSTHs_analyze, spontaneous_PSTHs):
    '''
    Quantify the extent to which spontaneous PSTHs predict a given cell's odor response. 
    
    Parameters 
    ----------
    cells_activated: an n_cells x n_valves matrix. Value set to 1 if cell is significantly activated with respect to mineral oil response. 
    valve_PSTHs_analyze: the trial averaged PSTHs for each cell over the response window period
    spontaneous_PSTHs: the trial averaged PSTHs for each cell over the spontaneous window period
    
    Outputs
    -------
    similarity_mats: a list of n_cells x n_cells binary matrices. 1 indicates that the dii<dij. 0 indicates that dii>dij. 
    
    '''
    similarity_mats = []
    all_distances = []
    neuron_distance = []
    n_valves = valve_PSTHs_analyze.shape[1]
    # loop through each odor 
    for odor1 in range(1,n_valves):
        # find the indices of cells that respond significantly to both odors
        multi_tuned_idx = np.where(cells_activated[:,odor1]>0)[0]
        similarity_mat = np.zeros((len(multi_tuned_idx), len(multi_tuned_idx)))
        # Now loop through the significantly responsive cells for all odor pairs 
        neuron_PSTH_distance = []
        for n1, cell1 in enumerate(valve_PSTHs_analyze[multi_tuned_idx,odor1,:]):
            #First, find the distance between the same cell's response to odor1 and the spontaneous psth
            cell1_norm_odor1 = cell1/np.max(cell1)
            cell1_norm_odor2 = spontaneous_PSTHs[multi_tuned_idx[n1], :]/np.max(spontaneous_PSTHs[multi_tuned_idx[n1], :])
            self_distance = np.linalg.norm((cell1_norm_odor1-cell1_norm_odor2))
            # Now, loop through the rest of the cells and calculate the distance between responses to odor 1 and the spontaneous psth 
            for n2, cell2 in enumerate(spontaneous_PSTHs[multi_tuned_idx,:]):
                cell1_norm = cell1/np.max(cell1)
                cell2_norm = cell2/np.max(cell2)
                PSTH_distance = np.linalg.norm((cell1_norm-cell2_norm))
                if (self_distance<PSTH_distance):
                    similarity_mat[n1,n2] = 1
                if n1 == n2:
                    similarity_mat[n1,n2] = np.nan
        # Append the matrices for all odor pairs 
        similarity_mats.append(similarity_mat)
    return similarity_mats


def get_PSTH_consistency_spont_shuff(cells_activated, valve_PSTHs_analyze, spontaneous_PSTHs):
    '''
    Quantify the extent to which spontaneous PSTHs predict a given cell's odor response using shuffled cell indices.
    
    Parameters 
    ----------
    cells_activated: an n_cells x n_valves matrix. Value set to 1 if cell is significantly activated with respect to mineral oil response. 
    valve_PSTHs_analyze: the trial averaged PSTHs for each cell over the response window period
    
    Outputs
    -------
    similarity_mats: a list of n_cells x n_cells binary matrices. 1 indicates that the dii<dij. 0 indicates that dii>dij. 
    
    '''
    similarity_mats = []
    all_distances = []
    neuron_distance = []
    n_valves = valve_PSTHs_analyze.shape[1]
    # loop throught each pair of odors 
    for odor1 in range(1,n_valves):
        # find the indices of cells that respond significantly to both odors
        multi_tuned_idx = np.where(cells_activated[:,odor1]>0)[0]
        multi_tuned_idx_rand = np.copy(multi_tuned_idx)
        np.random.shuffle(multi_tuned_idx_rand)
        similarity_mat = np.zeros((len(multi_tuned_idx), len(multi_tuned_idx)))
        # Now loop through the significantly responsive cells for all odor pairs 
        neuron_PSTH_distance = []
        for n1, cell1 in enumerate(valve_PSTHs_analyze[multi_tuned_idx,odor1,:]):
            #First, find the distance between the same cell's response to odor1 and the spontaneous psth, except in this case, the comparison cell is randomized
            cell1_norm_odor1 = cell1/np.max(cell1)
            cell1_norm_odor2 = spontaneous_PSTHs[multi_tuned_idx_rand[n1], :]/np.max(spontaneous_PSTHs[multi_tuned_idx_rand[n1], :])
            self_distance = np.linalg.norm((cell1_norm_odor1-cell1_norm_odor2))
            # Now, loop through the rest of the cells and calculate the distance between responses to odor 1 and odor 2. 
            for n2, cell2 in enumerate(spontaneous_PSTHs[multi_tuned_idx_rand,:]):
                cell1_norm = cell1/np.max(cell1)
                cell2_norm = cell2/np.max(cell2)
                PSTH_distance = np.linalg.norm((cell1_norm-cell2_norm))
                if (self_distance<PSTH_distance):
                    similarity_mat[n1,n2] = 1
                if n1 == n2:
                    similarity_mat[n1,n2] = np.nan
        # Append the matrices for all odor pairs 
        similarity_mats.append(similarity_mat)
    return similarity_mats