#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy import signal
import pickle as pkl
import PGanalysis
import phase_analysis

def save_phase_raster_psth(raster_save_path, psth_save_path, datafile, mtrigger_file, expt_metadata_path, expt_type = 0, save_data = True, return_output = False):
    '''
    Saves rasters and PSTHs aligned to trigger onset for each stimulus type. 

    Parameters 
    ----------
    raster_save_path: path of raster save location

    psth_save_path: path of psth save location 

    datafile: list of HDF5 files with spike arrays 

    mtrigger_file: the file with mightex trigger array extracted in pre-processing step
    
    expt_metadata_path: path to the experiment metadata file path to the list of rasters for each cell extracted during phase pre-processing 
    '''
    tsecs = phase_analysis.get_tsecs(datafile)
    mtrigger_array = PGanalysis.get_events(mtrigger_file)
    with open(expt_metadata_path, 'rb') as f:
        expt_metadata = pkl.load(f)
    frame_on_time_s = phase_analysis.get_frame_on_time_s(mtrigger_array) 
    # get the time in seconds when each trial (DMD frame) was presented 
    if expt_type == 0:
        frame_on_time_s = phase_analysis.get_frame_on_time_s(mtrigger_array)         
    # if expt_type variable = 1, we will only get the frame on times from the first half of the recording, generally corresponding to contra-occlusion. 
    elif expt_type == 1:
        frame_on_time_s = phase_analysis.get_frame_on_time_s(mtrigger_array)
        print(len(frame_on_time_s))
        frame_on_time_s = frame_on_time_s[0:int((len(frame_on_time_s)/2))]
    # if expt_type variable = 2, we will only get the frame on times from the second half of the recording, generally corresponding to ipsi-occlusion. 
    elif expt_type == 2:
        frame_on_time_s = phase_analysis.get_frame_on_time_s(mtrigger_array)
        frame_on_time_s = frame_on_time_s[int((len(frame_on_time_s)/2)):]
    trial_type_indices = phase_analysis.get_trial_type_indices(expt_metadata)
    cell_align_raster = PGanalysis.get_cell_aligned_raster(tsecs, frame_on_time_s)
    trial_type_raster,_,trial_type_PSTH, PSTH_timepoints = PGanalysis.get_raster_psth(cell_align_raster, trial_type_indices)
    if save_data == True:
        np.save(raster_save_path, trial_type_raster, allow_pickle = True)
        np.save(psth_save_path, np.array(trial_type_PSTH), allow_pickle = True)
    if return_output == True: 
        return trial_type_raster, trial_type_PSTH
    
    
def save_instantaneous_phase(resp_phase_save_path, resp_file, fs = 30000, chunk_size = 5):
    '''
    Saves the instantaneous phase angle of the respiration tract computed using the Hilbert transform. 

    Parameters 
    ----------
    resp_phase_save_path: path to save instantaneous phase data
    
    resp_file: the respiration data extracted from continuous.dat file. 
    '''
    chunk_size = chunk_size*fs
    resp_array = PGanalysis.get_events(resp_file)
    sos = signal.butter(1, 10, 'lp', fs=30000, output='sos')
    n_chunks = int(len(resp_array)/chunk_size)
    instantaneous_phase = np.empty(len(resp_array))
    for time_chunk in range(n_chunks):
        chunk_start = time_chunk*chunk_size
        chunk_end = (time_chunk+1)*chunk_size
        filtered = signal.sosfilt(sos, resp_array[chunk_start:chunk_end])
        filtered_chunk = filtered -np.mean(filtered)
        analytic_signal_chunk = hilbert(filtered_chunk)
        instantaneous_phase_chunk = np.angle(analytic_signal_chunk)
        instantaneous_phase[chunk_start:chunk_end] = instantaneous_phase_chunk
    if (len(filtered)%n_chunks)>0:
        filtered = signal.sosfilt(sos, resp_array[chunk_end:])
        filtered_chunk = filtered -np.mean(filtered)
        analytic_signal_chunk = hilbert(filtered_chunk)
        instantaneous_phase_chunk = np.angle(analytic_signal_chunk)
        instantaneous_phase[chunk_end:] = instantaneous_phase_chunk

    instantaneous_phase_corrected_noshift = np.zeros(instantaneous_phase.shape)
    instantaneous_phase_corrected_noshift[instantaneous_phase<= .5*np.pi] = instantaneous_phase[instantaneous_phase<=.5*np.pi] + .5*np.pi
    instantaneous_phase_corrected_noshift[instantaneous_phase> .5*np.pi] = instantaneous_phase[instantaneous_phase > .5*np.pi] - 1.5*np.pi
    instantaneous_phase_corrected = instantaneous_phase_corrected_noshift + np.pi

    np.save(resp_phase_save_path, instantaneous_phase_corrected, allow_pickle = True)
    
    return resp_array, instantaneous_phase_corrected 

