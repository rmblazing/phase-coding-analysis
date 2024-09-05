#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
sys.path.append(r'C:\Users\rmb55\most_updated_pattern_stim\pattern_stim_analysis\paper\pattern_stim_code\analysis_packages')
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import kruskal
import pickle as pkl
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import permutation_test_score
from sklearn.feature_selection import mutual_info_classif
import PGanalysis
import phase_analysis

class Phase_data_analysis:
    def __init__(self, mtrigger_file, expt_metadata_path, trial_type_psth_array_path, trial_type_raster_array_path, resp_phase_path, expt_type = 0, fs = 30000):
        '''
        Parameters 
        ----------
        mtrigger_file: the file with mightex trigger array extracted in pre-processing step

        expt_metadata_path: path to the experiment metadata file

        trial_type_psth_array_path: path to the psth array extracted during phase pre-processing

        trial_type_raster_array_path: path to the list of rasters for each cell extracted during phase pre-processing 
        '''
    
        # get the trigger times
        self.mtrigger_array = PGanalysis.get_events(mtrigger_file)

        # get the time in seconds when each trial (DMD frame) was presented 
        if expt_type == 0:
            self.frame_on_time_s = phase_analysis.get_frame_on_time_s(self.mtrigger_array) 
        
        # if expt_type variable = 1, we will only get the frame on times from the first half of the recording, corresponding to contra-occlusion. 
        elif expt_type == 1:
            self.frame_on_time_s = phase_analysis.get_frame_on_time_s(self.mtrigger_array)
            self.frame_on_time_s = self.frame_on_time_s[0:int((len(self.frame_on_time_s)/2))]
            
        # if expt_type variable = 2, we will only get the frame on times from the second half of the recording, corresponding to ipsi-occlusion. 
        elif expt_type == 2:
            self.frame_on_time_s = phase_analysis.get_frame_on_time_s(self.mtrigger_array)
            self.frame_on_time_s = self.frame_on_time_s[int((len(self.frame_on_time_s))/2):]

        # load metadata
        with open(expt_metadata_path, 'rb') as f:
            self.expt_metadata = pkl.load(f)

        # sort the trigger indices by which spot was triggered. Array with n_spots x n_trials_per_spot. 
        self.trial_type_indices = phase_analysis.get_trial_type_indices(self.expt_metadata)

        # load the psth array and the raster array made during the phase preprocessing step. 
        self.trial_type_psth_array = np.load(trial_type_psth_array_path, allow_pickle = True)
        self.trial_type_raster_array = np.load(trial_type_raster_array_path, allow_pickle = True)

        # load the preprocessed instantaneous respiration phase angle data
        self.instantaneous_phase_corrected = np.load(resp_phase_path, allow_pickle = True)
        

        # bin each trial by the respiration phase in which it occured. Also output the bins used for this analysis.
        self.bin_indices_all, self.bins = phase_analysis.get_bin_indices(self.trial_type_indices, self.expt_metadata, self.frame_on_time_s, 
                                                               self.instantaneous_phase_corrected)

        # get the spike counts over the first 300ms for each trial. Also get the average spike rate over the first 300ms for each trial.
        print(self.trial_type_psth_array.shape)
        self.trial_type_spike_counts, self.trial_type_spike_rate = phase_analysis.get_trial_type_spike_counts(self.trial_type_raster_array, self.trial_type_psth_array)

        # bin the spike counts for each trial of each stimulus by phase. Output is a list that is n_stim x n_bins x n_trials_per_bin.
        # note that n_trials_per_bin is different for each bin because stimulation is stochastic with respect to phase. 
        self.trial_type_phase_binned_spike_counts = phase_analysis.bin_spike_counts_by_phase(self.trial_type_spike_counts, self.bin_indices_all)

        self.n_cells = self.trial_type_psth_array.shape[0]
        self.n_stim = self.trial_type_psth_array.shape[1]
        self.n_bins = len(self.bins)-1

    def save_spike_count_decoder_data(self, save_path):
        # save the spike counts and bin indices for use in decoding analyses. 
        processed_phase_data = {'trial_type_spike_counts':self.trial_type_spike_counts, 'bin_indices_all':self.bin_indices_all, 'n_stim':self.n_stim, 'n_bins':self.n_bins}
        np.save(save_path, [processed_phase_data], allow_pickle = True)
        return processed_phase_data
        
    def save_cell_stim_phase_tuning(self, save_path):
        # save the tuning curves for all cell x stim combinations
        phase_tuning_by_cell = np.empty((self.n_stim, self.n_bins, self.n_cells))
        for cell in range(self.n_cells):
            all_psth_phase_mean_norm = phase_analysis.get_phase_tuning_by_stim(cell, self.bins, self.trial_type_psth_array, self.bin_indices_all, plot =                                                                                      False, norm = False)
            phase_tuning_by_cell[:,:,cell] = all_psth_phase_mean_norm
        np.save(save_path, [phase_tuning_by_cell], allow_pickle = True)
        return phase_tuning_by_cell
    
    def get_phase_stats(self, stats_save_path, kruskal_p_thresh = .01, rank_sum_p_thresh = .0025):  
        '''
        Outputs
        -------
        a dictionary with relevant values for analyzing phase tuning data. The fields include: 

        output_phase_stats = {'Kruskal_pval_thresh':kruskal_pval_thresh, 'Mann_Whitney_responsive_cell_bins': responsive_cell_bins, 'all_stim_resp_pref_phase':all_stim_resp_pref_phase, 'all_stim_time_to_50':all_stim_rise_times, 'stimulation_phase_preference':stimulation_phase_preference}

        - Kruskal_pval_thresh: nstim x ncells array denoting whether a particular stim x cell combination exhibits significant phase tuning. 

        - Mann_Whitney_responsive_cell_bins: nstim x nbins x ncells array denoting whether each stim x bin x cell combination is signifcantly activated or suppressed relative to its blank bin x cell counterpart. 

        - all_stim_resp_pref_phase: preferred reponse phase of all significant stim x bin x cell combinations. 

        - all_stim_time_to_50: time to 50% of max response for all significant stim x bin x cell combinations.

        - stimulation_phase_preference: preferred stimulation phase (bin of max response in degrees) of each significant stim x cell combination. 

        '''
        # run kruskal-wallis test to identify significantly tuned cells
        kruskal_pval_thresh = phase_analysis.kruskal_wallis_phase_tuning(self.trial_type_phase_binned_spike_counts, p_thresh = kruskal_p_thresh)

        # running 20 different significance tests for trials at each bin compared to the blank stimulus in the same bin. 
        # Bonferroni corrected p-value is .05/20 = .0025 
        responsive_cell_bins = phase_analysis.rank_sum_test(self.trial_type_phase_binned_spike_counts, p_thresh = rank_sum_p_thresh)
        activated_cell_bins = responsive_cell_bins['activated']
        suppressed_cell_bins = responsive_cell_bins['suppressed']

        stim_activated_thresh = np.sum(activated_cell_bins,1)

        activated_tuned_cells = np.zeros(self.n_stim)+np.nan
        activated_nontuned_cells = np.zeros(self.n_stim)+np.nan

        # find cells that have at least one significantly activated bin and are also significantly tuned (kruskal p<.01, mann-whitney p<.0025)
        for stim in range(self.n_stim):
            activated_tuned_cells[stim] = len(np.where((kruskal_pval_thresh[stim,:]>0)&(stim_activated_thresh[stim,:]>0))[0])

        # find cells that have at least one significantly activated bin and are not significantly tuned (kruskal p<.01, mann-whitney p<.0025)
        activated_nontuned_cells = np.zeros(self.n_stim)+np.nan
        for stim in range(self.n_stim):
            activated_nontuned_cells[stim] = len(np.where((kruskal_pval_thresh[stim,:]==0)&(stim_activated_thresh[stim,:]>0))[0])

        stim_suppressed_thresh = np.sum(suppressed_cell_bins,1)

        suppressed_tuned_cells = np.zeros(self.n_stim)+np.nan
        suppressed_nontuned_cells = np.zeros(self.n_stim)+np.nan

        # find cells that have at least one significantly suppressed bin and are also significantly tuned (kruskal p<.01, mann-whitney p<.0025)
        for stim in range(self.n_stim):
            suppressed_tuned_cells[stim] = len(np.where((kruskal_pval_thresh[stim,:]>0)&(stim_suppressed_thresh[stim,:]>0))[0])

        # find cells that have at least one significantly suppressed bin and are not significantly tuned (kruskal p<.01, mann-whitney p<.0025)
        suppressed_nontuned_cells = np.zeros(self.n_stim)+np.nan
        for stim in range(self.n_stim):
            suppressed_nontuned_cells[stim] = len(np.where((kruskal_pval_thresh[stim,:]==0)&(stim_suppressed_thresh[stim,:]>0))[0])

        # get the tuning curve correlations between each pair of response-eliciting spots (both activating and suppressing) for each cell. 
        tuning_curve_correlations = phase_analysis.get_tuning_curve_correlations(self.trial_type_psth_array, self.bins, self.bin_indices_all, activated_cell_bins, suppressed_cell_bins, kruskal_pval_thresh, shuff = False)    

        # get the shuffled tuning curve correlations between each pair of response-eliciting spots (both activating and suppressing) for each cell. 
        tuning_curve_correlations_shuff = phase_analysis.get_tuning_curve_correlations(self.trial_type_psth_array, self.bins, self.bin_indices_all, activated_cell_bins, suppressed_cell_bins, kruskal_pval_thresh, shuff = True)    

        # the following code looks at several different response features, specifically for the spots that elicit responses in each cell.
        # first, initialize arrays to store each of these features. 
        all_stim_resp_pref_phase = np.zeros((self.n_stim, self.n_bins, self.n_cells)) + np.nan # stores the preferred response (NOT STIMULATION) phase of each stim x bin x cell combo.
        all_stim_rise_times = np.zeros((self.n_stim, self.n_bins, self.n_cells)) + np.nan # stores the time to 50% of max response for each stim x bin x cell combo 
        all_stim_time_to_peak = np.zeros((self.n_stim, self.n_bins, self.n_cells)) + np.nan # stores the time to peak response for each stim x bin x cell combo 
        stimulation_phase_preference = np.zeros((self.n_stim, self.n_cells)) + np.nan # stores the preferred (peak) stimulation phase for each stim x cell combo 

        #loop through cells
        for cell in range(self.n_cells):
            # for this cell, find the stimulated spots on the bulb that produce significantly activating and significantly tuned responses. 
            resp_spots = np.where(np.sum(activated_cell_bins[:,:,cell],1)>0)[0]
            resp_spots = resp_spots[kruskal_pval_thresh[resp_spots,cell]>0]
            if resp_spots.any():
                # get the preferred response phase, peak time, mean psth by bin, and time to 50% of max for each stim x bin combo 
                stim_resp_pref_phase, peak_times, all_psth_phase_mean_sorted, time_to_50 = phase_analysis.get_pref_resp_phase_by_stim(cell, self.bins, self.trial_type_psth_array, self.frame_on_time_s, self.trial_type_indices, self.bin_indices_all, self.instantaneous_phase_corrected)

                # get the mean firing rate over the first 300ms for each stim x bin combo, normalize to the max responding bin within each spot (stim). 
                all_psth_phase_mean_norm = phase_analysis.get_phase_tuning_by_stim(cell, self.bins, self.trial_type_psth_array, self.bin_indices_all, plot = False, norm = True)

                # make an instance of the cell phase response stats class 
                Cell_phase_response_stats = phase_analysis.Phase_response_stats(cell, self.bins, resp_spots, stim_resp_pref_phase, peak_times, all_psth_phase_mean_sorted, time_to_50, all_psth_phase_mean_norm, activated_cell_bins)

                # extract response phase for each activating spots x bin combo
                all_stim_resp_pref_phase[:,:,cell] = Cell_phase_response_stats.return_spot_x_bin_response_pref_resp_phase()

                # extract time to 50% of max for each activating spots x bin combo 
                all_stim_rise_times[:,:,cell] = Cell_phase_response_stats.return_spot_x_bin_time_to_50()
                
                # extract time to peak response for each activating spots x bin combo
                all_stim_time_to_peak[:,:,cell] = Cell_phase_response_stats.return_spot_x_bin_time_to_peak()

                # extract the preferred stimulation phase for activating spots 
                stimulation_phase_preference[:,cell] = Cell_phase_response_stats.return_spot_x_bin_respond_pref_stim_phase()['max_phase_stim_x_cell']

        output_phase_stats = {'Kruskal_pval_thresh':kruskal_pval_thresh, 'Mann_Whitney_responsive_cell_bins': responsive_cell_bins, 'all_stim_resp_pref_phase':all_stim_resp_pref_phase, 'all_stim_time_to_50':all_stim_rise_times, 'all_stim_time_to_peak':all_stim_time_to_peak, 'stimulation_phase_preference':stimulation_phase_preference, 'tuning_curve_correlations':tuning_curve_correlations, 'tuning_curve_correlations_shuff':tuning_curve_correlations_shuff}
        
        # save the output dictionary to the save path for loading later...
        np.save(stats_save_path, [output_phase_stats], allow_pickle = True)
    
    @staticmethod 
    def return_min_trial_num(n_stim, n_bins, trial_type_phase_binned_spike_counts):
        '''
        Output
        ------
        min_trial_num: the minimum number of trials across all bins
        '''
        trial_num = []
        for stim in range(n_stim):
            for bin_num in range(n_bins):
                trial_num.append(len(trial_type_phase_binned_spike_counts[stim][bin_num][0]))
        min_trial_num = np.min(trial_num)
        return min_trial_num
    
    def phase_trial_avg_PCA(self, PCA_save_path, n_components = 3): 
        '''
        Output
        ------
        - saves a dictionary containing fields all_trial_PCA and mean_PCA.
        - all_trial_PCA: the first 3 PCx across the n_bin x n_trial numpy array 
        - mean_PCA: the PCs for each bin averaged across the trials. n_bin x n_PC numpy array.
        '''
        # because there are different numbers of trials for each bin, get the minimum number of trials across all bins
        min_trial_num = Phase_data_analysis.return_min_trial_num(self.n_stim, self.n_bins, self.trial_type_phase_binned_spike_counts)

        # get the phase binned spike counts the first 1:min_trial_num trials 
        trial_type_phase_binned_spike_counts_clip = np.array(phase_analysis.bin_spike_counts_by_phase(self.trial_type_spike_counts, self.bin_indices_all, max_trial = min_trial_num))
        
        # index zero is always the blank. Exclude blank from from PCA analysis. 
        mean_binned_phase_count_clip = np.mean(trial_type_phase_binned_spike_counts_clip[1:,:,:,:],0)
        mean_binned_phase_count_clip = np.moveaxis(mean_binned_phase_count_clip, (0,1), (1,0))

        # get the first three principal components of the population response averaged across spots.
        population_analyses = PGanalysis.population_analyses(mean_binned_phase_count_clip, self.n_bins)
        all_trial_PCA, stim_idxs = population_analyses.PCA_spike_counts(plot = False, n_components = n_components)

        # get the mean PCs across the 30 trials 
        mean_PCA = np.empty((self.n_bins,n_components))
        for index in np.arange(self.n_bins):
            mean_PCA[index,:] =  [np.mean(all_trial_PCA[stim_idxs[index],0]), np.mean(all_trial_PCA[stim_idxs[index],1]), np.mean(all_trial_PCA[stim_idxs[index],2])]

        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(1,1,1, projection = '3d')
        ax.plot(mean_PCA[:,0], mean_PCA[:,1], mean_PCA[:,2],'-o')
        
        output_PCA = {'all_trial_PCA':all_trial_PCA, 'mean_PCA':mean_PCA}
        
        np.save(PCA_save_path, [output_PCA], allow_pickle = True)
        
    def get_spike_counts(self):
        return self.trial_type_spike_counts

    def get_phase_bin_indices(self):
        return self.bin_indices_all
        
    def within_stim_phase_decoding_SVM(self, decoding_save_path, num_training_trials = 20):
        '''
        Output
        ------
        all_stim_conf_mx: This is an n_bins x n_bins x n_stim numpy array representing the phase decoding confusion matrix for each spot
        '''    
        
        all_stim_conf_mx = np.empty((self.n_bins, self.n_bins, self.n_stim))
        
        all_stim_scores = np.empty((self.n_stim, self.n_bins*num_training_trials))
        
        # get the spike counts in the first 20 trials of stimulation for each bin
        trial_type_phase_binned_spike_counts_clip = np.array(phase_analysis.bin_spike_counts_by_phase(self.trial_type_spike_counts, self.bin_indices_all, max_trial = num_training_trials))
        
        for stim in range(self.n_stim):

            # reshape spike counts into 2d matrix and get the training labels
            trial_type_phase_binned_spike_counts_clip_move = np.moveaxis(trial_type_phase_binned_spike_counts_clip[stim,:,:,:], (0,1), (1,0))
            spike_counts_reshaped, labels, stim_idxs = PGanalysis.spike_shaper_2d(trial_type_phase_binned_spike_counts_clip_move)
            X_train = spike_counts_reshaped
            y_train = labels    
            
            # initialize the classifier with a linear kernel
            svm_clf = SVC(kernel = 'linear')

            # cross validate using the leave one out method 
            loo = LeaveOneOut()
            y_train_pred = cross_val_predict(svm_clf, X_train, y_train, cv = loo)
            
            # get the cross validation scores (0 or 1 for each loo cross-validation)
            all_stim_scores[stim,:] = cross_val_score(svm_clf, X_train, y_train, cv = loo, scoring = 'accuracy')

            # get the confusion matrix and normalize by the number of trials
            conf_mx = confusion_matrix(y_train, y_train_pred)
            conf_mx = conf_mx/num_training_trials
            all_stim_conf_mx[:,:,stim] = conf_mx
        
        # create a dictionary with confusion matrices and decoding scores
        classifier_accuracy = {'all_stim_conf_mx':all_stim_conf_mx, 'all_stim_scores':all_stim_scores}
        
        # save the data 
        np.save(decoding_save_path, classifier_accuracy, allow_pickle = True)
        
        return all_stim_conf_mx

        
