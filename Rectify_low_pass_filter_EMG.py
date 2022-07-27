"""Import modules"""
import copy
import pickle
import functions as f
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import scipy.signal as sp
import scipy.stats as st
import pandas as pd
import biosignalsnotebooks as bsnb

"""
Add_EMG_data is developed and written by Thomas van Hogerwou, master student TU-Delft

It adds EMG data to the Outputs result dictionary based on peak accelerations
Contact E-Mail: thom.hogerwou@gmail.com
Version 1.5 (2021-05-18)
"""

"""
Input area
"""
# pitchers = ['PP02','PP03','PP04','PP05','PP06','PP07','PP08','PP12','PP14','PP15'] #PP01 - PP15\
pitchers = ['PP03']
fs_EMG = 2000
fs_opti = 120
fs_scaling = fs_opti / fs_EMG
EMG_markers = ['BIC','DM','FMP','LD','PC','PS','TRI']
Wn = 20 #Hz of the lowpass filter
N = 2 #Order of lowpass filter

frequencies = ['2000 Hz']
# pitchers = ['PP03']

# Regression function.
def normReg(thresholdLevel):
    threshold_0_perc_level = (- avg_pre_pro_signal) / float(std_pre_pro_signal)
    threshold_100_perc_level = (np.max(smooth_signal[EMG_marker][pitch]) - avg_pre_pro_signal) / float(std_pre_pro_signal)
    m, b = st.linregress([0, 100], [threshold_0_perc_level, threshold_100_perc_level])[:2]
    return m * thresholdLevel + b

for pitcher in pitchers:
    for frequency in frequencies:
        # --- Define path where cumulative data is saved --- #
        path = "data/EMG_Data/02 Preprocessed data/"+frequency+"/" + pitcher + "/"
        filename = "Cumulative_EMG"

        # --- Load data from pickle --- #
        filenameIn = path + filename
        infile = open(filenameIn, 'rb')
        EMG_data = pickle.load(infile)
        infile.close()

        # --- Prepare rectified dictionary --- #
        Rectified_storage = copy.deepcopy(EMG_data)
        for pitch in Rectified_storage['real_time']:
            Rectified_storage['real_time'][pitch] = np.array(Rectified_storage['real_time'][pitch])
            Rectified_storage['time_s'][pitch] = np.array(Rectified_storage['time_s'][pitch])

        # --- Rectify EMG_data --- #
        for EMG_marker in EMG_markers:
            for pitch in EMG_data[EMG_marker]:
                Rectified_storage[EMG_marker][pitch] = np.array(np.abs(EMG_data[EMG_marker][pitch]))

        # --- Save Rectified Dictionary --- #
        path = "data/EMG_Data/03 Rectified data/"+frequency+"/" + pitcher + "_Cumulative_EMG"
        outfile = open(path, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(Rectified_storage, outfile)
        outfile.close()
        print(frequency + ' Rectified EMG data has been saved.')

        # --- Prepare filtered dictionary --- #
        Filtered_storage = copy.deepcopy(Rectified_storage)
        Filtered_storage['ACC_filtered'] = dict.fromkeys(Filtered_storage['ACC'])

        if frequency == '120 Hz':
            fs = fs_opti
        else:
            fs = fs_EMG
        b, a = sp.butter(N, (Wn/(fs/2)))
        # --- Filter EMG_data --- #
        for EMG_marker in EMG_markers:
            for pitch in EMG_data[EMG_marker]:
                Filtered_storage[EMG_marker][pitch] = sp.filtfilt(b = b, a = a, x = np.array(Rectified_storage[EMG_marker][pitch]))
                Filtered_storage['ACC_filtered'][pitch]  = sp.filtfilt(b=b, a=a, x=np.array(Rectified_storage['ACC'][pitch]))

        # --- Save Filtered Dictionary --- #
        path = "data/EMG_Data/04 Low Pass Filtered/"+frequency+"/" + pitcher + "_Cumulative_EMG"
        outfile = open(path, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(Filtered_storage, outfile)
        outfile.close()
        print(frequency + ' Filtered EMG data has been saved.')

        """
        MVC 
        """
        # --- Create Normalized Dictionary --- #
        Normalized_dictionary = copy.deepcopy(Filtered_storage)
        # --- Fill Normalized Dictionary --- #
        for EMG_marker in EMG_markers:
            window_size = int(.2 * fs)  # .2s
            numbers_series = pd.Series(Filtered_storage[EMG_marker]['MVC'])
            windows = numbers_series.rolling(window_size)
            moving_averages = windows.mean()

            moving_averages_list = moving_averages.tolist()
            Filtered_storage[EMG_marker]['MVC_value'] = np.nanmax(moving_averages)

            for pitch in Filtered_storage[EMG_marker]:
                if 'pitch' in pitch:
                    Normalized_dictionary[EMG_marker][pitch] = Filtered_storage[EMG_marker][pitch]/ Filtered_storage[EMG_marker]['MVC_value']

        # --- Save Normalized Dictionary --- #
        path = "data/EMG_Data/05 MVC Normalized/"+frequency+"/" + pitcher + "_Cumulative_EMG"
        outfile = open(path, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(Normalized_dictionary, outfile)
        outfile.close()
        print(frequency + ' Normalized EMG data has been saved.')

        '''
        TEKO
        '''
        pre_pro_signal = copy.deepcopy(EMG_data)
        tkeo = copy.deepcopy(pre_pro_signal)
        rect_signal = copy.deepcopy(tkeo)
        smooth_signal = copy.deepcopy(rect_signal)
        binary_signal = copy.deepcopy(smooth_signal)
        Muscle_activations = copy.deepcopy(binary_signal)
        Muscle_deactivations = copy.deepcopy(Muscle_activations)

        # --- Preprocess EMG_data --- #
        for EMG_marker in EMG_markers:
            for pitch in EMG_data[EMG_marker]:
                if 'pitch' in pitch:
                    pre_pro_signal[EMG_marker][pitch] = pre_pro_signal[EMG_marker][pitch] - np.average(pre_pro_signal[EMG_marker][pitch])
                    # [Signal Filtering]
                    low_cutoff = 10  # Hz
                    high_cutoff = 300  # Hz

                    # Application of the signal to the filter.
                    pre_pro_signal[EMG_marker][pitch] = bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal[EMG_marker][pitch], low_cutoff, high_cutoff, fs)

                    # [Application of TKEO Operator]

                    tkeo[EMG_marker][pitch] = []
                    for i in range(0, len(pre_pro_signal[EMG_marker][pitch])):
                        if i == 0 or i == len(pre_pro_signal[EMG_marker][pitch]) - 1:
                            tkeo[EMG_marker][pitch].append(pre_pro_signal[EMG_marker][pitch][i])
                        else:
                            tkeo[EMG_marker][pitch].append(np.power(pre_pro_signal[EMG_marker][pitch][i], 2) - (pre_pro_signal[EMG_marker][pitch][i + 1] * pre_pro_signal[EMG_marker][pitch][i - 1]))

                    # Smoothing level [Size of sliding window used during the moving average process (a function of sampling frequency)]
                    smoothing_level_perc = 5  # Percentage.
                    smoothing_level = int((smoothing_level_perc / 100) * fs)

                    # [Signal Rectification]

                    rect_signal[EMG_marker][pitch] = np.absolute(tkeo[EMG_marker][pitch])

                    # [First Moving Average Filter]
                    rect_signal[EMG_marker][pitch] = bsnb.aux_functions._moving_average(rect_signal[EMG_marker][pitch], int(.05 * fs))

                    # [Second Smoothing Phase]

                    smooth_signal[EMG_marker][pitch] = []
                    for i in range(0, len(rect_signal[EMG_marker][pitch])):
                        if smoothing_level < i < len(rect_signal[EMG_marker][pitch]) - smoothing_level:
                            smooth_signal[EMG_marker][pitch].append(np.mean(rect_signal[EMG_marker][pitch][i - smoothing_level:i + smoothing_level]))
                        else:
                            smooth_signal[EMG_marker][pitch].append(0)

                    # [Threshold]
                    avg_pre_pro_signal = np.average(pre_pro_signal[EMG_marker][pitch])
                    std_pre_pro_signal = np.std(pre_pro_signal[EMG_marker][pitch])

                    # Chosen Threshold Level (Example with two extreme values)
                    threshold_level = 15  # % Relative to the average value of the smoothed signal
                    threshold_level_norm_10 = normReg(threshold_level)

                    threshold_10 = avg_pre_pro_signal + threshold_level_norm_10 * std_pre_pro_signal

                    # Generation of a square wave reflecting the activation and inactivation periods.

                    binary_signal[EMG_marker][pitch] = []
                    for i in range(0, len(Rectified_storage['time_s'][pitch])):
                        if smooth_signal[EMG_marker][pitch][i] >= threshold_10:
                            binary_signal[EMG_marker][pitch].append(1)
                        else:
                            binary_signal[EMG_marker][pitch].append(0)

                    diff_signal = np.diff(binary_signal[EMG_marker][pitch])
                    act_begin = np.where(diff_signal == 1)[0]
                    act_end = np.where(diff_signal == -1)[0]

                    Muscle_activations[EMG_marker][pitch] = act_begin
                    Muscle_deactivations[EMG_marker][pitch] = act_end

        pitch = 'pitch_12'
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title(pitch + ' : First moving average = .05 s' + ', smoothing level = ' + str(
            smoothing_level_perc) + '% , threshold level = ' + str(threshold_level) + '%')
        plt.plot(Rectified_storage['time_s'][pitch][1:2798], EMG_data['BIC'][pitch][1:2798], label = 'Raw')
        plt.plot(Rectified_storage['time_s'][pitch][1:2798], pre_pro_signal['BIC'][pitch][1:2798], label = 'Prepro')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(Rectified_storage['time_s'][pitch][1:2798], tkeo['BIC'][pitch][1:2798], label='tkeo')
        plt.plot(Rectified_storage['time_s'][pitch][1:2798], rect_signal['BIC'][pitch][1:2798],
                 label='rect')
        plt.plot(Rectified_storage['time_s'][pitch][1:2798], smooth_signal['BIC'][pitch][1:2798],
                 label='smooth')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(Rectified_storage['time_s'][pitch][1:2798], binary_signal['BIC'][pitch][1:2798], label = 'binary')
        plt.plot(Rectified_storage['time_s'][pitch][1:2798], Normalized_dictionary['BIC'][pitch][1:2798], label = 'Normalized (not TKEO)')
        plt.legend()
        plt.show()

        # --- Save Muscle_activations Dictionary --- #
        path = "data/EMG_Data/06 Muscle Activations/" + frequency + "/" + pitcher + "_Cumulative_EMG"
        outfile = open(path, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(Muscle_activations, outfile)
        outfile.close()
        print(frequency + ' Muscle_activations have been saved.')

        # --- Save Muscle_deactivations Dictionary --- #
        path = "data/EMG_Data/07 Muscle Deactivations/" + frequency + "/" + pitcher + "_Cumulative_EMG"
        outfile = open(path, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(Muscle_deactivations, outfile)
        outfile.close()
        print(frequency + ' Muscle_deactivations have been saved.')







