"""Import modules"""
import copy
import pickle
import functions as f
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import scipy.signal as sp
import pandas as pd

"""
Add_EMG_data is developed and written by Thomas van Hogerwou, master student TU-Delft

It adds EMG data to the Outputs result dictionary based on peak accelerations
Contact E-Mail: thom.hogerwou@gmail.com
Version 1.5 (2021-05-18)
"""

"""
Input area
"""
# pitchers = ['PP02','PP03','PP04','PP05','PP06','PP07','PP08','PP12','PP14','PP15'] #PP01 - PP15
pitchers = ['PP12']
# pitchers = ['PP09','PP10','PP11','PP13']

fs_EMG = 2000
fs_opti = 120
lead = .6
lag = .8
fs_scaling = fs_opti / fs_EMG
EMG_markers = ['ACC','BIC','DM','FMP','LD','PC','PS','TRI','real_time','time_s']

for pitcher in pitchers:
    """
    individual pitcher information
    """
    if pitcher == 'PP01':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_5','Inning_6','Inning_7','Inning_8'] # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP02':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6'] # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP03':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP04':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP05':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP06':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP07':
        Innings = ['Inning_1', 'Inning_2', 'Inning_3', 'Inning_4', 'Inning_5',
                   'Inning_6']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP08':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP09':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP10':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP11':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP12':
        Innings = ['Inning_1', 'Inning_2', 'Inning_3', 'Inning_4', 'Inning_5',
                   'Inning_6']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP13':
        Innings = ['Inning_1', 'Inning_2', 'Inning_5',
                   'Inning_6', 'Inning_7', 'Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP14':
        Innings = ['Inning_1', 'Inning_2','Inning_5','Inning_6',
                   'Inning_7','Inning_8']  # Inning where you want to look, for pitches gives all pitches in inning

    if pitcher == 'PP15':
        Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11']  # Inning where you want to look, for pitches gives all pitches in inning

    Cumulative_ds_storage = dict.fromkeys(EMG_markers)
    for EMG_marker in EMG_markers:
        Cumulative_ds_storage[EMG_marker] = dict()

    Cumulative_2000Hz_storage = copy.deepcopy(Cumulative_ds_storage)

    for inning in Innings:
        """
        Downsampled storage
        """
        Downsampled_storage = dict.fromkeys(EMG_markers)
        for EMG_marker in EMG_markers:
            Downsampled_storage[EMG_marker] = dict()

        """
        2000 Hz storage
        """
        Uncompressed_storage = copy.deepcopy(Downsampled_storage)

        # --- Reset EMG_downsampled --- #
        EMG_downsampled = dict.fromkeys(EMG_markers)


        # --- Define path where EMG data is stored --- #
        path = "data/EMG_Data/01 Raw Data/"+pitcher+"/EMG_Cut/"
        j = 10*int(inning[7:])
        filename = pitcher + "_EMGCut_"+str(j)

        # --- Load data from pickle --- #
        filenameIn = path + filename
        infile = open(filenameIn, 'rb')
        EMG_data = pickle.load(infile)
        infile.close()

        """
        Downsample EMG data to new dictionary
        """
        for EMG_marker in EMG_markers:
            EMG_downsampled[EMG_marker] = sp.resample(EMG_data[EMG_marker], int(len(EMG_data[EMG_marker]) * fs_scaling))
            ds_peaks = sp.find_peaks((-EMG_downsampled['ACC']),height=(np.nanmax((-EMG_downsampled['ACC'])) * .45), distance=800)
            uncompressed_peaks = sp.find_peaks((-EMG_data['ACC']),height=(np.nanmax((-EMG_data['ACC'])) * .45), distance=(800 / fs_scaling))
            ds_peaks = list(ds_peaks[0])
            uncompressed_peaks = list(uncompressed_peaks[0])

        """
        remove bad peaks
        """
        if len(ds_peaks) > 10:
            print(len(ds_peaks))
            print(ds_peaks)
            print(uncompressed_peaks)
            remove_peaks = []
            distance = []
            plt.figure()
            plt.plot((-EMG_downsampled['ACC']))
            # Use ginput to manually select cut points
            tuples = plt.ginput(15, -1, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
            for i in range(len(tuples)):
                remove_peaks.append(int(tuples[i][0]))
            for remove in remove_peaks:
                distance = [abs(remove - peak) for peak in ds_peaks]
                ds_peaks.remove(ds_peaks[np.argmin(distance)])
                if len(uncompressed_peaks) > 10:
                    uncompressed_peaks.remove(uncompressed_peaks[np.argmin(distance)])

        if len(ds_peaks) < 10:
            print(len(ds_peaks))
            print(ds_peaks)
            plt.figure()
            plt.plot((-EMG_downsampled['ACC']))
            # Use ginput to manually select extra points
            tuples = plt.ginput(15, -1, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
            for i in range(len(tuples)):
                ds_peaks.append(int(tuples[i][0]))
                ds_peaks.sort()
                if len(uncompressed_peaks) < 10:
                    uncompressed_peaks.append(int(tuples[i][0])/fs_scaling)
                    uncompressed_peaks.sort()

        print(ds_peaks)
        print(uncompressed_peaks)

        for EMG_marker in EMG_markers:
            inning_pitch_numbers = np.linspace(j-9, j, 10)
            inning_pitches = ['pitch_' + str(int(inning_pitch_number)) for inning_pitch_number in inning_pitch_numbers]
            for pitch in inning_pitches:
                i = int(pitch[-1]) -1
                if i == -1:
                    i = 9
                Downsampled_storage[EMG_marker][pitch] = EMG_downsampled[EMG_marker][int(ds_peaks[i] - lead * fs_opti):int(ds_peaks[i] + lag * fs_opti)]
                Uncompressed_storage[EMG_marker][pitch] = EMG_data[EMG_marker][int(uncompressed_peaks[i] - lead * fs_EMG):int(uncompressed_peaks[i] + lag * fs_EMG)]
                Cumulative_ds_storage[EMG_marker][pitch] = Downsampled_storage[EMG_marker][pitch]
                Cumulative_2000Hz_storage[EMG_marker][pitch] = Uncompressed_storage[EMG_marker][pitch]

        # --- Define path where 120 Hz inning data is saved --- #
        path = "data/EMG_Data/02 Preprocessed data/120 Hz/"+pitcher+"/"+inning+"/"
        filename = "Cut_EMG"

        # --- Save data --- #
        filenameIn = path + filename
        outfile = open(filenameIn, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(Downsampled_storage, outfile)
        outfile.close()
        print('120 Hz EMG data has been saved.')

        # --- Define path where 2000 Hz inning data is saved --- #
        path = "data/EMG_Data/02 Preprocessed data/2000 Hz/"+pitcher+"/"+inning+"/"
        filename = "Cut_EMG"

        # --- Save data --- #
        filenameIn = path + filename
        outfile = open(filenameIn, 'wb')
        # Write the dictionary into the binary file
        pickle.dump(Uncompressed_storage, outfile)
        outfile.close()
        print('2000 Hz EMG data has been saved.')

    # --- Add MVC data to Cumulative storage --- #
    Cumulative_ds_storage = f.Add_MVC_EMG_data(Cumulative_ds_storage, pitcher, fs_desired=fs_opti)

    # --- Define path where 120 Hz cumulative data is saved --- #
    path = "data/EMG_Data/02 Preprocessed data/120 Hz/" + pitcher + "/"
    filename = "Cumulative_EMG"

    # --- Save data --- #
    filenameIn = path + filename
    outfile = open(filenameIn, 'wb')
    # Write the dictionary into the binary file
    pickle.dump(Cumulative_ds_storage, outfile)
    outfile.close()
    print('120 Hz EMG data has been saved.')

    # --- Add MVC data to Cumulative storage --- #
    Cumulative_2000Hz_storage = f.Add_MVC_EMG_data(Cumulative_2000Hz_storage, pitcher, fs_desired=fs_EMG)

    # --- Define path where 2000 Hz inning data is saved --- #
    path = "data/EMG_Data/02 Preprocessed data/2000 Hz/" + pitcher + "/"
    filename = "Cumulative_EMG"

    # --- Save data --- #
    filenameIn = path + filename
    outfile = open(filenameIn, 'wb')
    # Write the dictionary into the binary file
    pickle.dump(Cumulative_2000Hz_storage, outfile)
    outfile.close()
    print('2000 Hz EMG data has been saved.')