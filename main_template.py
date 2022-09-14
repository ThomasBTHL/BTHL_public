"""Import modules"""
import copy
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import functions as f
import pickle
import pandas as pd
from tqdm import tqdm
import time
import xlrd
import c3d
import os
import re
import scipy.signal as sp
import seaborn as sns

colorspallete = sns.color_palette()

"""3D inverse dynamic model is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
Contact E-Mail: a.j.r.leenen@vu.nl
"" Changes made by Thomas van Hogerwou, Master student TU Delft: Thom.hogerwou@gmail.com

Version 1.5 (2020-07-15)"""

"""
Input area
"""
pitcher = 'PP03'
length = 'Pitches' # Pitches or Innings
Polyfit = 1
fs_opti = 120
fs_EMG = 2000
inning_plot = False
side = 'right'
fignum = int(pitcher[2:])

Innings = ['Inning_1','Inning_2','Inning_3','Inning_4','Inning_5','Inning_6','Inning_7','Inning_8','Inning_9','Inning_10','Inning_11']  # Inning where you want to look, for pitches gives all pitches in inning
# Innings = ['Inning_1']
problem_pitches = [5,10,14,22,28,38,41,43,57,60,72,73,88,93,101]  # pitches to remove
mean_forearm_length = 27.386957807443334
mean_upperarm_length = 27.890377309454344
mean_hand_length = 18.83490902787976

"""
Load EMG data
"""
# --- Open Normalized Dictionary --- #
frequency = '2000 Hz'
filename = "data/EMG_Data/05 MVC Normalized/" + frequency + "/" + pitcher + "_Cumulative_EMG"
infile = open(filename, 'rb')
EMG_data = pickle.load(infile)
infile.close()
synced_EMG_data = copy.deepcopy(EMG_data)

path = "data/EMG_Data/06 Muscle Activations/" + frequency + "/" + pitcher + "_Cumulative_EMG"
infile = open(path, 'rb')
Muscle_Activations = pickle.load(infile)
infile.close()

"""
Cumulative output setup.
"""

segments = ['hand', 'forearm', 'upperarm']
outputs = ['max_norm_moment', 'max_abduction_moment']
Cumulative_Fatigue_dictionary = dict.fromkeys(segments)
for segment in segments:
    Cumulative_Fatigue_dictionary[segment] = dict.fromkeys(outputs)
    for output in outputs:
        Cumulative_Fatigue_dictionary[segment][output] = dict()

markers = ['ACC', 'FMP', 'BIC', 'TRI']
EMG_Feature_dictionary = dict.fromkeys(markers)
for key in EMG_Feature_dictionary:
    EMG_Feature_dictionary[key] = dict()

Total_max_M_events = []

for Inning in Innings:


    """
    Override Fatigue dictionary
    """
    Fatigue_dictionary = {}
    segments = ['hand', 'forearm', 'upperarm']
    outputs = ['max_norm_moment', 'max_abduction_moment']
    for segment in segments:
        Fatigue_dictionary[segment] = {}
        for output in outputs:
            Fatigue_dictionary[segment][output] = {}

    """
    Inning setup
    """
    j = 0 # used for time sync of inning data
    k = 0
    Inning_MER_events = []
    Inning_cross_corr_events = []
    Inning_max_normM_events = []
    Inning_max_M_events = []
    Inning_MER_s_events = []
    Inning_seg_M_joint = dict()
    Inning_F_joint = dict()

    """
    Load inning data and remove unwanted pitches
    """
    # Path where the pitching dictionary is saved
    filename = 'data' + '/' + length + '/' + 'Unfiltered' + '/' + pitcher + '/' + Inning

    # Read the dictionary as a new variable
    infile = open(filename, 'rb')
    inning_data_raw = pickle.load(infile)
    infile.close()

    # remove unwanted problem pitches
    pitches_to_remove = []
    for i in range(len(problem_pitches)):
        pitches_to_remove.append("pitch_{0}".format(problem_pitches[i]))

    if len(inning_data_raw) == 14:
        inning_data = dict()
        inning_data['whole inning'] = inning_data_raw
    else:
        inning_data = copy.deepcopy(inning_data_raw)
        for pitch in inning_data_raw:
            if pitch in pitches_to_remove:
                inning_data.pop(pitch)
                if not pitcher == 'PP01':
                    for marker in EMG_data:
                        EMG_data[marker].pop(pitch)
                        synced_EMG_data[marker].pop(pitch)

    fs = fs_opti

    for pitch_number in inning_data:
        # Subdivide dictionary into separate variables
        pitch = inning_data[pitch_number]
        # Calculate the segment parameters
        if side == 'right':
            # --- Pelvis Segment --- #
            pelvis_motion = f.calc_pelvis(pitch['VU_Baseball_R_RASIS'], pitch['VU_Baseball_R_LASIS'], pitch['VU_Baseball_R_RPSIS'], pitch['VU_Baseball_R_LPSIS'], gender='male',sample_freq=fs)
            # --- Thorax Segment --- #
            thorax_motion = f.calc_thorax(pitch['VU_Baseball_R_IJ'], pitch['VU_Baseball_R_PX'], pitch['VU_Baseball_R_C7'], pitch['VU_Baseball_R_T8'], gender='male',sample_freq=fs)
            # --- Upperarm Segment --- #
            upperarm_motion = f.calc_upperarm(pitch['VU_Baseball_R_RLHE'], pitch['VU_Baseball_R_RMHE'], pitch['VU_Baseball_R_RAC'], side, gender='male',sample_freq=fs, mean_seg_length = mean_upperarm_length)
            # --- Forearm Segment --- #
            forearm_motion = f.calc_forearm(pitch['VU_Baseball_R_RLHE'], pitch['VU_Baseball_R_RMHE'], pitch['VU_Baseball_R_RUS'], pitch['VU_Baseball_R_RRS'], side, gender='male',sample_freq=fs, mean_seg_length= mean_forearm_length)
            # --- Hand Segment --- #
            hand_motion = f.calc_hand(pitch['VU_Baseball_R_RUS'], pitch['VU_Baseball_R_RRS'], pitch['VU_Baseball_R_RHIP3'], side, gender='male',sample_freq=fs, mean_seg_length = mean_hand_length)
            # Combine all the referenced segment dictionaries into dictionary in order to loop through the keys for net force and moment calculations
            model = f.segments2combine(pelvis_motion, thorax_motion, upperarm_motion, forearm_motion, hand_motion)

        if side == 'left':
            # Calculate the segment parameters
            # --- Pelvis Segment --- #
            pelvis_motion = f.calc_pelvis(pitch['VU_Baseball_L_RASIS'], pitch['VU_Baseball_L_LASIS'],pitch['VU_Baseball_L_RPSIS'], pitch['VU_Baseball_L_LPSIS'], gender='male',sample_freq=fs)
            # --- Thorax Segment --- #
            thorax_motion = f.calc_thorax(pitch['VU_Baseball_L_IJ'], pitch['VU_Baseball_L_PX'],pitch['VU_Baseball_L_C7'], pitch['VU_Baseball_L_T8'], gender='male',sample_freq=fs)
            # --- Upperarm Segment --- #
            upperarm_motion = f.calc_upperarm(pitch['VU_Baseball_L_LLHE'], pitch['VU_Baseball_L_LMHE'],pitch['VU_Baseball_L_LAC'], side, gender='male', sample_freq=fs,mean_seg_length=mean_upperarm_length)
            # --- Forearm Segment --- #
            forearm_motion = f.calc_forearm(pitch['VU_Baseball_L_LLHE'], pitch['VU_Baseball_L_LMHE'],pitch['VU_Baseball_L_LUS'], pitch['VU_Baseball_L_LRS'], side,gender='male', sample_freq=fs, mean_seg_length=mean_forearm_length)
            # --- Hand Segment --- #
            hand_motion = f.calc_hand(pitch['VU_Baseball_L_LUS'], pitch['VU_Baseball_L_LRS'],pitch['VU_Baseball_L_LHIP3'], side, gender='male', sample_freq=fs,mean_seg_length=mean_hand_length)
            # Combine all the referenced segment dictionaries into dictionary in order to loop through the keys for net force and moment calculations
            model = f.segments2combine(pelvis_motion, thorax_motion, upperarm_motion, forearm_motion, hand_motion)

        # Rearrange model to have the correct order of segments for 'top-down' method
        model = f.rearrange_model(model, 'top-down')

        # Angle study for force - length / force - velocity
        elbow_angles = f.euler_angles('ZXY', model['forearm']['gRseg'], model['upperarm']['gRseg'])
        flexion_extension_angle = elbow_angles[0, :]

        # Filter angles
        Filtered_elbow_angles = copy.deepcopy(elbow_angles)
        for angle in range(2):
            Filtered_elbow_angles[angle, :] = f.butter_lowpass_filter(elbow_angles[angle,:], 12, 120, 2)

        flexion_extension_acc = np.gradient(Filtered_elbow_angles[0,:])*fs_opti

        # Calculate the net forces according the newton-euler method
        F_joint = f.calc_net_reaction_force(model)

        # Calculate the net moments according the newton-euler method
        M_joint = f.calc_net_reaction_moment(model, F_joint)

        if (np.isnan(np.nanmean(M_joint['hand']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['forearm']['M_proximal'])) == False) and (
                np.isnan(np.nanmean(M_joint['upperarm']['M_proximal'])) == False):

            # Project the calculated net moments according the newton-euler method to local coordination system to be anatomically meaningful
            joints = {'hand': 'wrist', 'forearm': 'elbow', 'upperarm': 'shoulder', 'thorax': 'spine', 'pelvis': 'hip'}  # Joints used to calculate the net moments according the newton-euler method

            # Initialise parameters
            seg_M_joint = dict()
            for segment in segments:
                seg_M_joint[segment] = f.moments2segment(model[segment]['gRseg'], M_joint[segment]['M_proximal'])
                if side == 'left':
                    seg_M_joint[segment][0:2, :] = -seg_M_joint[segment][0:2, :]

            seg_M_joint['time_line'] = np.linspace(0,1.4,len(seg_M_joint['hand'][0,:]))

            # Determine pitch PX max acc index
            if side == 'right':
                PX_array = np.gradient(np.array(pitch['VU_Baseball_R_PX']['Y']))
            else:
                PX_array = np.gradient(np.array(pitch['VU_Baseball_L_PX']['Y']))

            if pitcher == 'PP15':
                PX_array = -np.gradient(np.array(pitch['VU_Baseball_R_PX']['Y']))
            elif pitcher == 'PP04':
                PX_array = -np.gradient(np.gradient(np.array(pitch['VU_Baseball_R_PX']['Y'])))
            elif pitcher == 'PP13':
                PX_array = -np.gradient(np.gradient(np.array(pitch['VU_Baseball_L_PX']['Y'])))


            PX_acc_index = np.nanargmax(PX_array)  # Just Y direction
            if pitcher == 'PP04':
                PX_peaks = sp.find_peaks(PX_array, height=0.002, distance=100)
                PX_acc_index = PX_peaks[0][0]

            # --- Select window based on found indices for each segment --- #
            seg_window = np.linspace(PX_acc_index - 2, PX_acc_index + 2, ((PX_acc_index + 2) - (PX_acc_index - 2) + 1),
                                     endpoint=True).astype(int)
            # --- Calculate the 2nd order polynomial function coefficients for each segment --- #
            seg_fit = np.polyfit(seg_window, PX_array[seg_window], 2)
            # --- Analytical calculation of the exact point in time of the occurrence of the peak angular velocity for each segment --- #
            if ~np.isnan(seg_fit[0]):
                PX_analytical_index = -(seg_fit[1] / (2 * seg_fit[0]))
            else:
                PX_analytical_index = PX_acc_index

            EMG_acc_index = np.nanargmax((-EMG_data['ACC_filtered'][pitch_number]))

            PX_acc_s = PX_analytical_index / fs_opti
            EMG_acc_s = EMG_acc_index / fs_EMG
            fs_delay = fs_EMG

            EMG_acc_delay = PX_acc_s - EMG_acc_s #postive delay means EMG signal is happening earlier
            lag = int(fs_delay * EMG_acc_delay)

            # Add delay to sync all EMG markers based on PX
            for marker in EMG_data:
                if marker != 'real_time':
                    if 'pitch' in pitch_number:
                        temp = np.append(np.array(synced_EMG_data[marker][pitch_number]),np.linspace(0,1.4,2800),axis = 0)
                        synced_EMG_data[marker][pitch_number] = np.reshape(temp, [len(EMG_data[marker][pitch_number]),2],order = 'F')
                        synced_EMG_data[marker][pitch_number][:, 1] = synced_EMG_data[marker][pitch_number][:, 1] + EMG_acc_delay

            """
            Time syncing Moment data for smooth overlayed plots and variability plots
            """
            # Determine Ball release
            hand_vNorm = [np.linalg.norm(model['hand']['vSeg'][:,i]) for i in range(len(model['hand']['vSeg'][0,:]))]
            BR_index = np.nanargmax(hand_vNorm) + 1

            print('Ball release time is')
            print((BR_index)/120)

            # Determine MER index
            [pitch_MER, pitch_index_MER] = f.MER_event(model)
            pitch_MER_s = pitch_index_MER / 120
            Inning_MER_s_events.append(pitch_MER_s)

            print('MER time is')
            print(pitch_MER_s)



            # Determine max abduction moment [0] correlation index
            max_M_index = np.nanargmax(seg_M_joint['forearm'][0,:])
            if pitcher == 'PP12':
                max_M_peaks = sp.find_peaks(seg_M_joint['forearm'][0,:], height=25, distance=5)
                max_M_index = max_M_peaks[0][0]


            # --- Select window based on found indices for each segment --- #
            seg_window = np.linspace(max_M_index - 2, max_M_index + 2, ((max_M_index + 2) - (max_M_index - 2) + 1),endpoint=True).astype(int)
            # --- Calculate the 2nd order polynomial function coefficients for each segment --- #
            seg_fit = np.polyfit(seg_window, seg_M_joint['forearm'][0,seg_window], 2)

            if ~np.isnan(seg_fit[0]):
                # --- Analytical calculation of the exact point in time of the occurrence of the peak angular velocity for each segment --- #
                analytical_max_M = -(seg_fit[1] / (2 * seg_fit[0]))
            else:
                analytical_max_M = max_M_index

            max_M_s = analytical_max_M / 120

            Inning_max_M_events.append(max_M_s)
            Total_max_M_events.append(max_M_s)

            # Visual check EMG features
            MER_plot_delay = Inning_MER_s_events[k] - Inning_MER_s_events[0]
            if inning_plot == True:
                Abd_plot_delay = Inning_max_M_events[k] - Inning_max_M_events[0]
            else:
                Abd_plot_delay = Inning_max_M_events[k] - Total_max_M_events[0]
            k += 1

            # Max moment data for fatigue study
            Fatigue_dictionary = f.max_moment_data(Fatigue_dictionary, seg_M_joint, segments, pitch_number, Polyfit)

            # seperation time for PP study
            separation_time, peak_ang_velocity, pelvis_index,thorax_index, pelvis_peak, thorax_peak = f.separation_time_pp(model, fs_opti, analytical=0)

            plt.figure(20 + fignum)
            plt.title('Thorax and Pelvis Angular Velocities')
            plt.plot(model['thorax']['avSegNorm'], label = 'Thorax')
            plt.plot(model['pelvis']['avSegNorm'], label = 'Pelvis')
            plt.legend()


            """
            Plotting
            """
            l = 1
            plt.figure(fignum)
            for marker in markers:
                plt.subplot(3,2,l)
                plt.title(pitcher+' '+markers[l-1])
                EMG_timeline = synced_EMG_data[marker][pitch_number][:,1] - Abd_plot_delay
                if marker == 'ACC':
                    plt.plot(EMG_timeline, (-synced_EMG_data['ACC_filtered'][pitch_number][:, 0]), label=pitch_number)
                else:
                    for activation in Muscle_Activations[marker][pitch_number]:
                        plt.vlines(EMG_timeline[activation],ymin = 0, ymax = np.nanmax(synced_EMG_data[marker][pitch_number][:,0]), linestyles= '--', colors= colorspallete[j])
                    plt.plot(EMG_timeline,synced_EMG_data[marker][pitch_number][:,0],label = pitch_number, color = colorspallete[j])
                plt.xlim(0.3,1)
                plt.vlines(pitch_MER_s,ymin = 0, ymax = np.nanmax(synced_EMG_data[marker][pitch_number][:,0]), linestyles= '--', colors= ['k'])
                # plt.legend()
                l += 1

            if l == 5:
                plt.subplot(3, 2, l)
                plt.title('gradient of PX_array')
                plt.plot((seg_M_joint['time_line'] - Abd_plot_delay), PX_array,
                         label=pitch_number)
                plt.xlim(0.3, 1)
                plt.vlines(pitch_MER_s, ymin=0, ymax=np.nanmax(PX_array),
                           linestyles='--',
                           colors=['k'])
                # plt.legend()
                l += 1

            if l == 6:
                plt.subplot(3, 2, l)
                plt.title('Abduction moment')
                plt.plot((seg_M_joint['time_line'] - Abd_plot_delay),seg_M_joint['forearm'][0,:],label = pitch_number)
                plt.xlim(0.3, 1)
                plt.vlines(pitch_MER_s, ymin=0, ymax=np.nanmax(seg_M_joint['forearm'][0,:]), linestyles='--',
                           colors=['k'])
                # plt.legend()


                plt.figure(60 + fignum)
                if inning_plot == True:
                    plt.subplot(4, 1, 1)
                    plt.xlim(0.3, 1)
                    plt.title(pitcher + ' Elbow flexion angles')
                    # plt.plot((seg_M_joint['time_line'] - Abd_plot_delay),elbow_angles[0, :])
                    plt.plot((seg_M_joint['time_line'] - Abd_plot_delay),Filtered_elbow_angles[0, :])

                    plt.subplot(4, 1, 2)
                    plt.xlim(0.3, 1)
                    plt.title(pitcher + ' Elbow abduction angles')
                    # plt.plot((seg_M_joint['time_line'] - Abd_plot_delay),elbow_angles[1, :])
                    plt.plot((seg_M_joint['time_line'] - Abd_plot_delay), Filtered_elbow_angles[1, :])

                    plt.subplot(4, 1, 3)
                    plt.xlim(0.3, 1)
                    plt.title(pitcher + 'SER')

                    # Determine rotation matrix of the upperarm
                    R_upperarm = model['upperarm']['gRseg']
                    # Determine rotation matrix of the thorax
                    R_thorax = model['thorax']['gRseg']
                    # Euler angles humerus relative to the thorax = shoulder external rotation
                    GH = f.euler_angles('YXY', R_upperarm, R_thorax)  # zyz
                    SER = GH[2, :]
                    for i in range(len(SER)):
                        if SER[i] < -100:
                            SER[i] = 360 + SER[i]

                    SER_filtered = f.butter_lowpass_filter(SER, 12, 120, 2)

                    plt.plot((seg_M_joint['time_line'] - Abd_plot_delay),SER_filtered)

                    plt.subplot(4, 1, 4)
                    plt.xlim(0.3, 1)
                    plt.title('Abduction moment')
                    plt.plot((seg_M_joint['time_line'] - Abd_plot_delay), seg_M_joint['forearm'][0, :],
                             label=pitch_number)
        j += 1
        plt.show()