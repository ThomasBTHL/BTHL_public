from functions import *
import copy
import matplotlib
import pickle
"""
This program is part of the preprocessing motion capture data.
1) Determine ball_pickups and save into markers
2) Filter the markers and save as markersFilter
3) Save all 4 configurations of filtered / cut

In this function you can use three plot functions, these can be found in functions.py:
1) visual_check_markers('RUS', 'RRS', 'RMHE', 'RLHE', markers)
2) visual_check_smoothing_effect(markerName, coordinateName, markers, markersNew)

Preprocessing_data.py is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam 
& Bart van Trigt, PhD-candidate TU Delft  & Thomas van Hogerwou, master student TU-Delft
Contact E-Mail: a.j.r.leenen@vu.nl; b.vantrigt@tudelft.nl; t.c.vanhogerwou@student.tudelft.nl

Logbook:
Version 1.0 (2022-03-18)
"""

subject_name = "PP12"

# Filter parameters
cutoff = 25
fs = 120
order = 4

# Define the path to load the data
path = os.path.abspath(os.path.join("data/Optitrack_Data/Preprocessed_data/" + subject_name + "_c3d/"))
## Load the c3d files to xlsx
markers_c3d = load_c3d_innings(path)

# Initial inning number
inning_number = 1

for Innings in markers_c3d:
    inning_name = Innings
    raw_markers = markers_c3d[Innings]

    """
    0)  Orient data set:
        col 1 'X' should be to the right 
        col 2 'Y' should point forwards
        col 3 'Z' should point up
    """

    Inning_markers = orient_markers(raw_markers)

    """
    1) cut and trim signal
    """

    ball_pickups = ball_pickup_indexs('VU_Baseball_R_RUS', 'VU_Baseball_R_RRS', 'VU_Baseball_R_RMHE', 'VU_Baseball_R_RLHE', Inning_markers)
    markers_cut_unfiltered = cut_markers(Inning_markers,ball_pickups,inning_number)  # Outputs dictionary of dictionaries contatining individual pitches, length = ballpickups + 1
    markers_trimmed_unfiltered = trim_markers(markers_cut_unfiltered, fs, lead = .6, lag = .8)

    # """
    # 2) filter the signal
    # This part smooth the signal with the Butterworth filter
    # """
    #
    # Filtered_pitches = butter_lowpass_filter_inning(markers_trimmed_unfiltered, cutoff, fs, order)

#    visual_check_smoothing_effect('VU_Baseball_R_C7', 'X', markers_trimmed_unfiltered['pitch_1'], Filtered_pitches['pitch_1'])

    """
    3.1) Save the unfiltered inning data
    """
    # Path where the pickle will be saved. Last part will be the name of the file
    filename = 'data/Innings/Unfiltered/'+ subject_name + '/' + inning_name
    # Initialize the pickle file
    outfile = open(filename, 'wb')
    # Write the dictionary into the binary file
    pickle.dump(Inning_markers, outfile)
    outfile.close()
    print('Unfiltered dictionary has been saved.')

    # """
    # 3.2) Save the filtered pitch data
    # """
    #
    # # Path where the pickle will be saved. Last part will be the name of the file
    # filename = 'data/Pitches/Filtered/'+ subject_name + '/' + inning_name
    # # Initialize the pickle file
    # outfile = open(filename, 'wb')
    # # Write the dictionary into the binary file
    # pickle.dump(Filtered_pitches, outfile)
    # outfile.close()
    # print('Filtered dictionary has been saved.')

    """
    3.3) Save unfiltered pitch data
    """

    # Path where the pickle will be saved. Last part will be the name of the file
    filename = 'data/Pitches/Unfiltered/' + subject_name + '/' + inning_name
    # Initialize the pickle jar file
    outfile = open(filename, 'wb')
    # Write the dictionary into the binary file
    pickle.dump(markers_trimmed_unfiltered, outfile)
    outfile.close()
    print('Unfiltered pitches have been saved.')

    inning_number = inning_number + 1