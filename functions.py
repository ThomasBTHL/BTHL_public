"""Import modules"""
import numpy.linalg
import scipy
from scipy.spatial.transform import Rotation as R
import scipy.signal as sp
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import repeat
import numpy as np
from numpy import linalg as la
import pandas as pd
from tqdm import tqdm
from time import sleep
from tkinter.ttk import *
from tkinter import *
import scipy
from scipy.signal import butter as butter
from scipy.signal import find_peaks
import time
import math
import copy
import xlrd
import c3d
import os
import re
import pickle
import openpyxl
import ast
from datetime import datetime
from datetime import timedelta

# Print Message After Packages Imported Successfully
print("Import of Packages Successful!")

"""3D inverse dynamic model is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
Contact E-Mail: a.j.r.leenen@vu.nl
Changes made by Thomas van Hogerwou

Version 1.5 (2020-07-15)"""


def convert_text(text):
    """ Converts the string text to integer when all characters in the string text are digits.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        text: string text that consists of characters and/or digits
    Returns:
        text: returns the string text converted to integer or
        returns the lowercased string text from the given string text.
    """

    # Check if all  characters in the string are digits
    if text.isdigit():
        # Converts number or string to integer
        return int(text)
    else:
        # Converts all uppercase characters to lowercase characters
        return text.lower()


def alphanum_key(text):
    """Turns the string text into a list of string and number chunks.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        text: string text that consists of characters and/or digits
    Returns:
        key: returns map object that applies the convert_text function
        to each item of an iterable and returns a list of the results.
    """

    # Turns the string text into a list of string and number chunks
    key = re.split('([0-9]+)', text)

    # The convert_text function will be applied to each item in key
    key[1::2] = map(convert_text, key[1::2])

    return key


def sorted_alphanumeric(listdir):
    """Sorts a given iterable in a natural way.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        listdir: list to be sorted in a natural way (10 before 9)
    Returns:
        listdir: sorted list in a natural way (10 after 9).
    """

    return sorted(listdir, key=alphanum_key)


def EMG_timeline(dataframe, sample_freq):
    """Construct timeline and add to the current dataframe.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        dataframe: original dataframe
        sample_freq: sample frequency (Hz)
    Returns:
        dataframe: original dataframe with added timeline in subsequent column
    """

    # Calculate the time interval based on the sample frequency
    delta_t = 1 / sample_freq

    # Calculate Timeline and Add to current dataframe in subsequent column
    time_s = np.arange(dataframe.shape[0]) * np.array(delta_t)

    # Append current numpy array with extra column containing the timeline
    dataframe = np.column_stack((dataframe, time_s))

    return dataframe

def timeline(dataframe, sample_freq):
    """Construct timeline and add to the current dataframe.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        dataframe: original dataframe
        sample_freq: sample frequency (Hz)
    Returns:
        pandas.DataFrame: original dataframe with added timeline in subsequent column
    """

    # Calculate the time interval based on the sample frequency
    delta_t = 1 / sample_freq

    # Calculate Timeline and Add to current dataframe in subsequent column
    dataframe['time_s'] = dataframe.index * delta_t

    return dataframe


# TODO: Rename predefined calibration and measurement labels to names used on SURFdrive


def rearrange_model(model, direction):
    """This function rearranges the model to enforce the order of the linked segment model.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        model: dictionary with all the segments for the calculation of the net forces and moments
        direction: order of the calculation of the net forces and moments 'bottom-up' or 'top-down'
    Returns:
        modelNew: rearranged model
    """

    # Initialise parameters
    key_order = []

    # Read number of segments in the dictionary model
    num_segments = len(model)

    # Selection of the direction to calculate the net forces and moments
    if direction == 'top-down':
        key_order = ['hand', 'forearm', 'upperarm', 'thorax', 'pelvis']  # Hand segments will be added soon
    elif direction == 'bottom-up':
        key_order = ['pelvis', 'thorax', 'upperarm', 'forearm', 'hand']  # Hand segments will be added soon

    # Rearrange the dictionaries of all the segments in the correct order
    tuples = [(key, model[key]) for key in key_order[:num_segments]]

    # Create new model with rearranged dictionaries
    modelNew = dict(tuples)

    return modelNew


def load_c3d(path):
    """Load the position data with .c3d binary files into dictionary.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        path: path or folder containing position data with .c3d binary files
    Returns:
        dictionary: containing (multiple) dataframe(s),
        samples on the rows and X, Y, Z, on the columns and ordered by marker name (keys)
    """

    # Initialise counter to append calibration and measurement names
    counter_1 = 0  # Define counter for calibrations
    counter_2 = 0  # Define counter for measurements
    counter_3 = 0  # Define counter for dictionary

    # Initialise names
    names = []

    for f in sorted_alphanumeric(os.listdir(path)):
        if f.endswith(".c3d"):
            # Predefine calibration and measurement names
            if f.endswith("Static.c3d"):
                names.append("STATIC_" + str(counter_1))
                counter_1 = counter_1 + 1
            else:
                names.append("PITCH_" + str(counter_2))
                counter_2 = counter_2 + 1

    # Initialise dictionary and nested dictionary
    dictionary = dict.fromkeys(names, dict())

    # Load position data with .c3d binary files from path or folder
    for f in sorted_alphanumeric(os.listdir(path)):

        # Initialise variables labels and data
        labels = []
        data = []

        # Load the .c3d binary files in the path or folder only
        if f.endswith(".c3d"):
            reader = c3d.Reader(open(f'{path}/{f}', 'rb'))
            print("Loading " + str(f) + " - " + str(path + "/" + f))

            # Predefined markers labels
            [labels.append(reader.point_labels[index].strip()) for index in range(reader.point_used)]

            # Read the frames containing the position data from the .c3d binary file into a numpy array
            [data.append(points) for frame, points, analog in reader.read_frames()]

            # Initialise nested dictionary
            dictionary_nested = dict.fromkeys(labels, pd.DataFrame([]))

            # Initialise pandas dataframes for key in nested dictionary containing ones
            for markers in range(reader.point_used):
                dictionary_nested[labels[markers]] = pd.DataFrame(data=np.ones((len(data), 3)), columns=["X", "Y", "Z"])

            # Loop through the markers and frames containing the position data and reorganise and combine the data from the .c3d binary file in the dictionary
            for markers in tqdm(range(reader.point_used), unit="marker"):
                # Displays progressbar in console
                sleep(0.001)
                pass

                for samples in range(len(data)):
                    dictionary_nested[labels[markers]].iloc[samples, 0:3] = data[samples][markers,
                                                                            0:3] / 1000  # Convert the position data directly from millimeter to meter

                # Loop through the markers in dictionary_nested to replace zeros with NaN values
                dictionary_nested[labels[markers]].replace(0, np.nan, inplace=True)

            # Combine the data from the calibration and measurements in dictionary_nested in dictionary
            dictionary[names[counter_3]] = dictionary_nested

            # Counter to loop through names to be defined as key in dictionary
            counter_3 = counter_3 + 1

    print("Finished")

    return dictionary


def load_c3d_innings(path):
    """Load the position data with .c3d binary files into dictionary.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam & Thomas van Hogerwou, Master student TU-Delft
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        path: path or folder containing position data with .c3d binary files
    Returns:
        dictionary: containing (multiple) dataframe(s),
        samples on the rows and X, Y, Z, on the columns and ordered by marker name (keys)
    """

    # Initialise counter to append calibration and measurement names
    counter_1 = 0  # Define counter for calibrations
    counter_2 = 1  # Define counter for measurements
    counter_3 = 0  # Define counter for dictionary

    # Initialise names
    names = []

    for f in sorted_alphanumeric(os.listdir(path)):
        if f.endswith(".c3d"):
            # Predefine calibration and measurement names
            if f.endswith("Static.c3d"):
                names.append("STATIC_" + str(counter_1))
                counter_1 = counter_1 + 1
            else:
                names.append("Inning_" + str(counter_2))
                counter_2 = counter_2 + 1

    # Initialise dictionary and nested dictionary
    dictionary = dict.fromkeys(names, dict())

    # Load position data with .c3d binary files from path or folder
    for f in sorted_alphanumeric(os.listdir(path)):

        # Initialise variables labels and data
        labels = []
        data = []

        # Load the .c3d binary files in the path or folder only
        if f.endswith(".c3d"):
            reader = c3d.Reader(open(f'{path}/{f}', 'rb'))
            print("Loading " + str(f) + " - " + str(path + "/" + f))

            # Predefined markers labels
            [labels.append(reader.point_labels[index].strip()) for index in range(reader.point_used)]

            # Read the frames containing the position data from the .c3d binary file into a numpy array
            [data.append(points) for frame, points, analog in reader.read_frames()]

            # Initialise nested dictionary
            dictionary_nested = dict.fromkeys(labels, pd.DataFrame([]))

            # Initialise pandas dataframes for key in nested dictionary containing ones
            for markers in range(reader.point_used):
                dictionary_nested[labels[markers]] = pd.DataFrame(data=np.ones((len(data), 3)), columns=["X", "Y", "Z"])

            # Loop through the markers and frames containing the position data and reorganise and combine the data from the .c3d binary file in the dictionary
            for markers in tqdm(range(reader.point_used), unit="marker"):
                # Displays progressbar in console
                sleep(0.001)
                pass

                for samples in range(len(data)):
                    dictionary_nested[labels[markers]].iloc[samples, 0:3] = data[samples][markers,
                                                                            0:3] / 1000  # Convert the position data directly from millimeter to meter

                # Loop through the markers in dictionary_nested to replace zeros with NaN values
                dictionary_nested[labels[markers]].replace(0, np.nan, inplace=True)

            # Combine the data from the calibration and measurements in dictionary_nested in dictionary
            dictionary[names[counter_3]] = dictionary_nested

            # Counter to loop through names to be defined as key in dictionary
            counter_3 = counter_3 + 1

    print("Finished")

    return dictionary


def save_dict2xlsx(dictionary, path, filename):
    """Save the position data ordered in a dictionary in .csv format to path or folder

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        dictionary: dictionary containing the position data
        path: path or folder where the .csv file(s) should be saved
        filename: name of the .csv file(s)
    Returns:
        .csv file(s): (multiple) dataframe(s) ordered by sheet which contains position data from one marker (X, Y, Z)
    """

    # Make sure the data directory is there.
    if not os.path.exists("data"):
        os.mkdir("data")

    # Define ExcelWriter object to write more than one dataframe to one sheet in the workbook
    with pd.ExcelWriter(path + "/" + filename + '.xlsx') as writer:
        print("Saving " + str(filename + '.xlsx') + " - " + str(path + "/" + filename + '.xlsx'))

        # Loop through the dictionary items and write one sheet named by key to the workbook
        for key in tqdm(dictionary, unit="marker"):
            dictionary[key].to_excel(writer, sheet_name=key, index=False)
            # Displays progressbar in console
            sleep(0.1)
            pass

    print("Finished")


def save_dict2xlsx_together(dictionary, path, filename):
    """Save the position data ordered in a dictionary in .csv format to path or folder

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        dictionary: dictionary containing the position data
        path: path or folder where the .csv file(s) should be saved
        filename: name of the .csv file(s)
    Returns:
        .csv file(s): (multiple) dataframe(s) ordered by sheet which contains position data from one marker (X, Y, Z)
    """

    # Make sure the data directory is there.
    if not os.path.exists("data"):
        os.mkdir("data")

    # Define ExcelWriter object to write more than one dataframe to one sheet in the workbook
    with pd.ExcelWriter(path + "/" + filename + '.xlsx') as writer:
        print("Saving " + str(filename + '.xlsx') + " - " + str(path + "/" + filename + '.xlsx'))

        # Loop through the dictionary items and write one sheet named by key to the workbook
        for key in dictionary:
            dictionary[key].to_excel(writer, sheet_name=key, index=True)
                # for key2 in dictionary[key]:
                #     dictionary[key][key2].to_excel(writer, sheet_name=key, index=True, columns=['a', 'b', 'c', 'd'])
            # Displays progressbar in console
            sleep(0.1)
            pass

    print("Finished")


def load_xlsx2dict(path):
    """Load the position data with .xlsx file(s) into dictionary.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        path: path or folder containing position data with .xlsx file(s)
    Returns:
        containing (multiple) dataframe(s),
        samples on the rows and X, Y, Z, on the columns and ordered by marker name (keys)
    """

    # Initialise dictionary
    dictionary = dict()

    # Load list of available data with .xlsx file(s) from path or folder
    for f in sorted_alphanumeric(os.listdir(path)):

        # Load the .xlsx file(s) in the path or folder only
        if f.endswith(".xlsx"):

            # Load the sheet names in the .xlsx file(s) that represents the marker names
            workbook = xlrd.open_workbook(f'{path}/{f}', on_demand=True)
            labels = workbook.sheet_names()

            # Initialise dictionary
            dictionary = dict.fromkeys(labels, pd.DataFrame([]))

            # Loop through the number of markers
            print("Loading " + str(f) + " - " + str(path + "/" + f))

            for markers in tqdm(range(len(labels)), unit="marker"):
                # Load the position data of each marker into dataframe
                dictionary[labels[markers]] = pd.read_excel(f'{path}/{f}', sheet_name=labels[markers])

                # Displays progressbar in console
                sleep(0.1)
                pass

    print("Finished")

    return dictionary


def calc_inertial_parameters(reg_parameters, seg_length, circumference):
    """ Calculates the thorax inertial parameters: mass (kg) and principal moments of inertia (kg * cm^2) around
    the x, y and z axis based on the segment specific regression parameters, biomechanical segment length,
    and segment circumference.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-15)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        dataframe[samples, coordinates]
        reg_parameters: regression parameters obtained from Zatsiorsky in
                        dataframe[1, 4(coefficients: mass, inertia_x, inertia_y, inertia_z)]
        seg_length: biomechanical segment length measured in centimeters (cm) in dataframe[1, 1]
        circumference: circumference of the segment measured in centimeters (cm) in dataframe[1, 1]
    Returns:
        dataframe [1, 4]: mass and inertial parameters for x, y and z
    """

    # Calculate the mass of the segment
    mass = np.array(reg_parameters[0] * seg_length * circumference ** 2 * 10 ** -5)

    # Calculate the principal moments of inertia around the x, y and z axis
    inertia_x = np.array(reg_parameters[1] * mass * seg_length ** 2 * 10 ** -2)
    inertia_y = np.array(reg_parameters[2] * mass * circumference ** 2 * 10 ** -2)
    inertia_z = np.array(reg_parameters[3] * mass * seg_length ** 2 * 10 ** -2)

    # Emerge in one dataframe
    inertial_parameters = np.transpose(np.array([mass, inertia_x, inertia_y, inertia_z]))

    return inertial_parameters


def inertial_parameters_ratio(inertial_sub, inertial_pop, segment='undefined'):
    """ The function displays the inertial parameters for a segment,
    and how these are related to the population average according to data derived from population (e.g. Zatsiorsky)

    Function is re-written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-13)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        inertial_sub: inertial parameters calculated for particular subject in
                          dataframe[1, 4(coefficients: mass, inertia_x, inertia_y, inertia_z)]
        inertial_pop: mean inertial parameters derived from population in
                          dataframe[1, 4(coefficients: mass, inertia_x, inertia_y, inertia_z)]
        segment: segment in string (standard: 'undefined')
    Returns:
        dataframe [1, 4]: mass and inertial parameters for x, y and z
    """

    # Calculate the deviation from the population expressed as a percentage (population is equal to 100%)
    mass_ratio = ((inertial_sub[0] / inertial_pop[0]) * 100) - 100
    inertial_x_ratio = ((inertial_sub[1] / inertial_pop[1]) * 100) - 100
    inertial_y_ratio = ((inertial_sub[2] / inertial_pop[2]) * 100) - 100
    inertial_z_ratio = ((inertial_sub[3] / inertial_pop[3]) * 100) - 100

    # Display percentages
    # print('Segment: ' + segment.capitalize() +
    #       '\nMass: ' + "{:.2f}".format(mass_ratio) + '%' +
    #       '\nInertial X-Axis: ' + "{:.2f}".format(inertial_x_ratio) + '%' +
    #       '\nInertial Y-Axis: ' + "{:.2f}".format(inertial_y_ratio) + '%' +
    #       '\nInertial Z-Axis: ' + "{:.2f}".format(inertial_z_ratio) + '%' + '\n')

    # Emerge in one dataframe
    inertial_ratios = np.transpose(np.array(([mass_ratio, inertial_x_ratio, inertial_y_ratio, inertial_z_ratio])))

    return inertial_ratios


def calc_derivative(signal, sample_freq):
    """ Calculates the numeric derivative of a signal.

    Function is re-written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-08-21)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        signal: dataframe or array of the signal [samples, signal] or [signal, samples]
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        differentiated_signal: dataframe or array equal to the dimensions of the signal
    """

    # Initialise differentiated_signal
    differentiated_signal = np.array(np.zeros([signal.shape[0], signal.shape[1]]))

    if signal.shape[0] < signal.shape[1]:  # Standing vector notation

        # Numeric derivative of the signal calculated according to the trapezoid theory
        for index in range(1, signal.shape[1] - 1):
            differentiated_signal[:, index] = np.array((signal[:, index + 1] - signal[:, index - 1]) / (2 * (1 / sample_freq)))

        # Correction of the first and last sample
        differentiated_signal[:, 0] = np.array((signal[:, 1] - signal[:, 0]) / (1 / sample_freq))
        differentiated_signal[:, signal.shape[1] - 1] = np.array((signal[:, signal.shape[1] - 1] - signal[:, signal.shape[1] - 2]) / (1 / sample_freq))

    elif signal.shape[0] > signal.shape[1]:  # Lying vector notation

        # Numeric derivative of the signal calculated according to the trapezoid theory
        for index in range(1, signal.shape[0] - 1):
            differentiated_signal[index, :] = np.array((signal[index + 1, :] - signal[index - 1, :]) / (2 * (1 / sample_freq)))

        # Correction of the sample shift due to the differences calculated
        differentiated_signal[0, :] = np.array((signal[1, :] - signal[0, :]) / (1 / sample_freq))
        differentiated_signal[signal.shape[0] - 1, :] = np.array((signal[signal.shape[0] - 1, :] - signal[signal.shape[0] - 2, :]) / (1 / sample_freq))

    return differentiated_signal


def matrix2vector(matrix):
    """ Converts a rotation or orientation matrix of a segments to a vector notation.

    Function is re-written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-08-26)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        matrix: rotation or orientation matrix of a segment in list[:, dataframes[3, 3]]
        [[Xx Yx Zx], [Xy Yy Zy], [Xz Yz Zz]]
    Returns:
        vector: rotation or orientation matrix of a segment in vector notation in array[:, 1:9]]
        [Xx, Xy, Xz, Yx, Yy, Yz, Z, Zy, Zz]
    """
    # Read number of samples of the list containing rotation or orientation matrices
    samples = len(matrix)

    # Initialise vector notation
    vector = np.ones([samples, 9])

    # Convert rotation or orientation to vector notation
    for index in range(len(matrix)):
        vector[index, [0, 3, 6]] = matrix[index][0, 0:3]
        vector[index, [1, 4, 7]] = matrix[index][1, 0:3]
        vector[index, [2, 5, 8]] = matrix[index][2, 0:3]

    return vector


def vector2matrix(vector):
    """ Converts the rotation or orientation vector of a segment to a matrix notation.

    Function is re-written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-08-26)

    Arguments:
        vector: rotation or orientation matrix of a segment in vector notation in array[:, 1:9]]
                [Xx, Xy, Xz, Yx, Yy, Yz, Z, Zy, Zz]
    Returns:
        matrix: rotation or orientation matrix of a segment in list[:, dataframes[3, 3]]
                [[Xx Yx Zx], [Xy Yy Zy], [Xz Yz Zz]]
    """
    # Read number of samples of the list containing rotation or orientation matrices
    samples = len(vector)

    # Convert pandas dataframe vector numpy array
    vector = np.array(vector)

    # Initialise matrix notation
    matrix = [1] * samples

    for index in range(samples):
        matrix[index] = pd.DataFrame(np.ones([3, 3]))

    # Convert rotation or orientation to vector notation
    for index in range(len(vector)):
        matrix[index].iloc[0, 0:3] = vector[index, [0, 3, 6]]
        matrix[index].iloc[1, 0:3] = vector[index, [1, 4, 7]]
        matrix[index].iloc[2, 0:3] = vector[index, [2, 5, 8]]

    return matrix


def calc_omega(gRseg, sample_freq):
    """ The angular velocity of the segment in the global reference frame is calculated by the numerical derivative of the
    orientation matrix and projection on the local segment coordination system. The angular velocity is derived and subsequently
    rotated to the global coordination system.

    Note: The function can also be used to calculate the angular velocity of one segment with respect to another segment.
    Example: segRseg (gRseg_1' * gRseg_2) inputted provides the angular velocity expressed in the local (segment) coordination system of gRseg_1.

    Function is re-written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-08-25)

    Arguments:
        gRseg: rotation matrix of the segment in list[:, dataframes[3, 3]
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        g_avSeg: angular velocity of the segment in the global coordination system [samples, X:Y:Z]
        avSeg: angular velocity of the segment in the local (segment) coordination system [samples, X:Y:Z]
    """

    # Read number of samples
    samples = len(gRseg)

    # Initialise avSeg and g_avSeg parameters
    avSeg = np.zeros([3, samples])
    g_avSeg = np.zeros([3, samples])

    # Convert gRseg to vector notation for numerical derivation
    vector_gRseg = matrix2vector(gRseg)

    # Calculate the derivative of gRseg
    vector_gRseg_derivative = calc_derivative(vector_gRseg, sample_freq)

    # Convert vector_gRseg_derivative back to matrix notation
    gRseg_derivative = vector2matrix(vector_gRseg_derivative)

    for index in range(samples):

        # Calculate the skew-symmetric matrix
        skew_matrix = 0.5 * (np.dot(gRseg_derivative[index], np.transpose(gRseg[index])) - np.dot(gRseg[index], np.transpose(gRseg_derivative[index])))  # The global angular velocity tensor with zeros on the diagonal

        # Selection of the correct elements of the positive angular velocity
        g_avSeg[:, index] = [skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]]

        # Convert segment angular velocity to local coordination system
        avSeg[:, index] = np.dot(np.linalg.inv(gRseg[index]), g_avSeg[:, index])

    return g_avSeg, avSeg


""" Functions to define local (anatomical) coordination systems for the required segments """


def calc_thorax(IJ, PX, C7, T8, circumference=96, sample_freq=[], gender='male'):
    """ Calculates the local coordination system of the trunk segment according to the ISB definition through
    bony land marks on a right-handed coordination system. The center of mass, center of mass origin, and
    inertial tensor are calculated according to the Zatsiorsky regression equations.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-13)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        dataframe[samples, X:Y:Z] in meters
            IJ: deepest point of the incisura jugularis sternalis
        PX: processus xiphoideus
        C7: processus spinosus of cervical vertebrae 7 (C7)
        T8: processus spinosus of thoracic vertebrae 8 (T8)
        circumference: circumference of the thorax (standard: 96 cm)
        gender: 'male' (standard) or 'female'
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: center of mass (com) in dataframe[samples, X:Y:Z]
                    center of mass origin (comO) in dataframe[samples, X:Y:Z]
                    rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
                    thorax segment mass (kg) and inertial tensor (gIzat; kg * m^2) in dataframe [mass:X:Y:Z]
    """

    """Definition of the ISB coordination system through bony land marks on a right-handed coordination system"""

    # Convert pandas dataframes to numpy arrays for speed optimisation
    IJ = np.array(IJ)
    PX = np.array(PX)
    C7 = np.array(C7)
    T8 = np.array(T8)

    # Definition of the ISB coordination system through bony land marks on a right-handed coordination system
    origin = np.array(IJ)  # Origin coincident with the incisura jugularis sternalis

    # Longitudinal axis pointing upwards (first axis)
    y_axis = np.array(((IJ + C7) / 2) - ((PX + T8) / 2))  # Convert from numpy array to dataframe to reset header names
    y_axis_norm = np.array([y_axis[index, :] / np.linalg.norm(y_axis[index, :]) for index in range(len(y_axis))])

    # Temporary axis
    z_axis = np.cross((IJ - (PX + T8) / 2), (C7 - (PX + T8) / 2))
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Sagittal axis pointing forwards (second axis)
    x_axis = np.cross(y_axis_norm, z_axis_norm)
    x_axis_norm = np.array([x_axis[index, :] / np.linalg.norm(x_axis[index, :]) for index in range(len(x_axis))])

    # Transversal axis pointing to the right (third axis)
    z_axis = np.cross(x_axis_norm, y_axis_norm)
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Compose rotationmatrix
    gRseg = [np.transpose(np.array((x_axis_norm[index, :], y_axis_norm[index, :], z_axis_norm[index, :]))) for index in
             range(len(z_axis_norm))]

    # Compose transformationmatrix
    gTseg = copy.deepcopy(gRseg)  # Initialise list for the transformation matrix

    for index in range(len(gRseg)):
        gTseg[index] = np.insert(gTseg[index], 3, origin[index, 0:3], axis=1)
        gTseg[index] = np.insert(gTseg[index], 3, np.array([0, 0, 0, 1]), axis=0)

    """Inertial parameters according to the Zatsiorsky regression equations"""

    # Inertial parameters are calculated according to the Zatsiorsky regression equations
    # seg_length = np.nanmean(abs(PX[:, 2] - C7[:, 2]) * 100)  # Conversion from m to cm #GLOBAL # Zat but just height
    # seg_length = np.nanmean([np.linalg.norm(PX[index, :] - C7[index, :]) for index in range(len(PX))]) * 100 # Zat but threw body
    seg_length = np.nanmean([np.linalg.norm(PX[index, :] - IJ[index, :]) for index in range(len(PX))]) * 100 # along front
    # seg_length = np.nanmean([np.linalg.norm(T8[index,:] - C7[index,:]) for index in range(len(PX))]) * 100 # along spine

    # Initialisation inertial_parameters_sub variable
    inertial_parameters = np.array([])
    COM = np.array([])

    if gender == 'male':
        # Determination of the center of mass of the segment (COM)

        # 45% From posterior to anterior (Plagenhoef - 1983)
        x_COM = T8[:, 0] + ((PX[:, 0] + IJ[:, 0]) / 2 - T8[:, 0]) * 0.45  # 'X'-Coordinates
        y_COM = T8[:, 1] + ((PX[:, 1] + IJ[:, 1]) / 2 - T8[:, 1]) * 0.45  # 'Y'-Coordinates

        # 49.34% From processus xiphoideus to processus spinosus of cervical vertebrae 7 (Leva from Zatsiorsky - 1996)
        z_COM = PX[:, 2] + (C7[:, 2] - PX[:, 2]) * 0.4934  # 'Z'-Coordinates

        # Biomechanical length correction
        seg_length = seg_length * (242.1 / 170.7)

        # Combine the x, y and z coordinates of the center of mass
        COM = np.array([x_COM, y_COM, z_COM])

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([5.72, 21.83, 1.35, 9.35])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    elif gender == 'female':
        # Determination of the center of mass of the segment (COM)

        # 45% From posterior to anterior (Plagenhoef - 1983)
        x_COM = T8[:, 0] + ((PX[:, 0] + IJ[:, 0]) / 2 - T8[:, 0]) * 0.45  # 'X'-Coordinates
        y_COM = T8[:, 1] + ((PX[:, 1] + IJ[:, 1]) / 2 - T8[:, 1]) * 0.45  # 'Y'-Coordinates

        # 49.34% From processus xiphoideus to processus spinosus of cervical vertebrae 7 (Leva from Zatsiorsky - 1996)
        z_COM = PX[:, 2] + (C7[:, 2] - PX[:, 2]) * 0.4950  # 'Z'-Coordinates

        # Combine the x, y and z coordinates of the center of mass
        COM = np.array([x_COM, y_COM, z_COM])

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([5.33, 21.71, 1.33, 9.83])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    """Check inertial parameters with average data of Zatsiorsky"""
    inertial_parameters_zat = np.transpose([11.6, 1725, 1454, 705])

    # Displays the inertial parameters for a segment and the relation with the population average
    # according to data derived from population (e.g. Zatsiorsky)
    inertial_ratios = inertial_parameters_ratio(inertial_parameters, inertial_parameters_zat, segment='thorax')

    """Construction of the inertial tensor according to Zatsiorsky"""

    # Construction of the inertial tensor (inertial_parameters_sub[mass, inertia_x, inertia_y, inertia_z]
    segIzat = np.array([[float(inertial_parameters[1]), 0, 0],
                        [0, float(inertial_parameters[2]), 0],
                        [0, 0, float(inertial_parameters[3])]])  # Inertial tensor in kg * cm^2

    # Convert inertial tensor to correct unit order
    segIzat = segIzat / (100 ** 2)  # Inertial tensor in kg * m^2

    # Post multiplication with rotation matrix (Zatsiorsky, Human Kinetics, page 286)
    gIzat = [np.dot(np.dot(gRseg[index], segIzat), np.transpose(gRseg[index])) for index in range(len(gRseg))]
    # gIzat = gRseg * segIzat * gRseg'

    """Determination of the segment acceleration"""

    # Determine segment acceleration by calculating the second derivative of the COM position data
    COM_velocity = calc_derivative(COM, sample_freq)
    COM_acceleration = calc_derivative(COM_velocity, sample_freq)

    """Determination of the segment angular velocity"""
    g_avSeg, avSeg = calc_omega(gRseg, sample_freq)
    g_alfaSeg = calc_derivative(g_avSeg, sample_freq)
    alfaSeg = calc_derivative(avSeg, sample_freq)

    """Determination of Euclidean norm angular velocity and acceleration"""
    norm_av = np.linalg.norm(np.rad2deg(g_avSeg), axis=0)  # Angular velocities converted to degrees/seconds
    norm_acceleration = np.linalg.norm(COM_acceleration, axis=0)

    """Emerge calculated parameters in one dictionary"""

    # Initialise dictionary
    dictionary = dict([])

    # Select all the calculated parameters to combine in one dictionary
    dictionary['seg_name'] = 'thorax'
    dictionary['mSeg'] = inertial_parameters[0]  # Mass of the segment
    dictionary['COM'] = COM  # Center of mass in meter
    dictionary['Origin'] = np.transpose(origin)  # Origin of the local coordination system
    dictionary['vSeg'] = COM_velocity  # Center of mass velocity in meter/seconds
    dictionary['aSeg'] = COM_acceleration  # Center of mass acceleration in meter/seconds2
    dictionary['aSegNorm'] = norm_acceleration  # Euclidean norm of the segment acceleration in the global coordination system in degrees/seconds
    dictionary['g_avSeg'] = g_avSeg  # Segment angular velocity in the global coordination system in rad/seconds
    dictionary['g_alfaSeg'] = g_alfaSeg  # Segment angular acceleration in the global coordination system in rad/seconds2
    dictionary['avSeg'] = avSeg  # Segment angular velocity in the local coordination system of the segment rad/seconds
    dictionary['avSegNorm'] = norm_av  # Euclidean norm of the segment angular velocity in the global coordination system in meter/seconds2
    dictionary['alfaSeg'] = alfaSeg  # Segment angular acceleration in the local coordination system
    dictionary['gRseg'] = gRseg  # Rotationmatrix
    dictionary['gTseg'] = gTseg  # Transformationmatrix
    dictionary['gIzat'] = gIzat  # Global inertial tensor
    dictionary['segIzat'] = segIzat  # Local inertial tensor
    dictionary['Iprincipal'] = inertial_parameters[1:4]  # Principal moments of inertia

    return dictionary


def calc_thorax_without_PX(IJ, C7, T8, circumference=96, sample_freq=[], gender='male'):
    """ Calculates the local coordination system of the trunk segment according to the ISB definition through
    bony land marks on a right-handed coordination system. The center of mass, center of mass origin, and
    inertial tensor are calculated according to the Zatsiorsky regression equations.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-13)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        dataframe[samples, X:Y:Z] in meters
        IJ: deepest point of the incisura jugularis sternalis
        C7: processus spinosus of cervical vertebrae 7 (C7)
        T8: processus spinosus of thoracic vertebrae 8 (T8)
        circumference: circumference of the thorax (standard: 96 cm)
        gender: 'male' (standard) or 'female'
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: center of mass (com) in dataframe[samples, X:Y:Z]
                    center of mass origin (comO) in dataframe[samples, X:Y:Z]
                    rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
                    thorax segment mass (kg) and inertial tensor (gIzat; kg * m^2) in dataframe [mass:X:Y:Z]
    """

    """Definition of the ISB coordination system through bony land marks on a right-handed coordination system"""

    # Convert pandas dataframes to numpy arrays for speed optimisation
    IJ = np.array(IJ)
    C7 = np.array(C7)
    T8 = np.array(T8)

    # Definition of the ISB coordination system through bony land marks on a right-handed coordination system
    origin = np.array(IJ)  # Origin coincident with the incisura jugularis sternalis

    # Longitudinal axis pointing upwards (first axis)
    y_axis = np.array((C7 - T8))  # Convert from numpy array to dataframe to reset header names
    y_axis_norm = np.array([y_axis[index, :] / np.linalg.norm(y_axis[index, :]) for index in range(len(y_axis))])

    # Temporary axis
    z_axis = np.cross((IJ - T8), (C7 - T8))
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Sagittal axis pointing forwards (second axis)
    x_axis = np.cross(y_axis_norm, z_axis_norm)
    x_axis_norm = np.array([x_axis[index, :] / np.linalg.norm(x_axis[index, :]) for index in range(len(x_axis))])

    # Transversal axis pointing to the right (third axis)
    z_axis = np.cross(x_axis_norm, y_axis_norm)
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Compose rotationmatrix
    gRseg = [np.transpose(np.array((x_axis_norm[index, :], y_axis_norm[index, :], z_axis_norm[index, :]))) for index in
             range(len(z_axis_norm))]

    # Compose transformationmatrix
    gTseg = copy.deepcopy(gRseg)  # Initialise list for the transformation matrix

    for index in range(len(gRseg)):
        gTseg[index] = np.insert(gTseg[index], 3, origin[index, 0:3], axis=1)
        gTseg[index] = np.insert(gTseg[index], 3, np.array([0, 0, 0, 1]), axis=0)

    """Inertial parameters according to the Zatsiorsky regression equations"""

    # Inertial parameters are calculated according to the Zatsiorsky regression equations
    seg_length = np.nanmean(abs(IJ[:, 2] - C7[:, 2]) * 100)  # Conversion from m to cm  #todo used here IJ instead of PX

    # Initialisation inertial_parameters_sub variable
    inertial_parameters = np.array([])
    COM = np.array([])

    if gender == 'male':
        # Determination of the center of mass of the segment (COM)

        # 45% From posterior to anterior (Plagenhoef - 1983)
        x_COM = T8[:, 0] + ((IJ[:, 0] + IJ[:, 0]) / 2 - T8[:, 0]) * 0.45  # 'X'-Coordinates #todo used here IJ instead of PX
        y_COM = T8[:, 1] + ((IJ[:, 1] + IJ[:, 1]) / 2 - T8[:, 1]) * 0.45  # 'Y'-Coordinates #todo used here IJ instead of PX

        # 49.34% From processus xiphoideus to processus spinosus of cervical vertebrae 7 (Leva from Zatsiorsky - 1996)
        z_COM = IJ[:, 2] + (C7[:, 2] - IJ[:, 2]) * 0.4934  # 'Z'-Coordinates #todo used here IJ instead of PX

        # Combine the x, y and z coordinates of the center of mass
        COM = np.array([x_COM, y_COM, z_COM])

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([5.72, 21.83, 1.35, 9.35])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    elif gender == 'female':
        # Determination of the center of mass of the segment (COM)

        # 45% From posterior to anterior (Plagenhoef - 1983)
        x_COM = T8[:, 0] + ((PX[:, 0] + IJ[:, 0]) / 2 - T8[:, 0]) * 0.45  # 'X'-Coordinates
        y_COM = T8[:, 1] + ((PX[:, 1] + IJ[:, 1]) / 2 - T8[:, 1]) * 0.45  # 'Y'-Coordinates

        # 49.34% From processus xiphoideus to processus spinosus of cervical vertebrae 7 (Leva from Zatsiorsky - 1996)
        z_COM = IJ[:, 2] + (C7[:, 2] - IJ[:, 2]) * 0.4950  # 'Z'-Coordinates

        # Combine the x, y and z coordinates of the center of mass
        COM = np.array([x_COM, y_COM, z_COM])

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([5.33, 21.71, 1.33, 9.83])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    """Check inertial parameters with average data of Zatsiorsky"""
    inertial_parameters_zat = np.transpose([11.6, 1725, 1454, 705])

    # Displays the inertial parameters for a segment and the relation with the population average
    # according to data derived from population (e.g. Zatsiorsky)
    inertial_ratios = inertial_parameters_ratio(inertial_parameters, inertial_parameters_zat, segment='thorax')

    """Construction of the inertial tensor according to Zatsiorsky"""

    # Construction of the inertial tensor (inertial_parameters_sub[mass, inertia_x, inertia_y, inertia_z]
    segIzat = np.array([[float(inertial_parameters[1]), 0, 0],
                        [0, float(inertial_parameters[2]), 0],
                        [0, 0, float(inertial_parameters[3])]])  # Inertial tensor in kg * cm^2

    # Convert inertial tensor to correct unit order
    segIzat = segIzat / (100 ** 2)  # Inertial tensor in kg * m^2

    # Post multiplication with rotation matrix (Zatsiorsky, Human Kinetics, page 286)
    gIzat = [np.dot(np.dot(gRseg[index], segIzat), np.transpose(gRseg[index])) for index in range(len(gRseg))]
    # gIzat = gRseg * segIzat * gRseg'

    """Determination of the segment acceleration"""

    # Determine segment acceleration by calculating the second derivative of the COM position data
    COM_velocity = calc_derivative(COM, sample_freq)
    COM_acceleration = calc_derivative(COM_velocity, sample_freq)

    """Determination of the segment angular velocity"""
    g_avSeg, avSeg = calc_omega(gRseg, sample_freq)
    g_alfaSeg = calc_derivative(g_avSeg, sample_freq)
    alfaSeg = calc_derivative(avSeg, sample_freq)

    """Determination of Euclidean norm angular velocity and acceleration"""
    norm_av = np.linalg.norm(np.rad2deg(g_avSeg), axis=0)  # Angular velocities converted to degrees/seconds
    norm_acceleration = np.linalg.norm(COM_acceleration, axis=0)

    """Emerge calculated parameters in one dictionary"""

    # Initialise dictionary
    dictionary = dict([])

    # Select all the calculated parameters to combine in one dictionary
    dictionary['seg_name'] = 'thorax'
    dictionary['mSeg'] = inertial_parameters[0]  # Mass of the segment
    dictionary['COM'] = COM  # Center of mass in meter
    dictionary['Origin'] = np.transpose(origin)  # Origin of the local coordination system
    dictionary['vSeg'] = COM_velocity  # Center of mass velocity in meter/seconds
    dictionary['aSeg'] = COM_acceleration  # Center of mass acceleration in meter/seconds2
    dictionary['aSegNorm'] = norm_acceleration  # Euclidean norm of the segment acceleration in the global coordination system in degrees/seconds
    dictionary['g_avSeg'] = g_avSeg  # Segment angular velocity in the global coordination system in rad/seconds
    dictionary['g_alfaSeg'] = g_alfaSeg  # Segment angular acceleration in the global coordination system in rad/seconds2
    dictionary['avSeg'] = avSeg  # Segment angular velocity in the local coordination system of the segment rad/seconds
    dictionary['avSegNorm'] = norm_av  # Euclidean norm of the segment angular velocity in the global coordination system in meter/seconds2
    dictionary['alfaSeg'] = alfaSeg  # Segment angular acceleration in the local coordination system
    dictionary['gRseg'] = gRseg  # Rotationmatrix
    dictionary['gTseg'] = gTseg  # Transformationmatrix
    dictionary['gIzat'] = gIzat  # Global inertial tensor
    dictionary['segIzat'] = segIzat  # Local inertial tensor
    dictionary['Iprincipal'] = inertial_parameters[1:4]  # Principal moments of inertia

    return dictionary


def calc_pelvis(RSIAS, LSIAS, RSIPS, LSIPS, sample_freq=[], circumference=97, gender='male'):
    """ Calculates the local coordination system of the pelvis segment according to the ISB definition through
    bony land marks on a right-handed coordination system. The center of mass, center of mass origin, and
    inertial tensor are calculated according to the Zatsiorsky regression equations.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-25)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        dataframe[samples, X:Y:Z] in meters
        RSIAS: right spina iliaca anterior superior
        LSIAS: left spina iliaca anterior superior
        RSIPS: right spina iliaca posterior superior
        LSIPS: left spina iliaca posterior superior
        circumference: circumference of the pelvis (standard: 97 cm)
        gender: 'male' (standard) or 'female'
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: center of mass (com) in dataframe[samples, X:Y:Z]
                    center of mass origin (comO) in dataframe[samples, X:Y:Z]
                    rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
                    thorax segment mass (kg) and inertial tensor (gIzat; kg * m^2) in dataframe [mass:X:Y:Z]
    """

    """Definition of the ISB coordination system through bony land marks on a right-handed coordination system"""

    # Convert pandas dataframes to numpy arrays for speed optimisation
    RSIAS = np.array(RSIAS)
    LSIAS = np.array(LSIAS)
    RSIPS = np.array(RSIPS)
    LSIPS = np.array(LSIPS)

    # Definition of the ISB coordination system through bony land marks on a right-handed coordination system
    origin = (LSIAS + RSIAS) / 2  # Origin coincident with the midpoint between the right and left spina iliaca anterior superior

    # Temporary axis (sacral root; SR) and plane (quasi transversal plane with normal vector pointing upwards; QTP)
    SR = (LSIPS + RSIPS) / 2  # Sacral root axis
    QTP = np.cross((SR - origin), (LSIAS - origin))  # Quasi transversal plane

    # Transversal axis pointing to the right (first axis)
    z_axis = RSIAS - LSIAS  # Convert from numpy array to dataframe to reset header names
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Sagittal axis pointing forwards (second axis)
    x_axis = np.cross(QTP, z_axis_norm)
    x_axis_norm = np.array([x_axis[index, :] / np.linalg.norm(x_axis[index, :]) for index in range(len(x_axis))])

    # Longitudinal axis pointing upwards (third axis)
    y_axis = np.cross(z_axis_norm, x_axis_norm)
    y_axis_norm = np.array([y_axis[index, :] / np.linalg.norm(y_axis[index, :]) for index in range(len(y_axis))])

    # Compose rotationmatrix. transpose is taken to correct for cross products giving laying vectors instead of standing
    gRseg = [np.transpose(np.array((x_axis_norm[index, :], y_axis_norm[index, :], z_axis_norm[index, :]))) for index in
             range(len(z_axis_norm))]

    # Compose transformationmatrix
    gTseg = copy.deepcopy(gRseg)  # Initialise list for the transformation matrix

    for index in range(len(gRseg)):
        gTseg[index] = np.insert(gTseg[index], 3, origin[index, 0:3], axis=1)
        gTseg[index] = np.insert(gTseg[index], 3, np.array([0, 0, 0, 1]), axis=0)

    """Calculations of the hip joint centers"""

    # Calculate the depth and width of the pelvis segment
    pelvis_depth = (origin - SR)
    pelvis_depth_norm = np.array([np.linalg.norm(pelvis_depth[index, :]) for index in range(len(pelvis_depth))])

    pelvis_width = RSIAS - LSIAS
    pelvis_width_norm = np.array([np.linalg.norm(pelvis_width[index, :]) for index in range(len(pelvis_width))])

    # Calculate the hip joint centers based on pelvis width according Bell (1999) and pelvis depth according to Leardini (1999)

    # Right hip joint center (local)
    RHJC = np.array([[-0.31 * pelvis_depth_norm[index],
                      -0.30 * pelvis_width_norm[index],
                      0.36 * pelvis_width_norm[index]] for index in range(len(pelvis_width_norm))])

    # Left hip joint center
    LHJC = np.array([[-0.31 * pelvis_depth_norm[index],
                      -0.30 * pelvis_width_norm[index],
                      -0.36 * pelvis_width_norm[index]] for index in range(len(pelvis_width_norm))])

    # Initialise dictionary
    JC = dict([])

    # Emerge both dataframes in one dictionary
    JC['RHJC'] = np.transpose(RHJC)
    JC['LHJC'] = np.transpose(LHJC)

    """Conversion of the bony landmarks to the local coordination system of the pelvis"""

    # Conversion to local coordination system
    #segMSIAS = np.array([gRseg[index].dot(((LSIAS + RSIAS) / 2)[index]) for index in range(len(pelvis_width_norm))])
    segMSIAS = numpy.zeros(numpy.shape(origin))
    #segMSIPS = np.array([numpy.ndarray.transpose(gRseg[index]).dot(((LSIPS + RSIPS) / 2 - origin)[index]) for index in range(len(pelvis_width_norm))])
    segMSIPS = np.array([(gRseg[index]).dot(((LSIPS + RSIPS) / 2 - origin)[index]) for index in range(len(pelvis_width_norm))])


    # Midpoint hip joint center
    segMHJC = np.array((RHJC + LHJC) / 2)

    """Inertial parameters according to the Zatsiorsky regression equations"""

    # Inertial parameters are calculated according to the Zatsiorsky regression equations
    seg_length = np.nanmean(abs(segMSIAS[:, 1] - segMHJC[:, 1]) * 100)  # Conversion from m to cm

    # Initialisation inertial_parameters_sub variable
    inertial_parameters = np.array([])
    segCOM = np.array([])

    if gender == 'male':
        # Determination of the center of mass of the segment (COM)

        # 63% From anterior to posterior (Plagenhoef - 1983)
        x_COM = segMSIPS[:, 0] * 0.63

        # Create pandas series containing zeros
        y_COM = np.zeros(len(x_COM))

        # 38.85% From hip joint center to midpoint of the spina iliaca anterior superior (Leva from Zatsiorsky - 1996)
        z_COM = segMHJC[:, 2] + (segMSIAS[:, 2] - segMHJC[:, 2]) * 0.3885

        # Combine the x, y and z coordinates of the center of mass
        segCOM = np.array([x_COM, y_COM, z_COM])

        # Calculate the biomechanical length of the pelvis segment (segment length correction)
        # Omphalion to the hip segmentation planes intersection, female: 256.8, male: 251.7 (biomechanical length
        # Omphalion to the midpoint of both hip joint centers, female: 181.5, male: 145.7 (alternative length)
        seg_length = seg_length * (251.7 / (251.7 - 145.7))

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([3.60, 10.90, 0.76, 8.92])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    elif gender == 'female':
        # Determination of the center of mass of the segment (COM)

        # 63% From anterior to posterior (Plagenhoef - 1983)
        x_COM = segMSIPS[:, 0] * 0.63

        # Create pandas series containing zeros
        y_COM = np.zeros(len(x_COM))

        # 50.80% From hip joint center to midpoint of the spina iliaca anterior superior (Leva from Zatsiorsky - 1996)
        z_COM = segMHJC[:, 2] + (segMSIAS[:, 2] - segMHJC[:, 2]) * 0.5080

        # Combine the x, y and z coordinates of the center of mass
        segCOM = np.array([x_COM, y_COM, z_COM])

        # Calculate the biomechanical length of the pelvis segment (segment length correction)
        # Omphalion to the hip segmentation planes intersection, female: 256.8, male: 251.7 (biomechanical length
        # Omphalion to the midpoint of both hip joint centers, female: 181.5, male: 145.7 (alternative length)
        seg_length = seg_length * (256.8 / 181.5)

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([3.43, 9.37, 0.74, 8.07])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    # Convert the center of mass defined in the local coordination system of the pelvis to global
    COM = np.transpose(np.array([np.dot(gRseg[index], segCOM[:, index]) for index in range(len(gRseg))]))

    """Check inertial parameters with average data of Zatsiorsky"""
    inertial_parameters_zat = np.transpose([8.16, 656, 592, 525])

    # Displays the inertial parameters for a segment and the relation with the population average
    # according to data derived from population (e.g. Zatsiorsky)
    inertial_ratios = inertial_parameters_ratio(inertial_parameters, inertial_parameters_zat, segment='pelvis')

    """Construction of the inertial tensor according to Zatsiorsky"""

    # Construction of the inertial tensor (inertial_parameters_sub[mass, inertia_x, inertia_y, inertia_z]
    segIzat = np.array([[float(inertial_parameters[1]), 0, 0],
                        [0, float(inertial_parameters[2]), 0],
                        [0, 0, float(inertial_parameters[3])]])  # Inertial tensor in kg * cm^2

    # Convert inertial tensor to correct unit order
    segIzat = segIzat / (100 ** 2)  # Inertial tensor in kg * m^2

    # Post multiplication with rotation matrix (Zatsiorsky, Human Kinetics, page 286)
    gIzat = [np.dot(np.dot(gRseg[index], segIzat), np.transpose(gRseg[index])) for index in range(len(gRseg))]
    # gIzat = gRseg * segIzat * gRseg'

    """Determination of the segment acceleration"""

    # Determine segment acceleration by calculating the second derivative of the COM position data
    COM_velocity = calc_derivative(COM, sample_freq)
    COM_acceleration = calc_derivative(COM_velocity, sample_freq)

    """Determination of the segment angular velocity"""
    g_avSeg, avSeg = calc_omega(gRseg, sample_freq)
    g_alfaSeg = calc_derivative(g_avSeg, sample_freq)
    alfaSeg = calc_derivative(avSeg, sample_freq)

    """Determination of Euclidean norm angular velocity and acceleration"""
    norm_av = np.linalg.norm(np.rad2deg(g_avSeg), axis=0)  # Angular velocities converted to degrees/seconds
    norm_acceleration = np.linalg.norm(COM_acceleration, axis=0)

    """Emerge calculated parameters in one dictionary"""

    # Initialise dictionary
    dictionary = dict([])

    # Select all the calculated parameters to combine in one dictionary
    dictionary['seg_name'] = 'pelvis'
    dictionary['mSeg'] = inertial_parameters[0]  # Mass of the segment
    dictionary['COM'] = COM  # Center of mass in meter
    dictionary['Origin'] = np.transpose(origin)  # Origin of the local coordination system
    dictionary['vSeg'] = COM_velocity  # Center of mass velocity in meter/seconds
    dictionary['aSeg'] = COM_acceleration  # Center of mass acceleration in meter/seconds2
    dictionary['aSegNorm'] = norm_acceleration  # Euclidean norm of the segment acceleration in the global coordination system in degrees/seconds
    dictionary['g_avSeg'] = g_avSeg  # Segment angular velocity in the global coordination system in rad/seconds
    dictionary['g_alfaSeg'] = g_alfaSeg  # Segment angular acceleration in the global coordination system in rad/seconds2
    dictionary['avSeg'] = avSeg  # Segment angular velocity in the local coordination system of the segment rad/seconds
    dictionary['avSegNorm'] = norm_av  # Euclidean norm of the segment angular velocity in the global coordination system in meter/seconds2
    dictionary['alfaSeg'] = alfaSeg  # Segment angular acceleration in the local coordination system
    dictionary['gRseg'] = gRseg  # Rotationmatrix
    dictionary['gTseg'] = gTseg  # Transformationmatrix
    dictionary['gIzat'] = gIzat  # Global inertial tensor
    dictionary['segIzat'] = segIzat  # Local inertial tensor
    dictionary['JC'] = JC  # Right and left hip joint center
    dictionary['Iprincipal'] = inertial_parameters[1:4]  # Principal moments of inertia

    return dictionary


def calc_pelvis_without_LASIS(RSIAS, LSIAS, RSIPS, LSIPS, sample_freq=[], circumference=97, gender='male'):
    """ Quick and dirty solutions: Makes local coordidnate system without LSIAS, can only be used for the magnitude not for other calculations!
    The center of mass, center of mass origin, and
    inertial tensor are calculated according to the Zatsiorsky regression equations.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-25)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        dataframe[samples, X:Y:Z] in meters
        RSIAS: right spina iliaca anterior superior
        LSIAS: left spina iliaca anterior superior
        RSIPS: right spina iliaca posterior superior
        LSIPS: left spina iliaca posterior superior
        circumference: circumference of the pelvis (standard: 97 cm)
        gender: 'male' (standard) or 'female'
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: center of mass (com) in dataframe[samples, X:Y:Z]
                    center of mass origin (comO) in dataframe[samples, X:Y:Z]
                    rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
                    thorax segment mass (kg) and inertial tensor (gIzat; kg * m^2) in dataframe [mass:X:Y:Z]
    """

    """Definition of the ISB coordination system through bony land marks on a right-handed coordination system"""

    # Convert pandas dataframes to numpy arrays for speed optimisation
    RSIAS = np.array(RSIAS)
    LSIAS = np.array(LSIAS)
    RSIPS = np.array(RSIPS)
    LSIPS = np.array(LSIPS)

    # Definition of the ISB coordination system through bony land marks on a right-handed coordination system
    origin = (LSIPS + RSIPS) / 2  # Origin coincident with the midpoint between the right and left spina iliaca anterior superior

    # Temporary axis (sacral root; SR) and plane (quasi transversal plane with normal vector pointing upwards; QTP)
    SR = (LSIPS + RSIPS) / 2  # Sacral root axis
    QTP = np.cross((SR - origin), (LSIAS - origin))  # Quasi transversal plane

    # Transversal axis pointing to the right (first axis)
    z_axis = LSIPS - RSIPS  # Convert from numpy array to dataframe to reset header names
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Sagittal axis pointing forwards (second axis)
    x_axis = np.cross((RSIAS-RSIPS), z_axis_norm)
    x_axis_norm = np.array([x_axis[index, :] / np.linalg.norm(x_axis[index, :]) for index in range(len(x_axis))])

    # Longitudinal axis pointing upwards (third axis)
    y_axis = np.cross(z_axis_norm, x_axis_norm)
    y_axis_norm = np.array([y_axis[index, :] / np.linalg.norm(y_axis[index, :]) for index in range(len(y_axis))])

    # Compose rotationmatrix
    gRseg = [np.transpose(np.array((x_axis_norm[index, :], y_axis_norm[index, :], z_axis_norm[index, :]))) for index in
             range(len(z_axis_norm))]

    # Compose transformationmatrix
    gTseg = copy.deepcopy(gRseg)  # Initialise list for the transformation matrix

    for index in range(len(gRseg)):
        gTseg[index] = np.insert(gTseg[index], 3, origin[index, 0:3], axis=1)
        gTseg[index] = np.insert(gTseg[index], 3, np.array([0, 0, 0, 1]), axis=0)

    """Calculations of the hip joint centers"""

    # Calculate the depth and width of the pelvis segment
    pelvis_depth = (((RSIAS + LSIAS) / 2) - SR)
    pelvis_depth_norm = np.array([np.linalg.norm(pelvis_depth[index, :]) for index in range(len(pelvis_depth))])

    pelvis_width = RSIAS - LSIAS
    pelvis_width_norm = np.array([np.linalg.norm(pelvis_width[index, :]) for index in range(len(pelvis_width))])

    # Calculate the hip joint centers based on pelvis width according Bell (1999) and pelvis depth according to Leardini (1999)

    # Right hip joint center
    RHJC = np.array([[-0.31 * pelvis_depth_norm[index],
                      -0.36 * pelvis_width_norm[index],
                      -0.30 * pelvis_width_norm[index]] for index in range(len(pelvis_width_norm))])

    # Left hip joint center
    LHJC = np.array([[-0.31 * pelvis_depth_norm[index],
                      0.36 * pelvis_width_norm[index],
                      -0.30 * pelvis_width_norm[index]] for index in range(len(pelvis_width_norm))])

    # Initialise dictionary
    JC = dict([])

    # Emerge both dataframes in one dictionary
    JC['RHJC'] = np.transpose(RHJC)
    JC['LHJC'] = np.transpose(LHJC)

    """Conversion of the bony landmarks to the local coordination system of the pelvis"""

    # Conversion to local coordination system
    segMSIAS = np.array([gRseg[index].dot(((LSIAS + RSIAS) / 2)[index])
                         for index in range(len(pelvis_width_norm))])

    segMSIPS = np.array([gRseg[index].dot(((LSIPS + RSIPS) / 2 - (LSIAS + RSIAS) / 2)[index])
                         for index in range(len(pelvis_width_norm))])

    # Midpoint hip joint center
    segMHJC = np.array((RHJC + LHJC) / 2)

    """Inertial parameters according to the Zatsiorsky regression equations"""

    # Inertial parameters are calculated according to the Zatsiorsky regression equations
    seg_length = np.nanmean(abs(segMSIAS[:, 2] - segMHJC[:, 2]) * 100)  # Conversion from mm to cm

    # Initialisation inertial_parameters_sub variable
    inertial_parameters = np.array([])
    segCOM = np.array([])

    if gender == 'male':
        # Determination of the center of mass of the segment (COM)

        # 63% From anterior to posterior (Plagenhoef - 1983)
        x_COM = segMSIPS[:, 0] * 0.63

        # Create pandas series containing zeros
        y_COM = np.zeros(len(x_COM))

        # 38.85% From hip joint center to midpoint of the spina iliaca anterior superior (Leva from Zatsiorsky - 1996)
        z_COM = segMHJC[:, 2] + (segMSIAS[:, 2] - segMHJC[:, 2]) * 0.3885

        # Combine the x, y and z coordinates of the center of mass
        segCOM = np.array([x_COM, y_COM, z_COM])

        # Calculate the biomechanical length of the pelvis segment (segment length correction)
        # Omphalion to the hip segmentation planes intersection, female: 256.8, male: 251.7 (biomechanical length
        # Omphalion to the midpoint of both hip joint centers, female: 181.5, male: 145.7 (alternative length)
        seg_length = seg_length * (251.7 / 145.7)

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([3.60, 10.90, 0.76, 8.92])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    elif gender == 'female':
        # Determination of the center of mass of the segment (COM)

        # 63% From anterior to posterior (Plagenhoef - 1983)
        x_COM = segMSIPS[:, 0] * 0.63

        # Create pandas series containing zeros
        y_COM = np.zeros(len(x_COM))

        # 50.80% From hip joint center to midpoint of the spina iliaca anterior superior (Leva from Zatsiorsky - 1996)
        z_COM = segMHJC[:, 2] + (segMSIAS[:, 2] - segMHJC[:, 2]) * 0.5080

        # Combine the x, y and z coordinates of the center of mass
        segCOM = np.array([x_COM, y_COM, z_COM])

        # Calculate the biomechanical length of the pelvis segment (segment length correction)
        # Omphalion to the hip segmentation planes intersection, female: 256.8, male: 251.7 (biomechanical length
        # Omphalion to the midpoint of both hip joint centers, female: 181.5, male: 145.7 (alternative length)
        seg_length = seg_length * (256.8 / 181.5)

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([3.43, 9.37, 0.74, 8.07])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    # Convert the center of mass defined in the local coordination system of the pelvis to global
    COM = np.transpose(np.array([np.dot(gRseg[index], segCOM[:, index]) for index in range(len(gRseg))]))

    """Check inertial parameters with average data of Zatsiorsky"""
    inertial_parameters_zat = np.transpose([8.16, 656, 592, 525])

    # Displays the inertial parameters for a segment and the relation with the population average
    # according to data derived from population (e.g. Zatsiorsky)
    inertial_ratios = inertial_parameters_ratio(inertial_parameters, inertial_parameters_zat, segment='pelvis')

    """Construction of the inertial tensor according to Zatsiorsky"""

    # Construction of the inertial tensor (inertial_parameters_sub[mass, inertia_x, inertia_y, inertia_z]
    segIzat = np.array([[float(inertial_parameters[1]), 0, 0],
                        [0, float(inertial_parameters[2]), 0],
                        [0, 0, float(inertial_parameters[3])]])  # Inertial tensor in kg * cm^2

    # Convert inertial tensor to correct unit order
    segIzat = segIzat / (100 ** 2)  # Inertial tensor in kg * m^2

    # Post multiplication with rotation matrix (Zatsiorsky, Human Kinetics, page 286)
    gIzat = [np.dot(np.dot(gRseg[index], segIzat), np.transpose(gRseg[index])) for index in range(len(gRseg))]
    # gIzat = gRseg * segIzat * gRseg'

    """Determination of the segment acceleration"""

    # Determine segment acceleration by calculating the second derivative of the COM position data
    COM_velocity = calc_derivative(COM, sample_freq)
    COM_acceleration = calc_derivative(COM_velocity, sample_freq)

    """Determination of the segment angular velocity"""
    g_avSeg, avSeg = calc_omega(gRseg, sample_freq)
    g_alfaSeg = calc_derivative(g_avSeg, sample_freq)
    alfaSeg = calc_derivative(avSeg, sample_freq)

    """Determination of Euclidean norm angular velocity and acceleration"""
    norm_av = np.linalg.norm(np.rad2deg(g_avSeg), axis=0)  # Angular velocities converted to degrees/seconds
    norm_acceleration = np.linalg.norm(COM_acceleration, axis=0)

    """Emerge calculated parameters in one dictionary"""

    # Initialise dictionary
    dictionary = dict([])

    # Select all the calculated parameters to combine in one dictionary
    dictionary['seg_name'] = 'pelvis'
    dictionary['mSeg'] = inertial_parameters[0]  # Mass of the segment
    dictionary['COM'] = COM  # Center of mass in meter
    dictionary['Origin'] = np.transpose(origin)  # Origin of the local coordination system
    dictionary['vSeg'] = COM_velocity  # Center of mass velocity in meter/seconds
    dictionary['aSeg'] = COM_acceleration  # Center of mass acceleration in meter/seconds2
    dictionary['aSegNorm'] = norm_acceleration  # Euclidean norm of the segment acceleration in the global coordination system in degrees/seconds
    dictionary['g_avSeg'] = g_avSeg  # Segment angular velocity in the global coordination system in rad/seconds
    dictionary['g_alfaSeg'] = g_alfaSeg  # Segment angular acceleration in the global coordination system in rad/seconds2
    dictionary['avSeg'] = avSeg  # Segment angular velocity in the local coordination system of the segment rad/seconds
    dictionary['avSegNorm'] = norm_av  # Euclidean norm of the segment angular velocity in the global coordination system in meter/seconds2
    dictionary['alfaSeg'] = alfaSeg  # Segment angular acceleration in the local coordination system
    dictionary['gRseg'] = gRseg  # Rotationmatrix
    dictionary['gTseg'] = gTseg  # Transformationmatrix
    dictionary['gIzat'] = gIzat  # Global inertial tensor
    dictionary['segIzat'] = segIzat  # Local inertial tensor
    dictionary['JC'] = JC  # Right and left hip joint center
    dictionary['Iprincipal'] = inertial_parameters[1:4]  # Principal moments of inertia

    return dictionary


def calc_pelvis_without_RASIS(RSIAS, LSIAS, RSIPS, LSIPS, sample_freq=[], circumference=97, gender='male'):
    """ Quick and dirty solutions: Makes local coordidnate system without RSIAS, can only be used for the magnitude not for other calculations!
    The center of mass, center of mass origin, and
    inertial tensor are calculated according to the Zatsiorsky regression equations.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-25)
    Version 1.1 (2020-11-05) - Pandas actions converted no numpy actions due to speed improvements

    Arguments:
        dataframe[samples, X:Y:Z] in meters
        RSIAS: right spina iliaca anterior superior
        LSIAS: left spina iliaca anterior superior
        RSIPS: right spina iliaca posterior superior
        LSIPS: left spina iliaca posterior superior
        circumference: circumference of the pelvis (standard: 97 cm)
        gender: 'male' (standard) or 'female'
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: center of mass (com) in dataframe[samples, X:Y:Z]
                    center of mass origin (comO) in dataframe[samples, X:Y:Z]
                    rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
                    thorax segment mass (kg) and inertial tensor (gIzat; kg * m^2) in dataframe [mass:X:Y:Z]
    """

    """Definition of the ISB coordination system through bony land marks on a right-handed coordination system"""

    # Convert pandas dataframes to numpy arrays for speed optimisation
    RSIAS = np.array(RSIAS)
    LSIAS = np.array(LSIAS)
    RSIPS = np.array(RSIPS)
    LSIPS = np.array(LSIPS)

    # Definition of the ISB coordination system through bony land marks on a right-handed coordination system
    origin = (LSIPS + RSIPS) / 2  # Origin coincident with the midpoint between the right and left spina iliaca anterior superior

    # Temporary axis (sacral root; SR) and plane (quasi transversal plane with normal vector pointing upwards; QTP)
    SR = (LSIPS + RSIPS) / 2  # Sacral root axis
    QTP = np.cross((SR - origin), (LSIAS - origin))  # Quasi transversal plane

    # Transversal axis pointing to the right (first axis)
    z_axis = LSIPS - RSIPS  # Convert from numpy array to dataframe to reset header names
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Sagittal axis pointing forwards (second axis)
    x_axis = np.cross((LSIAS-LSIPS), z_axis_norm)
    x_axis_norm = np.array([x_axis[index, :] / np.linalg.norm(x_axis[index, :]) for index in range(len(x_axis))])

    # Longitudinal axis pointing upwards (third axis)
    y_axis = np.cross(z_axis_norm, x_axis_norm)
    y_axis_norm = np.array([y_axis[index, :] / np.linalg.norm(y_axis[index, :]) for index in range(len(y_axis))])

    # Compose rotationmatrix
    gRseg = [np.transpose(np.array((x_axis_norm[index, :], y_axis_norm[index, :], z_axis_norm[index, :]))) for index in
             range(len(z_axis_norm))]

    # Compose transformationmatrix
    gTseg = copy.deepcopy(gRseg)  # Initialise list for the transformation matrix

    for index in range(len(gRseg)):
        gTseg[index] = np.insert(gTseg[index], 3, origin[index, 0:3], axis=1)
        gTseg[index] = np.insert(gTseg[index], 3, np.array([0, 0, 0, 1]), axis=0)

    """Calculations of the hip joint centers"""

    # Calculate the depth and width of the pelvis segment
    pelvis_depth = (((RSIAS + LSIAS) / 2) - SR)
    pelvis_depth_norm = np.array([np.linalg.norm(pelvis_depth[index, :]) for index in range(len(pelvis_depth))])

    pelvis_width = RSIAS - LSIAS
    pelvis_width_norm = np.array([np.linalg.norm(pelvis_width[index, :]) for index in range(len(pelvis_width))])

    # Calculate the hip joint centers based on pelvis width according Bell (1999) and pelvis depth according to Leardini (1999)

    # Right hip joint center
    RHJC = np.array([[-0.31 * pelvis_depth_norm[index],
                      -0.36 * pelvis_width_norm[index],
                      -0.30 * pelvis_width_norm[index]] for index in range(len(pelvis_width_norm))])

    # Left hip joint center
    LHJC = np.array([[-0.31 * pelvis_depth_norm[index],
                      0.36 * pelvis_width_norm[index],
                      -0.30 * pelvis_width_norm[index]] for index in range(len(pelvis_width_norm))])

    # Initialise dictionary
    JC = dict([])

    # Emerge both dataframes in one dictionary
    JC['RHJC'] = np.transpose(RHJC)
    JC['LHJC'] = np.transpose(LHJC)

    """Conversion of the bony landmarks to the local coordination system of the pelvis"""

    # Conversion to local coordination system
    segMSIAS = np.array([gRseg[index].dot(((LSIAS + RSIAS) / 2)[index])
                         for index in range(len(pelvis_width_norm))])

    segMSIPS = np.array([gRseg[index].dot(((LSIPS + RSIPS) / 2 - (LSIAS + RSIAS) / 2)[index])
                         for index in range(len(pelvis_width_norm))])

    # Midpoint hip joint center
    segMHJC = np.array((RHJC + LHJC) / 2)

    """Inertial parameters according to the Zatsiorsky regression equations"""

    # Inertial parameters are calculated according to the Zatsiorsky regression equations
    seg_length = np.nanmean(abs(segMSIAS[:, 2] - segMHJC[:, 2]) * 100)  # Conversion from mm to cm

    # Initialisation inertial_parameters_sub variable
    inertial_parameters = np.array([])
    segCOM = np.array([])

    if gender == 'male':
        # Determination of the center of mass of the segment (COM)

        # 63% From anterior to posterior (Plagenhoef - 1983)
        x_COM = segMSIPS[:, 0] * 0.63

        # Create pandas series containing zeros
        y_COM = np.zeros(len(x_COM))

        # 38.85% From hip joint center to midpoint of the spina iliaca anterior superior (Leva from Zatsiorsky - 1996)
        z_COM = segMHJC[:, 2] + (segMSIAS[:, 2] - segMHJC[:, 2]) * 0.3885

        # Combine the x, y and z coordinates of the center of mass
        segCOM = np.array([x_COM, y_COM, z_COM])

        # Calculate the biomechanical length of the pelvis segment (segment length correction)
        # Omphalion to the hip segmentation planes intersection, female: 256.8, male: 251.7 (biomechanical length
        # Omphalion to the midpoint of both hip joint centers, female: 181.5, male: 145.7 (alternative length)
        seg_length = seg_length * (251.7 / 145.7)

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([3.60, 10.90, 0.76, 8.92])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    elif gender == 'female':
        # Determination of the center of mass of the segment (COM)

        # 63% From anterior to posterior (Plagenhoef - 1983)
        x_COM = segMSIPS[:, 0] * 0.63

        # Create pandas series containing zeros
        y_COM = np.zeros(len(x_COM))

        # 50.80% From hip joint center to midpoint of the spina iliaca anterior superior (Leva from Zatsiorsky - 1996)
        z_COM = segMHJC[:, 2] + (segMSIAS[:, 2] - segMHJC[:, 2]) * 0.5080

        # Combine the x, y and z coordinates of the center of mass
        segCOM = np.array([x_COM, y_COM, z_COM])

        # Calculate the biomechanical length of the pelvis segment (segment length correction)
        # Omphalion to the hip segmentation planes intersection, female: 256.8, male: 251.7 (biomechanical length
        # Omphalion to the midpoint of both hip joint centers, female: 181.5, male: 145.7 (alternative length)
        seg_length = seg_length * (256.8 / 181.5)

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([3.43, 9.37, 0.74, 8.07])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    # Convert the center of mass defined in the local coordination system of the pelvis to global
    COM = np.transpose(np.array([np.dot(gRseg[index], segCOM[:, index]) for index in range(len(gRseg))]))

    """Check inertial parameters with average data of Zatsiorsky"""
    inertial_parameters_zat = np.transpose([8.16, 656, 592, 525])

    # Displays the inertial parameters for a segment and the relation with the population average
    # according to data derived from population (e.g. Zatsiorsky)
    inertial_ratios = inertial_parameters_ratio(inertial_parameters, inertial_parameters_zat, segment='pelvis')

    """Construction of the inertial tensor according to Zatsiorsky"""

    # Construction of the inertial tensor (inertial_parameters_sub[mass, inertia_x, inertia_y, inertia_z]
    segIzat = np.array([[float(inertial_parameters[1]), 0, 0],
                        [0, float(inertial_parameters[2]), 0],
                        [0, 0, float(inertial_parameters[3])]])  # Inertial tensor in kg * cm^2

    # Convert inertial tensor to correct unit order
    segIzat = segIzat / (100 ** 2)  # Inertial tensor in kg * m^2

    # Post multiplication with rotation matrix (Zatsiorsky, Human Kinetics, page 286)
    gIzat = [np.dot(np.dot(gRseg[index], segIzat), np.transpose(gRseg[index])) for index in range(len(gRseg))]
    # gIzat = gRseg * segIzat * gRseg'

    """Determination of the segment acceleration"""

    # Determine segment acceleration by calculating the second derivative of the COM position data
    COM_velocity = calc_derivative(COM, sample_freq)
    COM_acceleration = calc_derivative(COM_velocity, sample_freq)

    """Determination of the segment angular velocity"""
    g_avSeg, avSeg = calc_omega(gRseg, sample_freq)
    g_alfaSeg = calc_derivative(g_avSeg, sample_freq)
    alfaSeg = calc_derivative(avSeg, sample_freq)

    """Determination of Euclidean norm angular velocity and acceleration"""
    norm_av = np.linalg.norm(np.rad2deg(g_avSeg), axis=0)  # Angular velocities converted to degrees/seconds
    norm_acceleration = np.linalg.norm(COM_acceleration, axis=0)

    """Emerge calculated parameters in one dictionary"""

    # Initialise dictionary
    dictionary = dict([])

    # Select all the calculated parameters to combine in one dictionary
    dictionary['seg_name'] = 'pelvis'
    dictionary['mSeg'] = inertial_parameters[0]  # Mass of the segment
    dictionary['COM'] = COM  # Center of mass in meter
    dictionary['Origin'] = np.transpose(origin)  # Origin of the local coordination system
    dictionary['vSeg'] = COM_velocity  # Center of mass velocity in meter/seconds
    dictionary['aSeg'] = COM_acceleration  # Center of mass acceleration in meter/seconds2
    dictionary['aSegNorm'] = norm_acceleration  # Euclidean norm of the segment acceleration in the global coordination system in degrees/seconds
    dictionary['g_avSeg'] = g_avSeg  # Segment angular velocity in the global coordination system in rad/seconds
    dictionary['g_alfaSeg'] = g_alfaSeg  # Segment angular acceleration in the global coordination system in rad/seconds2
    dictionary['avSeg'] = avSeg  # Segment angular velocity in the local coordination system of the segment rad/seconds
    dictionary['avSegNorm'] = norm_av  # Euclidean norm of the segment angular velocity in the global coordination system in meter/seconds2
    dictionary['alfaSeg'] = alfaSeg  # Segment angular acceleration in the local coordination system
    dictionary['gRseg'] = gRseg  # Rotationmatrix
    dictionary['gTseg'] = gTseg  # Transformationmatrix
    dictionary['gIzat'] = gIzat  # Global inertial tensor
    dictionary['segIzat'] = segIzat  # Local inertial tensor
    dictionary['JC'] = JC  # Right and left hip joint center
    dictionary['Iprincipal'] = inertial_parameters[1:4]  # Principal moments of inertia

    return dictionary


def calc_upperarm(LHE, MHE, AC, side='right', sample_freq=[], circumference=29, gender='male', mean_seg_length = []):
    """ Calculates the local coordination system of the upperarm segment according to the ISB definition through
    bony land marks on a right-handed coordination system. The center of mass, center of mass origin, and
    inertial tensor are calculated according to the Zatsiorsky regression equations.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-26)

    Arguments:
        dataframe[samples, X:Y:Z] in meters
        LHE: epicondylus lateralis humeralis
        MHE: epicondylus medialis humeralis
        AC: acromion
        side: arm side 'right' or 'left' (standard: 'right')
        circumference: circumference of the pelvis (standard: 29 cm)
        gender: 'male' (standard) or 'female'
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: center of mass (com) in dataframe[samples, X:Y:Z]
                    center of mass origin (comO) in dataframe[samples, X:Y:Z]
                    rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
                    thorax segment mass (kg) and inertial tensor (gIzat; kg * m^2) in dataframe [mass:X:Y:Z]
    """

    # Convert pandas dataframes to numpy arrays for speed optimisation
    LHE = np.array(LHE)
    MHE = np.array(MHE)
    AC = np.array(AC)

    """Calculation of the shoulder and elbow joint center"""
    EJC = (LHE + MHE) * 0.5
    SJC = AC + (EJC - AC) * (10.4 / (100 - 4.3))  # Longitudinal joint center position (de Leva - 1996)

    """Longitudinal joint center correction"""

    # The extra distance from the acromion to the acromion cluster introduces error in the estimation
    # of the shoulder joint center and the segment length of the upper arm

    # Calculation of the upper arm segment length
    seg_length_raw = np.nanmean([np.linalg.norm(SJC[index, :] - EJC[index, :]) for index in range(len(EJC))]) * 100  # Conversion from m to cm
    seg_length = seg_length_raw

    """Definition of the ISB coordination system through bony land marks on a right-handed coordination system"""

    # Definition of the ISB coordination system through bony land marks on a right-handed coordination system
    origin = SJC  # Origin coincident with midpoint of the lateral and medial humeral epicondylus

    # Initialise variable
    QFP = np.array([])

    # Longitudinal axis pointing proximally from the ulnar styloid to elbow joint center (first axis)
    y_axis = SJC - EJC
    y_axis_norm = np.array([y_axis[index, :] / np.linalg.norm(y_axis[index, :]) for index in range(len(y_axis))])

    # Temporary transversal axis pointing to the right
    z_axis_temp = np.array([])
    # z_axis_temp = MHE - LHE

    if side == 'right':
        z_axis_temp = LHE - MHE
    elif side == 'left':
        z_axis_temp = MHE - LHE

    # Sagittal axis pointing forward; axis perpendicular to the plane formed by ulnar styloid, medial and lateral humeral epicondyle (second axis)
    x_axis = np.cross(y_axis_norm, z_axis_temp)
    x_axis_norm = np.array([x_axis[index, :] / np.linalg.norm(x_axis[index, :]) for index in range(len(x_axis))])

    # Transversal axis pointing the right (third axis)
    z_axis = np.cross(x_axis_norm, y_axis_norm)
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Compose rotationmatrix
    gRseg = [np.transpose(np.array((x_axis_norm[index, :], y_axis_norm[index, :], z_axis_norm[index, :]))) for index in
             range(len(z_axis_norm))]

    # Compose transformationmatrix
    gTseg = copy.deepcopy(gRseg)  # Initialise list for the transformation matrix

    for index in range(len(gRseg)):
        gTseg[index] = np.insert(gTseg[index], 3, origin[index, 0:3], axis=1)
        gTseg[index] = np.insert(gTseg[index], 3, np.array([0, 0, 0, 1]), axis=0)

    """Inertial parameters according to the Zatsiorsky regression equations"""

    # Inertial parameters are calculated according to the Zatsiorsky regression equations

    # Initialisation inertial_parameters_sub variable
    inertial_parameters = np.array([])
    COM = np.array([])

    if gender == 'male':
        # Determination of the center of mass of the segment (COM)

        # 57.72% (Leva from Zatsiorsky - 1996)
        COM = np.transpose(SJC + (EJC - SJC) * 0.5772)

        # Calculate the biomechanical length of the upper arm segment (segment length correction)
        # Acromion to the radius, female: 235.9, male: 244.8 (biomechanical length; measured in 90 degrees abduction position)
        # Shoulder joint center to the elbow joint centers, female: 275.1, male: 281.7 (alternative length)
        seg_length = seg_length * (244.8 / 281.7) # zat = 244.8, leva =

        if mean_seg_length:
            seg_length = mean_seg_length


        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([9.67, 10.81, 2.06, 9.71])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    elif gender == 'female':
        # Determination of the center of mass of the segment (COM)

        # 57.54% (Leva from Zatsiorsky - 1996)
        COM = np.transpose(SJC + (EJC - SJC) * 0.5754)

        # Calculate the biomechanical length of the upper arm segment (segment length correction)
        # Acromion to the radius, female: 235.9, male: 244.8 (biomechanical length; measured in 90 degrees abduction position)
        # Shoulder joint center to the elbow joint centers, female: 275.1, male: 281.7 (alternative length)
        seg_length = seg_length * (235.9 / 275.1)

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([9.49, 10.50, 2.34, 9.18])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    """Check inertial parameters with average data of Zatsiorsky"""
    inertial_parameters_zat = np.transpose([1.98, 127, 38, 114])

    # Displays the inertial parameters for a segment and the relation with the population average
    # according to data derived from population (e.g. Zatsiorsky)
    inertial_ratios = inertial_parameters_ratio(inertial_parameters, inertial_parameters_zat, segment='upper arm')

    """Construction of the inertial tensor according to Zatsiorsky"""

    # Construction of the inertial tensor (inertial_parameters_sub[mass, inertia_x, inertia_y, inertia_z]
    segIzat = np.array([[float(inertial_parameters[1]), 0, 0],
                        [0, float(inertial_parameters[2]), 0],
                        [0, 0, float(inertial_parameters[3])]])  # Inertial tensor in kg * cm^2

    # Convert inertial tensor to correct unit order
    segIzat = segIzat / (100 ** 2)  # Inertial tensor in kg * m^2

    # Post multiplication with rotation matrix (Zatsiorsky, Human Kinetics, page 286)
    gIzat = [np.dot(np.dot(gRseg[index], segIzat), np.transpose(gRseg[index])) for index in range(len(gRseg))]
    # gIzat = gRseg * segIzat * gRseg'

    """Emerge calculated joint centers"""
    # Initialise dictionary
    JC = dict([])

    # Emerge both dataframes in one dictionary
    JC['EJC'] = np.transpose(EJC)
    JC['SJC'] = np.transpose(SJC)

    """Determination of the segment acceleration"""

    # Determine segment acceleration by calculating the second derivative of the COM position data
    COM_velocity = calc_derivative(COM, sample_freq)
    COM_acceleration = calc_derivative(COM_velocity, sample_freq)

    """Determination of the segment angular velocity"""
    g_avSeg, avSeg = calc_omega(gRseg, sample_freq)
    g_alfaSeg = calc_derivative(g_avSeg, sample_freq)
    alfaSeg = calc_derivative(avSeg, sample_freq)

    """Determination of Euclidean norm angular velocity and acceleration"""
    norm_av = np.linalg.norm(np.rad2deg(g_avSeg), axis=0)  # Angular velocities converted to degrees/seconds
    norm_acceleration = np.linalg.norm(COM_acceleration, axis=0)

    """Emerge calculated parameters in one dictionary"""

    # Initialise dictionary
    dictionary = dict([])

    # Select all the calculated parameters to combine in one dictionary
    dictionary['seg_name'] = 'upperarm'
    dictionary['mSeg'] = inertial_parameters[0]  # Mass of the segment
    dictionary['COM'] = COM  # Center of mass in meter
    dictionary['Origin'] = np.transpose(origin)  # Origin of the local coordination system
    dictionary['vSeg'] = COM_velocity  # Center of mass velocity in meter/seconds
    dictionary['aSeg'] = COM_acceleration  # Center of mass acceleration in meter/seconds2
    dictionary['aSegNorm'] = norm_acceleration  # Euclidean norm of the segment acceleration in the global coordination system in degrees/seconds
    dictionary['g_avSeg'] = g_avSeg  # Segment angular velocity in the global coordination system in rad/seconds
    dictionary['g_alfaSeg'] = g_alfaSeg  # Segment angular acceleration in the global coordination system in rad/seconds2
    dictionary['avSeg'] = avSeg  # Segment angular velocity in the local coordination system of the segment rad/seconds
    dictionary['avSegNorm'] = norm_av  # Euclidean norm of the segment angular velocity in the global coordination system in meter/seconds2
    dictionary['alfaSeg'] = alfaSeg  # Segment angular acceleration in the local coordination system
    dictionary['gRseg'] = gRseg  # Rotationmatrix
    dictionary['gTseg'] = gTseg  # Transformationmatrix
    dictionary['gIzat'] = gIzat  # Global inertial tensor
    dictionary['segIzat'] = segIzat  # Local inertial tensor
    dictionary['JC'] = JC  # Shoulder and elbow joint center
    dictionary['Iprincipal'] = inertial_parameters[1:4]  # Principal moments of inertia
    dictionary['seg_length'] = seg_length

    return dictionary


def calc_forearm(LHE, MHE, US, RS, side='right', sample_freq=[], circumference=26, gender='male', mean_seg_length = []):
    """ Calculates the local coordination system of the forearm segment according to the ISB definition through
    bony land marks on a right-handed coordination system. The center of mass, center of mass origin, and
    inertial tensor are calculated according to the Zatsiorsky regression equations.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-27)

    Arguments:
        dataframe[samples, X:Y:Z] in meters
        LHE: epicondylus lateralis humeralis
        MHE: epicondylus medialis humeralis
        US: processus styloideus ulnae
        RS: processus styloideus radii
        side: arm side 'right' or 'left' (standard: 'right')
        circumference: circumference of the pelvis (standard: 26 cm)
        gender: 'male' (standard) or 'female'
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: center of mass (com) in dataframe[samples, X:Y:Z]
                    center of mass origin (comO) in dataframe[samples, X:Y:Z]
                    rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
                    thorax segment mass (kg) and inertial tensor (gIzat; kg * m^2) in dataframe [mass:X:Y:Z]
    """

    # Convert pandas dataframes to numpy arrays for speed optimisation
    LHE = np.array(LHE)
    MHE = np.array(MHE)
    US = np.array(US)
    RS = np.array(RS)

    """Calculation of the elbow and wrist joint center"""
    EJC = (LHE + MHE) * 0.5
    WJC = (US + RS) * 0.5

    """Definition of the ISB coordination system through bony land marks on a right-handed coordination system"""

    # Definition of the ISB coordination system through bony land marks on a right-handed coordination system (Ge Wu et al.)
    origin = US  # Origin coincident with ulnar styloid

    # Longitudinal axis pointing proximally from the ulnar styloid to elbow joint center (first axis)
    y_axis = EJC - US
    y_axis_norm = np.array([y_axis[index, :] / np.linalg.norm(y_axis[index, :]) for index in range(len(y_axis))])

    # Temporary transversal axis pointing to the right
    z_axis_temp = np.array([])
    # z_axis_temp = LHE - MHE

    if side == 'right':
        z_axis_temp = LHE - MHE
    elif side == 'left':
        z_axis_temp = MHE - LHE

    # Sagittal axis pointing forward; axis perpendicular to the plane formed by ulnar styloid, medial and lateral humeral epicondyle (second axis)
    x_axis = np.cross(y_axis_norm, z_axis_temp)
    x_axis_norm = np.array([x_axis[index, :] / np.linalg.norm(x_axis[index, :]) for index in range(len(x_axis))])

    # Transversal axis pointing the right (third axis)
    z_axis = np.cross(x_axis_norm, y_axis_norm)
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Compose rotationmatrix
    gRseg = [np.transpose(np.array((x_axis_norm[index, :], y_axis_norm[index, :], z_axis_norm[index, :]))) for index in
             range(len(z_axis_norm))]

    # Compose transformationmatrix
    gTseg = copy.deepcopy(gRseg)  # Initialise list for the transformation matrix

    for index in range(len(gRseg)):
        gTseg[index] = np.insert(gTseg[index], 3, origin[index, 0:3], axis=1)
        gTseg[index] = np.insert(gTseg[index], 3, np.array([0, 0, 0, 1]), axis=0)

    """Inertial parameters according to the Zatsiorsky regression equations"""

    # Inertial parameters are calculated according to the Zatsiorsky regression equations

    # Calculation of the upper arm segment length
    seg_length = np.nanmean([np.linalg.norm(EJC[index, :] - WJC[index, :]) for index in range(len(WJC))]) * 100  # Conversion from m to cm

    # Initialisation inertial_parameters_sub variable
    inertial_parameters = np.array([])
    COM = np.array([])

    if gender == 'male':
        # Determination of the center of mass of the segment (COM)

        # 45.74% (Leva from Zatsiorsky - 1996) - The transpose is used to convert to standing vector notations
        COM = np.transpose(EJC + (US - EJC) * 0.4574)  # Originally WJC is used instead of the US; US chosen due to major fluctuations in position of the WJC

        # Calculate the biomechanical length of the forearm segment (segment length correction)
        # Radius to the processus styloideus radii, female: 247.1, male: 251.3 (biomechanical length; measured in 90 degrees abduction position)
        # Elbow joint center to the wrist joint center, female: 264.3, male: 268.9 (alternative length)
        seg_length = seg_length * (251.3 / 268.9)

        if mean_seg_length:
            seg_length = mean_seg_length

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.array([6.26, 7.55, 1.51, 7.03])  # The lying vector notation is not problematic due to selection of individual scalars

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    elif gender == 'female':
        # Determination of the center of mass of the segment (COM)

        # 45.59% (Leva from Zatsiorsky - 1996) - The transpose is used to convert to standing vector notations
        COM = np.transpose(EJC + (US - EJC) * 0.4559)  # Originally WJC is used instead of the US; US chosen due to major fluctuations in position of the WJC

        # Calculate the biomechanical length of the forearm segment (segment length correction)
        # Radius to the processus styloideus radii, female: 247.1, male: 251.3 (biomechanical length; measured in 90 degrees abduction position)
        # Elbow joint center to the wrist joint center, female: 264.3, male: 268.9 (alternative length)
        seg_length = seg_length * (247.1 / 264.3)

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.array([6.43, 7.81, 1.14, 7.95])  # The lying vector notation is not problematic due to selection of individual scalars

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    """Check inertial parameters with average data of Zatsiorsky"""
    inertial_parameters_zat = np.transpose([1.17, 64, 12, 60])

    # Displays the inertial parameters for a segment and the relation with the population average
    # according to data derived from population (e.g. Zatsiorsky)
    inertial_ratios = inertial_parameters_ratio(inertial_parameters, inertial_parameters_zat, segment='forearm')

    """Construction of the inertial tensor according to Zatsiorsky"""

    # Construction of the inertial tensor (inertial_parameters_sub[mass, inertia_x, inertia_y, inertia_z]
    segIzat = np.array([[float(inertial_parameters[1]), 0, 0],
                        [0, float(inertial_parameters[2]), 0],
                        [0, 0, float(inertial_parameters[3])]])  # Inertial tensor in kg * cm^2

    # Convert inertial tensor to correct unit order
    segIzat = segIzat / (100 ** 2)  # Inertial tensor from km*cm^2 to kg * m^2 (divide by 100^2)

    # Post multiplication with rotation matrix (Zatsiorsky, Human Kinetics, page 286)
    gIzat = [np.dot(np.dot(gRseg[index], segIzat), np.transpose(gRseg[index])) for index in range(len(gRseg))]
    # gIzat = gRseg * segIzat * gRseg'

    """Emerge calculated joint centers"""
    # Initialise dictionary
    JC = dict([])

    # Emerge both dataframes in one dictionary
    JC['WJC'] = np.transpose(WJC)
    JC['EJC'] = np.transpose(EJC)

    """Determination of the segment acceleration"""

    # Determine segment acceleration by calculating the second derivative of the COM position data
    COM_velocity = calc_derivative(COM, sample_freq)
    COM_acceleration = calc_derivative(COM_velocity, sample_freq)

    """Determination of the segment angular velocity"""
    g_avSeg, avSeg = calc_omega(gRseg, sample_freq)
    g_alfaSeg = calc_derivative(g_avSeg, sample_freq)
    alfaSeg = calc_derivative(avSeg, sample_freq)

    """Determination of Euclidean norm angular velocity and acceleration"""
    norm_av = np.linalg.norm(np.rad2deg(g_avSeg), axis=0)  # Angular velocities converted to degrees/seconds
    norm_acceleration = np.linalg.norm(COM_acceleration, axis=0)

    """Emerge calculated parameters in one dictionary"""

    # Initialise dictionary
    dictionary = dict([])

    # Select all the calculated parameters to combine in one dictionary
    dictionary['seg_name'] = 'forearm'
    dictionary['mSeg'] = inertial_parameters[0]  # Mass of the segment
    dictionary['COM'] = COM  # Center of mass in meter
    dictionary['Origin'] = np.transpose(origin) # Origin of the local coordination system
    dictionary['vSeg'] = COM_velocity  # Center of mass velocity in meter/seconds
    dictionary['aSeg'] = COM_acceleration  # Center of mass acceleration in meter/seconds2
    dictionary['aSegNorm'] = norm_acceleration  # Euclidean norm of the segment acceleration in the global coordination system in degrees/seconds
    dictionary['g_avSeg'] = g_avSeg  # Segment angular velocity in the global coordination system in rad/seconds
    dictionary['g_alfaSeg'] = g_alfaSeg  # Segment angular acceleration in the global coordination system in rad/seconds2
    dictionary['avSeg'] = avSeg  # Segment angular velocity in the local coordination system of the segment rad/seconds
    dictionary['avSegNorm'] = norm_av  # Euclidean norm of the segment angular velocity in the global coordination system in meter/seconds2
    dictionary['alfaSeg'] = alfaSeg  # Segment angular acceleration in the local coordination system
    dictionary['gRseg'] = gRseg  # Rotationmatrix
    dictionary['gTseg'] = gTseg  # Transformationmatrix
    dictionary['gIzat'] = gIzat  # Global inertial tensor
    dictionary['segIzat'] = segIzat  # Local inertial tensor
    dictionary['JC'] = JC  # Elbow and wrist joint center
    dictionary['Iprincipal'] = inertial_parameters[1:4]  # Principal moments of inertia
    dictionary['seg_length'] = seg_length

    return dictionary


def calc_hand(US, RS, MH3, side='right', sample_freq=[], circumference=21, gender='male', mean_seg_length = []):
    """ Calculates the local coordination system of the hand segment according to the ISB definition through
    bony land marks on a right-handed coordination system. The center of mass, center of mass origin, and
    inertial tensor are calculated according to the Zatsiorsky regression equations.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-12-08)

    Arguments:
        dataframe[samples, X:Y:Z] in meters
        US: processus styloideus ulnae
        RS: processus styloideus radii
        MH3: os metacarpalis of the hand 3
        side: arm side 'right' or 'left' (standard: 'right')
        circumference: circumference of the pelvis (standard: 26 cm)
        gender: 'male' (standard) or 'female'
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: center of mass (com) in dataframe[samples, X:Y:Z]
                    center of mass origin (comO) in dataframe[samples, X:Y:Z]
                    rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
                    thorax segment mass (kg) and inertial tensor (gIzat; kg * m^2) in dataframe [mass:X:Y:Z]
    """

    # Convert pandas dataframes to numpy arrays for speed optimisation
    US = np.array(US)
    RS = np.array(RS)
    MH3 = np.array(MH3)

    """Calculation of the elbow and wrist joint center"""
    WJC = (US + RS) * 0.5

    """Definition of the ISB coordination system through bony land marks on a right-handed coordination system"""

    # Definition of the ISB coordination system through bony land marks on a right-handed coordination system (Ge Wu et al.)
    origin = WJC  # Origin coincident with wrist joint center

    # Temporary z_axis based on the ulnar styloid and radial styloid dependent of chosen side
    z_axis_temp = np.array([])
    # z_axis_temp = RS - WJC

    if side == "right":
        z_axis_temp = RS - WJC
    elif side == "left":
        z_axis_temp = WJC - RS

    # Longitudinal axis pointing proximally from os metacarpal three to the wrist joint center (first axis)
    y_axis = WJC - MH3
    y_axis_norm = np.array([y_axis[index, :] / np.linalg.norm(y_axis[index, :]) for index in range(len(y_axis))])

    # Sagittal axis pointing forward; axis perpendicular to the plane formed by ulnar styloid, radial styloid, os metacarpal three and the wrist joint center (second axis)
    x_axis = np.cross(y_axis, z_axis_temp)
    x_axis_norm = np.array([x_axis[index, :] / np.linalg.norm(x_axis[index, :]) for index in range(len(x_axis))])

    # Transversal axis pointing to the right (third axis)
    z_axis = np.cross(x_axis_norm, y_axis_norm)
    z_axis_norm = np.array([z_axis[index, :] / np.linalg.norm(z_axis[index, :]) for index in range(len(z_axis))])

    # Compose rotationmatrix
    gRseg = [np.transpose(np.array((x_axis_norm[index, :], y_axis_norm[index, :], z_axis_norm[index, :]))) for index in
             range(len(z_axis_norm))]

    # Compose transformationmatrix
    gTseg = copy.deepcopy(gRseg)  # Initialise list for the transformation matrix

    for index in range(len(gRseg)):
        gTseg[index] = np.insert(gTseg[index], 3, origin[index, 0:3], axis=1)
        gTseg[index] = np.insert(gTseg[index], 3, np.array([0, 0, 0, 1]), axis=0)

    """Inertial parameters according to the Zatsiorsky regression equations"""

    # Inertial parameters are calculated according to the Zatsiorsky regression equations

    # Calculation of the hand segment length
    seg_length = np.nanmean([np.linalg.norm(MH3[index, :] - WJC[index, :]) for index in range(len(WJC))]) * 100  # Conversion from m to cm

    # Initialisation inertial_parameters_sub variable
    inertial_parameters = np.array([])
    COM = np.array([])

    if gender == 'male':
        # Determination of the center of mass of the segment (COM)

        # 79.00% (Leva from Zatsiorsky - 1996)
        COM = np.transpose(WJC + (MH3 - WJC) * 0.7900)

        # Fingertip distance (Leva from Zatsiorsky - 1996)
        FT3 = np.transpose(WJC + (MH3 - WJC) * (187.9 / 79.00))

        # Calculate the biomechanical length of the hand segment (segment length correction)
        # Radius styloid to the third dactilon, female: 172.0, male: 189.9 (biomechanical length; measured in 90 degrees abduction position)
        # Wrist joint center to the third dactilon, female: 170.1, male: 187.9 (alternative length)
        # Writs joint center to the third os metacarpalis, female: 74.74, male: 79.00 (alternative length)
        seg_length = seg_length * (189.9 / 79.00)

        if mean_seg_length:
            seg_length = mean_seg_length

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([5.54, 6.65, 2.29, 4.86])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    elif gender == 'female':
        # Determination of the center of mass of the segment (COM)

        # 74.74% (Leva from Zatsiorsky - 1996)
        COM = np.transpose(WJC + (MH3 - WJC) * 0.7474)

        # Fingertip distance (Leva from Zatsiorsky - 1996)
        FT3 = np.transpose(WJC + (MH3 - WJC) * (172.0 / 74.74))

        # Calculate the biomechanical length of the hand segment (segment length correction)
        # Radius styloid to the third dactilon, female: 172.0, male: 189.9 (biomechanical length; measured in 90 degrees abduction position)
        # Wrist joint center to the third dactilon, female: 170.1, male: 187.9 (alternative length)
        # Writs joint center to the third os metacarpalis, female: 74.74, male: 79.00 (alternative length)
        seg_length = seg_length * (172.0 / 74.74)

        # Regression parameters for inertial parameters calculations obtained from Zatsiorsky
        reg_parameters = np.transpose([5.56, 5.85, 1.58, 4.32])

        # Calculate the thorax inertial parameters (mass, inertia_x, inertia_y, inertia_z)
        inertial_parameters = calc_inertial_parameters(reg_parameters, seg_length, circumference)

    """Check inertial parameters with average data of Zatsiorsky"""
    inertial_parameters_zat = np.transpose([0.447, 13.2, 5.37, 8.76])

    # Displays the inertial parameters for a segment and the relation with the population average
    # according to data derived from population (e.g. Zatsiorsky)
    inertial_ratios = inertial_parameters_ratio(inertial_parameters, inertial_parameters_zat, segment='hand')

    """Construction of the inertial tensor according to Zatsiorsky"""

    # Construction of the inertial tensor (inertial_parameters_sub[mass, inertia_x, inertia_y, inertia_z]
    segIzat = np.array([[float(inertial_parameters[1]), 0, 0],
                        [0, float(inertial_parameters[2]), 0],
                        [0, 0, float(inertial_parameters[3])]])  # Inertial tensor in kg * cm^2

    # Convert inertial tensor to correct unit order
    segIzat = segIzat / (100 ** 2)  # Inertial tensor in kg * m^2

    # Post multiplication with rotation matrix (Zatsiorsky, Human Kinetics, page 286)
    gIzat = [np.dot(np.dot(gRseg[index], segIzat), np.transpose(gRseg[index])) for index in range(len(gRseg))]
    # gIzat = gRseg * segIzat * gRseg'

    """Emerge calculated joint centers"""
    # Initialise dictionary
    JC = dict([])

    # Emerge both dataframes in one dictionary
    JC['FT3'] = FT3
    JC['WJC'] = np.transpose(WJC)

    """Determination of the segment acceleration"""

    # Determine segment acceleration by calculating the second derivative of the COM position data
    COM_velocity = calc_derivative(COM, sample_freq)
    COM_acceleration = calc_derivative(COM_velocity, sample_freq)

    """Determination of the segment angular velocity"""
    g_avSeg, avSeg = calc_omega(gRseg, sample_freq)
    g_alfaSeg = calc_derivative(g_avSeg, sample_freq)
    alfaSeg = calc_derivative(avSeg, sample_freq)

    """Determination of Euclidean norm angular velocity and acceleration"""
    norm_av = np.linalg.norm(np.rad2deg(g_avSeg), axis=0)  # Angular velocities converted to degrees/seconds
    norm_acceleration = np.linalg.norm(COM_acceleration, axis=0)

    """Emerge calculated parameters in one dictionary"""

    # Initialise dictionary
    dictionary = dict([])

    # Select all the calculated parameters to combine in one dictionary
    dictionary['seg_name'] = 'hand'
    dictionary['mSeg'] = inertial_parameters[0]  # Mass of the segment
    dictionary['COM'] = COM  # Center of mass in meter
    dictionary['Origin'] = np.transpose(origin)  # Origin of the local coordination system
    dictionary['vSeg'] = COM_velocity  # Center of mass velocity in meter/seconds
    dictionary['aSeg'] = COM_acceleration  # Center of mass acceleration in meter/seconds2
    dictionary['aSegNorm'] = norm_acceleration  # Euclidean norm of the segment acceleration in the global coordination system in degrees/seconds
    dictionary['g_avSeg'] = g_avSeg  # Segment angular velocity in the global coordination system in rad/seconds
    dictionary['g_alfaSeg'] = g_alfaSeg  # Segment angular acceleration in the global coordination system in rad/seconds2
    dictionary['alfaSeg'] = alfaSeg  # Segment angular acceleration in the local coordination system
    dictionary['avSeg'] = avSeg  # Segment angular velocity in the local coordination system of the segment rad/seconds
    dictionary['avSegNorm'] = norm_av  # Euclidean norm of the segment angular velocity in the global coordination system in meter/seconds2
    dictionary['gRseg'] = gRseg  # Rotationmatrix
    dictionary['gTseg'] = gTseg  # Transformationmatrix
    dictionary['gIzat'] = gIzat  # Global inertial tensor
    dictionary['segIzat'] = segIzat  # Local inertial tensor
    dictionary['JC'] = JC  # Wrist joint center
    dictionary['Iprincipal'] = inertial_parameters[1:4]  # Principal moments of inertia
    dictionary['seg_length'] = seg_length

    return dictionary


def calc_tech_system(M1, M2, M3, sample_freq=[]):
    """ Calculates the local coordination system of a segment or cluster through
    bony land marks on a right-handed coordination system.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2021-01-20)

    Arguments:
        dataframe[samples, X:Y:Z] in meters
        M1: processus styloideus ulnae
        M2: processus styloideus radii
        M3: os metacarpalis of the hand 3
        sample_freq: sample frequency of the position data measured with motion capture system in Hz
    Returns:
        dictionary: rotation matrix/transformation (gRseg/gTseg) in list[:, dataframes[3 or 4, 3 or 4]]
    """
    # Convert pandas dataframes to numpy arrays for speed optimisation
    M1 = np.array(M1)
    M2 = np.array(M2)
    M3 = np.array(M3)

    # Calculate the mean of the x, y and z position coordinates of the cluster markers
    mean_x = np.nanmean([M1[:, 0], M2[:, 0], M3[:, 0]], 0)
    mean_y = np.nanmean([M1[:, 1], M2[:, 1], M3[:, 1]], 0)
    mean_z = np.nanmean([M1[:, 2], M2[:, 2], M3[:, 2]], 0)

    # Define the center of the cluster as origin
    origin = np.array([mean_x, mean_y, mean_z])


    return


""" Functions to describe rotation and transformation matrices of each segment relative to reference position"""


def calibrate2reference(seg_motion, seg_reference, ref_frame):
    """ Calculates the transformation matrix for each segment, describing the rotation and translation
    of each segment relative to the reference position.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-05-29)

    Arguments:
        seg_motion: transformation or rotation matrix of the segment motion in list[:, dataframes[3 or 4, 3 or 4]]
        seg_reference: transformation or rotation matrix (ONE sample or frame) of the segment reference position in list[0, dataframes[3 or 4, 3 or 4]]
        ref_frame: reference frame [scalar]
    Returns:
        referenced_motion: transformation or rotation matrix relative to the reference position in list[:, dataframes[3 or 4, 3 or 4]]
    """

    # --- gRseg --- #
    # Select the gRseg from motion and reference data
    gRseg_motion = seg_motion['gRseg']
    gRseg_reference = seg_reference['gRseg'][ref_frame]

    # Post multiplication with inverse of the reference orientation
    gRseg_referenced = [gRseg_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                        range(len(gRseg_motion))]  # (Zatsiorsky, Human Kinetics, page 286)

    # --- gTseg --- #
    # Select the gTseg from motion and reference data
    gTseg_motion = seg_motion['gTseg']
    gTseg_reference = seg_reference['gTseg'][ref_frame]

    # Post multiplication with inverse of the reference orientation
    gTseg_referenced = [gTseg_motion[index].dot(np.linalg.inv(gTseg_reference)) for index in
                        range(len(gTseg_motion))]  # (Zatsiorsky, Human Kinetics, page 286)

    # --- COM --- #
    # Select the COM from motion and reference data
    COM_motion = seg_motion['COM']

    # Post multiplication with inverse of the reference orientation
    COM_referenced = np.array([COM_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                               range(len(COM_motion))])  # (Zatsiorsky, Human Kinetics, page 286)

    # --- vSeg --- #
    # Select the vSeg from motion and reference data
    vSeg_motion = seg_motion['vSeg']

    # Post multiplication with inverse of the reference orientation
    vSeg_referenced = np.array([vSeg_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                                range(len(vSeg_motion))])  # (Zatsiorsky, Human Kinetics, page 286)

    # --- aSeg --- #
    # Select the aSeg from motion and reference data
    aSeg_motion = seg_motion['aSeg']

    # Post multiplication with inverse of the reference orientation
    aSeg_referenced = np.array([aSeg_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                                range(len(aSeg_motion))])  # (Zatsiorsky, Human Kinetics, page 286)

    # --- g_avSeg --- #
    # Select the g_avSeg from motion and reference data
    g_avSeg_motion = seg_motion['g_avSeg']

    # Post multiplication with inverse of the reference orientation
    g_avSeg_referenced = np.array([g_avSeg_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                                   range(len(g_avSeg_motion))])  # (Zatsiorsky, Human Kinetics, page 286)

    # --- avSeg --- #
    # Select the avSeg from motion and reference data
    avSeg_motion = seg_motion['avSeg']

    # Post multiplication with inverse of the reference orientation
    avSeg_referenced = np.array([avSeg_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                                 range(len(avSeg_motion))])  # (Zatsiorsky, Human Kinetics, page 286)

    # --- gIzat --- #
    # Select the gIzat from motion and reference data
    gIzat_motion = seg_motion['gIzat']

    # Post multiplication with inverse of the reference orientation
    gIzat_referenced = [gIzat_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                        range(len(gIzat_motion))]  # (Zatsiorsky, Human Kinetics, page 286)

    # --- JC --- #
    # Select the JC from motion and reference data
    JC_referenced = dict()

    if 'JC' in seg_motion:
        for joint in seg_motion['JC']:
            JC_motion = seg_motion['JC'][joint]

            # Post multiplication with inverse of the reference orientation
            JC_referenced[joint] = np.array([JC_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                                             range(len(JC_motion))])  # (Zatsiorsky, Human Kinetics, page 286)
        seg_motion['JC'] = JC_referenced

    # --- g_alfaSeg --- #
    # Select the g_alfaSeg from motion and reference data
    galfaSeg_motion = seg_motion['g_alfaSeg']

    # Post multiplication with inverse of the reference orientation
    galfaSeg_referenced = np.array([galfaSeg_motion[index].dot(np.linalg.inv(gRseg_reference)) for index in
                                    range(len(galfaSeg_motion))])  # (Zatsiorsky, Human Kinetics, page 286)

    # Emerge referenced with dictionary
    seg_motion['COM'] = COM_referenced
    seg_motion['vSeg'] = vSeg_referenced
    seg_motion['aSeg'] = aSeg_referenced
    seg_motion['g_avSeg'] = g_avSeg_referenced
    seg_motion['avSeg'] = avSeg_referenced
    seg_motion['gRseg'] = gRseg_referenced
    seg_motion['gTseg'] = gTseg_referenced
    seg_motion['gIzat'] = gIzat_referenced
    seg_motion['g_alfaSeg'] = galfaSeg_referenced

    return seg_motion


def segments2combine(*segment_dictionaries):
    """ Combines the calculated segment parameters collected in segment dictionaries into one dictionary to be able to
    iterate through the keys for the calculations of net forces and moments.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-08-19)

    Arguments:
        segment_dictionaries: segment dictionaries consisting of the calculated parameters with the segment functions
    Returns:
        combined_dictionary: combined dictionary with the dictionaries consisting the calculated segment parameters
    """

    # Read segments names
    segment_names = [segment_dictionaries[index]['seg_name'] for index in range(len(segment_dictionaries))]

    # Initialise combined_dictionary
    combined_dictionary = dict.fromkeys(segment_names, dict())

    # Combine all segment dictionaries into one combined dictionary
    for index in range(len(segment_dictionaries)):
        combined_dictionary[segment_names[index]] = dict(segment_dictionaries[index])

    return combined_dictionary


def separation_time(model, sample_freq, threshold=1.0, analytical=1):
    """ Calculates the separation time between the peak angular velocity of the all segments in model in a
    proximal-to-distal sequence manner according a numerical or analytical method.

    Function is developed and written by Bart van Trigt, PhD-Candidate Delft Technical University
    Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: b.vantrigt@tudelft.nl, a.j.r.leenen@vu.nl

    Version 1.0 (2021-03-24)
    Version 2.0 (2021-03-31) - Analytical method has been added to the function by Ton Leenen

    Arguments:
        model: combined dictionary with the dictionaries consisting the calculated segment parameters (output of the function 'segments2combine')
        sample_freq: sample frequency of the data measured with motion capture system in Hz
        threshold: relative threshold used to detect mainly the highest peaks (default: 1.0, normal peak height)
        analytical: calculate the angular velocities and separation times based on analytical method (default: 1.0, analytical method)
    Returns:
        separation_time: separation time between the peak angular velocity of all segments in model
        peak_ang_velocity: peak angular velocities of all segments in model
    """

    # --- Initialize variables --- #
    av_pelvis = model['pelvis']['avSegNorm']
    av_thorax = model['thorax']['avSegNorm']
    av_upperarm = model['upperarm']['avSegNorm']
    av_forearm = model['forearm']['avSegNorm']
    av_hand = model['hand']['avSegNorm']

    # --- Calculate relative threshold to detect the peak angular velocities --- #
    threshold_pelvis = threshold * np.nanmax(av_pelvis)
    threshold_thorax = threshold * np.nanmax(av_thorax)
    threshold_upperarm = threshold * np.nanmax(av_upperarm)
    threshold_forearm = threshold * np.nanmax(av_forearm)
    threshold_hand = threshold * np.nanmax(av_hand)

    # --- Find index and magnitude of the peak angular velocities --- #
    pelvis_index, pelvis_peak = sp.find_peaks(av_pelvis, height=threshold_pelvis)
    thorax_index, thorax_peak = sp.find_peaks(av_thorax, height=threshold_thorax)
    upperarm_index, upperarm_peak = sp.find_peaks(av_upperarm, height=threshold_upperarm)
    forearm_index, forearm_peak = sp.find_peaks(av_forearm, height=threshold_forearm)
    hand_index, hand_peak = sp.find_peaks(av_hand, height=threshold_hand)

    # --- First peaks in the Euclidean normalized angular velocities signals are used for calculations --- #

    # --- NUMERICAL METHOD --- #
    if analytical == 0:
        # --- Calculate separation time in proximal-to-distal sequence --- #
        separation_time = (np.array([thorax_index[0] - pelvis_index[0],
                                    upperarm_index[0] - thorax_index[0],
                                    forearm_index[0] - upperarm_index[0],
                                    hand_index[0] - forearm_index[0]]) / sample_freq) * 1000  # Conversion from seconds to milliseconds

        # --- Emerge peak angular velocities used to detected by 'find_peaks' function --- #
        peak_ang_velocity = np.array([pelvis_peak['peak_heights'][0],
                                      thorax_peak['peak_heights'][0],
                                      upperarm_peak['peak_heights'][0],
                                      forearm_peak['peak_heights'][0],
                                      hand_peak['peak_heights'][0]])

    # --- ANALYTICAL METHOD --- #
    elif analytical == 1:
        # --- Fitting 2nd order polynomial function on the data points in the selected window --- #

        # --- Select window based on found indices for each segment --- #
        window_pelvis = np.linspace(pelvis_index[0] - 5, pelvis_index[0] + 5,
                                    ((pelvis_index[0] + 5) - (pelvis_index[0] - 5) + 1), endpoint=True).astype(int)

        window_thorax = np.linspace(thorax_index[0] - 5, thorax_index[0] + 5,
                                    ((thorax_index[0] + 5) - (thorax_index[0] - 5) + 1), endpoint=True).astype(int)

        window_upperarm = np.linspace(upperarm_index[0] - 5, upperarm_index[0] + 5,
                                      ((upperarm_index[0] + 5) - (upperarm_index[0] - 5) + 1), endpoint=True).astype(int)

        window_forearm = np.linspace(forearm_index[0] - 5, forearm_index[0] + 5,
                                     ((forearm_index[0] + 5) - (forearm_index[0] - 5) + 1), endpoint=True).astype(int)

        window_hand = np.linspace(hand_index[0] - 5, hand_index[0] + 5,
                                  ((hand_index[0] + 5) - (hand_index[0] - 5) + 1), endpoint=True).astype(int)

        # --- Construct timeline based on the sample frequency --- #
        timeline = np.arange(av_pelvis.shape[0]) * (1 / sample_freq)

        # --- Calculate the 2nd order polynomial function coefficients for each segment --- #
        fit_pelvis = np.polyfit(window_pelvis, av_pelvis[window_pelvis], 2)
        fit_thorax = np.polyfit(window_thorax, av_thorax[window_thorax], 2)
        fit_upperarm = np.polyfit(window_upperarm, av_upperarm[window_upperarm], 2)
        fit_forearm = np.polyfit(window_forearm, av_forearm[window_forearm], 2)
        fit_hand = np.polyfit(window_hand, av_hand[window_hand], 2)

        # --- Analytical calculation of the exact point in time of the occurrence of the peak angular velocity for each segment --- #
        analytical_pelvis = -(fit_pelvis[1] / (2 * fit_pelvis[0]))
        analytical_thorax = -(fit_thorax[1] / (2 * fit_thorax[0]))
        analytical_upperarm = -(fit_upperarm[1] / (2 * fit_upperarm[0]))
        analytical_forearm = -(fit_forearm[1] / (2 * fit_forearm[0]))
        analytical_hand = -(fit_hand[1] / (2 * fit_hand[0]))

        # --- Calculate separation time in proximal-to-distal sequence --- #
        separation_time = (np.array([analytical_thorax - analytical_pelvis,
                                     analytical_upperarm - analytical_thorax,
                                     analytical_forearm - analytical_upperarm,
                                     analytical_hand - analytical_forearm]) / sample_freq) * 1000  # Conversion from seconds to milliseconds

        # --- Emerge peak angular velocities used to detected by 'find_peaks' function --- #
        peak_ang_velocity = np.array([pelvis_peak['peak_heights'][0],
                                      thorax_peak['peak_heights'][0],
                                      upperarm_peak['peak_heights'][0],
                                      forearm_peak['peak_heights'][0],
                                      hand_peak['peak_heights'][0]])

    return separation_time, peak_ang_velocity, pelvis_index,thorax_index, upperarm_index, forearm_index, hand_index, pelvis_peak, thorax_peak, upperarm_peak, forearm_peak, hand_peak

def separation_time_pp(model, sample_freq, threshold=1.0, analytical=1):
    """ Calculates the separation time between the peak angular velocity of the all segments in model in a
    proximal-to-distal sequence manner according a numerical or analytical method.

    Function is developed and written by Bart van Trigt, PhD-Candidate Delft Technical University
    Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: b.vantrigt@tudelft.nl, a.j.r.leenen@vu.nl

    Version 1.0 (2021-03-24)
    Version 2.0 (2021-03-31) - Analytical method has been added to the function by Ton Leenen

    Arguments:
        model: combined dictionary with the dictionaries consisting the calculated segment parameters (output of the function 'segments2combine')
        sample_freq: sample frequency of the data measured with motion capture system in Hz
        threshold: relative threshold used to detect mainly the highest peaks (default: 1.0, normal peak height)
        analytical: calculate the angular velocities and separation times based on analytical method (default: 1.0, analytical method)
    Returns:
        separation_time: separation time between the peak angular velocity of all segments in model
        peak_ang_velocity: peak angular velocities of all segments in model
    """

    # --- Initialize variables --- #
    av_pelvis = model['pelvis']['avSegNorm']
    av_pelvis = butter_lowpass_filter(av_pelvis,12,120,2)
    av_thorax = model['thorax']['avSegNorm']
    av_thorax = butter_lowpass_filter(av_thorax, 12, 120, 2)

    if not (np.isnan(np.nanmax(av_pelvis)) or np.isnan(np.nanmax(av_thorax))):

        # --- Calculate relative threshold to detect the peak angular velocities --- #
        threshold_pelvis = threshold * np.nanmax(av_pelvis)
        threshold_thorax = threshold * np.nanmax(av_thorax)

        # --- Find index and magnitude of the peak angular velocities --- #
        pelvis_index, pelvis_peak = sp.find_peaks(av_pelvis, height=threshold_pelvis)
        thorax_index, thorax_peak = sp.find_peaks(av_thorax, height=threshold_thorax)

        if (len(pelvis_index) == 0) or (len(thorax_index) == 0):
            pelvis_index = [np.nanargmax(av_pelvis)]
            thorax_index = [np.nanargmax(av_thorax)]
            pelvis_peak = dict()
            pelvis_peak['peak_heights'] = [np.nanmax(av_pelvis)]
            thorax_peak = dict()
            thorax_peak['peak_heights'] = [np.nanmax(av_thorax)]

        # --- First peaks in the Euclidean normalized angular velocities signals are used for calculations --- #

        # --- NUMERICAL METHOD --- #
        if analytical == 0:
            # --- Calculate separation time in proximal-to-distal sequence --- #
            separation_time = (np.array([thorax_index[0] - pelvis_index[0]]) / sample_freq) * 1000  # Conversion from seconds to milliseconds

            # --- Emerge peak angular velocities used to detected by 'find_peaks' function --- #
            peak_ang_velocity = np.array([pelvis_peak['peak_heights'][0],
                                          thorax_peak['peak_heights'][0]])

        # --- ANALYTICAL METHOD --- #
        elif analytical == 1:
            # --- Fitting 2nd order polynomial function on the data points in the selected window --- #

            # --- Select window based on found indices for each segment --- #
            window_pelvis = np.linspace(pelvis_index[0] - 5, pelvis_index[0] + 5,
                                        ((pelvis_index[0] + 5) - (pelvis_index[0] - 5) + 1), endpoint=True).astype(int)

            window_thorax = np.linspace(thorax_index[0] - 5, thorax_index[0] + 5,
                                        ((thorax_index[0] + 5) - (thorax_index[0] - 5) + 1), endpoint=True).astype(int)

            # --- Construct timeline based on the sample frequency --- #
            timeline = np.arange(av_pelvis.shape[0]) * (1 / sample_freq)

            # --- Calculate the 2nd order polynomial function coefficients for each segment --- #
            fit_pelvis = np.polyfit(window_pelvis, av_pelvis[window_pelvis], 2)
            fit_thorax = np.polyfit(window_thorax, av_thorax[window_thorax], 2)

            # --- Analytical calculation of the exact point in time of the occurrence of the peak angular velocity for each segment --- #
            analytical_pelvis = -(fit_pelvis[1] / (2 * fit_pelvis[0]))
            analytical_thorax = -(fit_thorax[1] / (2 * fit_thorax[0]))

            # --- Calculate separation time in proximal-to-distal sequence --- #
            separation_time = (np.array([analytical_thorax - analytical_pelvis,
                                         ]) / sample_freq) * 1000  # Conversion from seconds to milliseconds

            # --- Emerge peak angular velocities used to detected by 'find_peaks' function --- #
            peak_ang_velocity = np.array([pelvis_peak['peak_heights'][0],
                                          thorax_peak['peak_heights'][0]])
    else:
        separation_time = [np.nan]
        peak_ang_velocity = [np.nan]
        pelvis_index= [np.nan]
        thorax_index= [np.nan]
        pelvis_peak = dict()
        pelvis_peak['peak_heights']= [np.nan]
        thorax_peak = dict()
        thorax_peak['peak_heights']= [np.nan]

    return separation_time, peak_ang_velocity, pelvis_index,thorax_index, pelvis_peak, thorax_peak


def calc_net_reaction_force(model, direction='top-down', F_extern=np.array([[0], [0], [0]])):
    """ Calculates the net reaction forces, according to the force balance, for the joint of interest. The input and output
    are forces that act on the segments whose forces are being balanced here. Attention: action = - reaction

    # --- FORCE BALANCE --- #
    Descriptions:
         F_proximal: proximal external forces acting on the distal part of the segment of interest in dataframe[samples, X:Y:Z]
         F_distal: distal external forces acting on the proximal part of the segment of interest in dataframe[samples, X:Y:Z]

    F_proximal + F_distal + F_gravity = (m * a)
                       <->
    F_proximal = (m * a) - F_gravity - F_distal
                       <->
    F_proximal = (m * a) - (m * g) - F_distal

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-12-15)

    Arguments:
        model: combined dictionary with the dictionaries consisting the calculated segment parameters (output of the function 'segments2combine')
        direction: order of the calculation of the net forces and moments 'bottom-up' or 'top-down'
        F_extern: external forces acting on the segment of interest in dataframe[samples, X:Y:Z]
    Returns:
        F_segment: combined dictionary with the dictionaries consisting the calculated segment parameters in dataframe[samples, X:Y:Z]:
        - F_distal: total forces from the distal segment that act on the subsequent proximal segment
        - F_proximal: total forces from the proximal segment that act on the subsequent distal segment
          !!! --- Attention: action = - reaction --- !!! <-> !!! --- F_distal = - F_proximal --- !!!
    """

    # Read number of samples of the measured data
    samples = len(model[next(iter(model))]['gRseg'])  # Read the length of the parameter gRseg of the first segment (based on the first key found)

    # Definition of the gravity vector and repeated for the number of samples
    g = np.transpose(np.array([[0, 0, -9.80665]] * samples))  # The conventional exact standard value of the gravity is in our coordination system defined as a negative value

    # Read number of segments from model
    segments = list(model.keys())

    # Initialise dictionary F_segment
    F_segment = dict.fromkeys(segments, dict())

    # Calculate the forces that act on the segment based on the above presented force balance
    for index in range(len(segments)):
        segment = segments[index]

        mSeg = model[segment]['mSeg']  # Mass of the segment expressed in kilograms
        aSeg = model[segment]['aSeg']  # Linear acceleration of the segment center of mass expressed in meters/seconds2

        # Initialise dictionary F_segment to ensure correct storage of the calculated parameters in the dictionary
        F_segment[segment] = dict()

        # Calculation of the forces for all the segments in model
        if index == 0:   # F_distal = 0
            # --- Force balance first segment --- #
            F_segment[segment]['F_proximal'] = (mSeg * aSeg) - (mSeg * g) - F_extern
        else:
            # --- Force balance subsequent segments --- #
            F_segment[segment]['F_distal'] = - F_segment[segments[index - 1]]['F_proximal']  # F_proximal2distal is used of the previous subsequent segment
            F_segment[segment]['F_proximal'] = (mSeg * aSeg) - (mSeg * g) - F_segment[segment]['F_distal']

    return F_segment


def calc_net_reaction_moment(model, Fjoint, side='right', Fext=np.array([[0], [0], [0]])):
    """ This function calculates the moments around the elbow in x,y,and z axis in global
    where the rotation point is at the center of mass
    Input:
    model
    Fjoint

    Output:
    Mjoint in global around three axis

    Moment balance:
    Mprox + MFprox + MFdist + Mdist - MFext + Mext = I*alfa (=MI)
                    <-------->
    Mprox = MI - MFprox - MFdist - Mdist + MFext - Mext

    Where:
    MFdist = d x Fdist
    MFprox = d x Fprox
    MI = M_alfa - M_omega
        M_alfa = [Ixx*alfa_x, Iyy*alfa_y, Izz*alfa_z]'
        omega_mat = [[0, -omega_z, omega_y],[omega_z, 0, -omega_x],[-omega_y, omega_x,0]]
        Iomega_mat = [Ixx*omega_x, Iyy*omega_y, Izz*omega_z]'
        M_omega = omega_mat*Iomega_mat

    External forces and moments are not included yet.
    """

    # Extract the needed variables
    segments = list(model.keys())

    # Initialise dictionary
    Mjoint = dict.fromkeys(segments, dict())

    for i in range(len(segments)):
        segment = segments[i]
        segment_prev = segments[i - 1]  # extract previous segment
        Mjoint[segment] = dict()

        if 'JC' in model[segment]:
            # Extract needed parameters
            COM = model[segment]['COM']
            jointCentre = model[segment]['JC']
            jointdist = list(model[segment]['JC'])[0]
            jointdist = jointCentre.get(jointdist)
            jointprox = list(model[segment]['JC'])[1]
            jointprox = jointCentre.get(jointprox)
            segIzat = model[segment]['segIzat']  # local inertia tensor
            alfaSeg = model[segment]['alfaSeg']  # local angular acceleration
            avSeg = model[segment]['avSeg']  # local angular velocity
            gRseg = model[segment]['gRseg']  # global to local rotation matrix
            segRg = [la.inv(gRseg[index]) for index in range(len(gRseg))]  # local to global rotation matrix

            if i == 0:  # Fdistal = 0 and Mdistal = 0
                # Moments due to forces
                Fprox = Fjoint[segment]['F_proximal']
                d2 = jointprox - COM
                MFprox = np.transpose(np.array([np.cross(d2[:,index], Fprox[:,index]) for index in range(len(gRseg))]))  # Moment due to proximal force

                # Calculate moment due to the inertia: MI = M_alfa + M_omega
                M_alfa = np.transpose(np.array([np.dot(segIzat, alfaSeg[:, index]) for index in range(len(gRseg))]))
                omega_mat = np.array([[[0, -avSeg[2,index], avSeg[1,index]],
                                      [avSeg[2,index], 0, -avSeg[0,index]],
                                      [-avSeg[1,index], avSeg[0,index], 0]] for index in range(len(gRseg))])
                Iomega_mat = np.transpose(np.array([np.dot(segIzat, avSeg[:, index]) for index in range(len(gRseg))]))
                M_omega = np.transpose(np.array([np.dot(omega_mat[index], Iomega_mat[:,index]) for index in range(len(gRseg))]))
                lMI = M_alfa + M_omega
                MI = np.transpose(np.array([np.dot(segRg[index], lMI[:,index]) for index in range(len(segRg))]))  # local to global
                Mjoint[segment]['angular_momentum'] = MI

                # Calculate the net moment around the joint
                MNetProx = MI - MFprox
                Mjoint[segment]['M_proximal'] = MNetProx

            else:
                # Moments due to forces
                Fdist = Fjoint[segment]['F_distal']
                d1 = jointdist - COM
                MFdist = np.transpose(np.array([np.cross(d1[:,index], Fdist[:,index]) for index in range(len(gRseg))]))  # Moment due to distal force

                Fprox = Fjoint[segment]['F_proximal']
                d2 = jointprox - COM
                MFprox = np.transpose(np.array([np.cross(d2[:,index], Fprox[:,index]) for index in range(len(gRseg))]))  # Moment due to proximal force

                # Calculate moment due to the inertia: MI = M_alfa + M_omega
                M_alfa = np.transpose(np.array([np.dot(segIzat, alfaSeg[:, index]) for index in range(len(gRseg))]))
                omega_mat = np.array([[[0, -avSeg[2, index], avSeg[1, index]],
                                       [avSeg[2, index], 0, -avSeg[0, index]],
                                       [-avSeg[1, index], avSeg[0, index], 0]] for index in range(len(gRseg))])
                Iomega_mat = np.transpose(np.array([np.dot(segIzat, avSeg[:, index]) for index in range(len(gRseg))]))
                M_omega = np.transpose(
                    np.array([np.dot(omega_mat[index], Iomega_mat[:, index]) for index in range(len(gRseg))]))
                lMI = M_alfa + M_omega
                MI = np.transpose(
                    np.array([np.dot(segRg[index], lMI[:, index]) for index in range(len(segRg))]))  # local to global
                Mjoint[segment]['angular_momentum'] = MI

                # Extract the distal moment based on the previous segment
                MNetDist = -Mjoint[segment_prev]['M_proximal']

                # Calculate the net moment around the joint
                MNetProx = MI - MFprox - MFdist - MNetDist
                Mjoint[segment]['M_proximal'] = MNetProx

        else:
            Mjoint[segment]['M_proximal'] = np.empty((3, len(model[segment]['gRseg'])))

    return Mjoint


def moments2segment(gRseg, Mjoint):
    """ Rotates the moment components expressed in a global coordination system to the local segment coordinate system gRseg.

    Function is written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam,
    Contact Email: a.j.r.leenen@vu.nl

    Version 1.0 (2020-11-26)

    Arguments:
        gRseg: rotationmatrix of the segment used to express the calculated moments in the global coordination system
        Mjoint: calculated joint moments in the global coordination system to be expressed in local coordination system of a segment
    Returns:
        segMjoint: moments expressed in the local coordination system of gRseg.
    """

    segMjoint = np.transpose(np.array([np.dot(np.linalg.inv(gRseg[index]),Mjoint[:, index]) for index in range(len(gRseg))]))


    return segMjoint


def FC_event(toe_marker, limit=0.3, peak='first', window=25, fs = 120):
    """
    This function calculates the FC_event based on the foot/toe marker.
    FC is determined based on when the acceleration of the toe marker comes below 0.3 m/s.
    A distinction is made between calculating the foot contact with the maximum peak of the signal (first)
    or the second maximum peak of the window. For some exceptions the desired peak is not the maximum, in response
    remove the data around the maximum peak with a given window.
    Input:
        pitch ( contains markers)
    Output: index value of Foot contact (FC)

   """
    # Determine the first maximum peak
    toe = np.array(toe_marker)  # toe marker
    v_toe = calc_derivative(toe, fs)  # velocity of the toe marker
    vtoe_max = np.nanmax(v_toe[:, 1])  # take the max of the velocity of the toe marker in y-direction
    index_vtoe_max = int(np.array(np.where(v_toe[:, 1] == vtoe_max)))  # select index max v_toe happens

    if peak == 'first': # the first peak equals the desired peak
        vtoe_still = np.array(np.where(v_toe[:, 1] < limit))  # select the indices where v_toe is below 0.3 m/s

        # FC index is where vtoe_still is below 0.3 m/s for the second time after max v_toe
        indexFC = vtoe_still[vtoe_still > index_vtoe_max][1]

    if peak == 'second':    # the first peak doesn't equal the desired peak
        # remove the part around the first max from the data and calculate the second occurring maximum
        vtoe_2 = np.row_stack((v_toe[:index_vtoe_max - window, :], v_toe[index_vtoe_max + window:, :]))
        vtoe_max2 = np.nanmax(vtoe_2[:, 1])
        index_vtoe_max2 = int(np.array(np.where(v_toe[:, 1] == vtoe_max2)))  # select index max v_toe happens

        vtoe_still = np.array(np.where(v_toe[:, 1] < limit))  # select the indices where v_toe is below 0.3 m/s

        # FC index is where vtoe_still is below 0.3 m/s for the second time after max v_toe
        indexFC = vtoe_still[vtoe_still > index_vtoe_max2][1]

    return indexFC


def BR_event(marker1, marker2, indexFC, step=30):
    """
     This function calculates the BR_event based on the forearm markers.
     BR is calculated as the first frame that the RUS > RMHE in the Y-axis- throwing direction.
     (Escamilla 1998)
    Input:
        pitch ( contains markers)
     Output: index value of Ball release (BR)

     """

    start = indexFC + step    # BR is always at least 20 samples after FC

    # Initiate the variables that are needed
    RUS = np.array(marker1)
    RUSy = RUS[start:,1]    # cut window from FC on and select the y-direction
    RMHE = np.array(marker2)
    RMHEy = RMHE[start:,1]    # cut window from FC on and select the y-direction

    # calculate the index of BR
    indexBR = np.nanargmax(RUSy > RMHEy) + start

    return indexBR


def MER_event(model):
    """
    This function determines the pitch event maximum shoulder external rotation
    This is found by first calculating the shoulder external rotation which is the euler angles of the humerus
     relative to the thorax. The MER is the maximum value of the shoulder external rotation at the third rotating
     direction, which is the Z-direction.
    """
    hand_vNorm = [np.linalg.norm(model['hand']['vSeg'][:, i]) for i in range(len(model['hand']['vSeg'][0, :]))]
    BR_index = np.nanargmax(hand_vNorm) + 1

    # Determine rotation matrix of the upperarm
    R_upperarm = model['upperarm']['gRseg']

    # Determine rotation matrix of the thorax
    R_thorax = model['thorax']['gRseg']

    # Euler angles humerus relative to the thorax = shoulder external rotation
    GH = euler_angles('YXY', R_upperarm, R_thorax) #zyz
    SER = GH[2,:]  # Select the rotation of the humerus relative to the thorax in the z-direction

    for i in range(len(SER)):
        if SER[i] < -100:
            SER[i] = 360 + SER[i]

    SER = -SER # Negative is external according to ISB
    SER = butter_lowpass_filter(SER, 12, 120, 2)

    # Calculate the maximum shoulder external rotation without taking the nans into account
    MER= np.nanmax(SER[BR_index-24:BR_index])
    if (abs(MER) > 0):
        # Determine the sample where MER occurs of SER
        indexMER = int(np.array(np.where(SER==MER)))

        return np.abs(MER), indexMER
    else:
        return float('NaN'), float('NaN')


def Cross_correlation_sync_event(model_1,model_2,fs = 120):
    """
    This function determines a time lag based on the cross correlation of 2 pitches
    """

    # Determine rotation matrix of the upperarm
    # R_upperarm = model_1['upperarm']['gRseg']

    # Determine rotation matrix of the thorax
    # R_thorax = model_1['thorax']['gRseg']

    # Euler angles humerus relative to the thorax = shoulder external rotation
    # GH_1 = euler_angles('xyz', R_upperarm, R_thorax) #zyz
    # SER = GH[2,:]  # Select the rotation of the humerus relative to the thorax in the z-direction

    # Calculate the net forces according the newton-euler method
    F_joint_1 = calc_net_reaction_force(model_1)
    F_joint_2 = calc_net_reaction_force(model_2)

    # Calculate the net moments according the newton-euler method
    M_joint_1 = calc_net_reaction_moment(model_1, F_joint_1)
    M_joint_2 = calc_net_reaction_moment(model_2, F_joint_2)

#    cross_correlation = numpy.correlate(M_joint_1['forearm']['M_proximal'][0,:], M_joint_2['forearm']['M_proximal'][0,:], mode='full')
    delay_indexes = lag_finder(M_joint_1['forearm']['M_proximal'][0,:], M_joint_2['forearm']['M_proximal'][0,:], fs)
    delay_s = delay_indexes / fs

    return delay_s, delay_indexes


def lag_finder(y1, y2, sr):
    not_nan = ~np.logical_or(np.isnan(y1), np.isnan(y2))
    y1 = y1[not_nan]
    y2 = y2[not_nan]

    n = len(y1)

    delay_index = np.argmax(scipy.correlate(y2, y1, mode='same')) -int(n/2)

    return delay_index
#    plt.figure()
#    plt.plot(delay_arr, corr)
#    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
#    plt.xlabel('Lag')
#    plt.ylabel('Correlation coeff')
#    plt.show()

def butter_lowpass_filter(data, cutoff, fs, order):
    # low-pass parameters, using ba
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # filter the data
    data_out = copy.deepcopy(data)
    indices = np.where(np.invert(np.isnan(data)))[0]    # select the indices where data is not nan (in any value of the 2D array)

    if len(indices) > 15:   # input for the filter must at least be of length 15 (3*max(len(a),len(b)))
        y = scipy.signal.filtfilt(b, a, data[indices])  # filter the selected indices
        data_out[indices] = y  # replace the data with filtered indices and keep the nan's
    else:
        data_out = data

    return data_out


def butter_lowpass_filter_inning(marker_innings, cutoff, fs, order):
    filtered_inning = copy.deepcopy(marker_innings)
    for single_pitch in marker_innings:
        # Interpolate the data and cut the data to the predefined window and apply a low-pass filter
        # initialize dictionaries
        pitch_int = copy.deepcopy(marker_innings[single_pitch])
        pitch = dict()
        keys_new = list(pitch_int.keys())   # all elements in the new marker data set

        for k in range(len(pitch_int)):
            key_new = keys_new[k]   # select element name

            # filter the data separately for X,Y,Z
            pitch[key_new] = pd.DataFrame() # initialize dictionary
            pitch[key_new]['X'] = butter_lowpass_filter(np.array(pitch_int[key_new]['X']), cutoff, fs, order)
            pitch[key_new]['Y'] = butter_lowpass_filter(np.array(pitch_int[key_new]['Y']), cutoff, fs, order)
            pitch[key_new]['Z'] = butter_lowpass_filter(np.array(pitch_int[key_new]['Z']), cutoff, fs, order)
        filtered_inning[single_pitch] = pitch
    return filtered_inning


def euler_angles(decomposition_order, gRseg, gRref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
    # import scipy
    # from scipy.spatial.transform import Rotation as R
    # BUG: Needs to have NAN when data isnt available, currently defaults to 0

    # Use global coordinate system as reference coordination system in case of gRref or absence of an input
    if gRref == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        gRref = [np.array([[1, 0, 0], [0, 1, 0], [0, 0,1]])]*len(gRseg) # Create gRref with equal length as gRseg using global coordinate system of Vicon
    # If the reference coordinate system is the global coordinate system of the DSEM expressed in the Vicon global coordinate system
    if gRref == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]:
        gRref = [np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])]*len(gRseg)  # Create gRref with equal length as gRseg using global coordinate system of the DSEM
    else:
        pass


    Rjoint = np.array([np.dot(np.linalg.inv(gRref[index]), gRseg[index]) for index in range(len(gRseg))])

    joint_angles = np.transpose(R.from_matrix(Rjoint).as_euler(decomposition_order, degrees=True))

    return joint_angles


def visual_check_markers(m1=[], m2=[], m3=[], m4=[], c3dFile=[], title=''):
    """plot de data from all the markers

        Arguments:
            1) dictionary of a file
            2) first 4 arguments should be strings with names from the markers.
        Returns:
            plots from different markers at different segments
        """

    coordinate1 = 'X'
    coordinate2 = 'Y'
    coordinate3 = 'Z'
    fig = plt.figure()

    # plt.figure()
    ax_x = fig.add_subplot(311)
    if m1:
        plt.plot(c3dFile[m1][coordinate1], label=m1)
    if m2:
        plt.plot(c3dFile[m2][coordinate1], label=m2)
    if m3:
        plt.plot(c3dFile[m3][coordinate1], label=m3)
    if m4:
        plt.plot(c3dFile[m4][coordinate1], label=m4)
    plt.title(title + coordinate1 + ' coordinate')
    plt.ylabel('position in [mm]')
    plt.xlabel('samples')
    plt.legend()

    ax_y = fig.add_subplot(312)
    if m1:
        plt.plot(c3dFile[m1][coordinate2], label=m1)
    if m2:
        plt.plot(c3dFile[m2][coordinate2], label=m2)
    if m3:
        plt.plot(c3dFile[m3][coordinate2], label=m3)
    if m4:
        plt.plot(c3dFile[m4][coordinate2], label=m4)
    plt.title(coordinate2 + ' coordinate')
    plt.ylabel('postion in [mm]')
    plt.xlabel('samples')
    plt.legend()

    ax_z = fig.add_subplot(313)
    if m1:
        plt.plot(c3dFile[m1][coordinate3], label=m1)
    if m2:
        plt.plot(c3dFile[m2][coordinate3], label=m2)
    if m3:
        plt.plot(c3dFile[m3][coordinate3], label=m3)
    if m4:
        plt.plot(c3dFile[m4][coordinate3], label=m4)
    plt.title(coordinate3 + ' coordinate')
    plt.ylabel('postion in [mm]')
    plt.xlabel('samples')
    plt.tight_layout(pad=0.5)
    plt.legend()
    plt.show()


def visual_check_markers_switching(m1=[], m2=[], m3=[], m4=[], c3dFile=[], change=False, title=''):
    """plot de data from all the markers

        Arguments:
            -you can add 4 different markers to check the X axis
            - The input should be strings with names from the markers.
            - change: means that you want to change markers by switching with the use of ginput
            - the program will ask you questions what to do in the python console.
            - select the
        Returns:
            plots from different segments
            if change is True, you can use ginput to change markers. with the right mouse click you can remove the latest point
        """
    coordinate1 = 'X'
    coordinate2 = 'Y'
    coordinate3 = 'Z'
    fig = plt.figure()

    # plt.figure()
    ax_x = fig.add_subplot(311)
    if m1:
        plt.plot(c3dFile[m1][coordinate1], label=m1)
    if m2:
        plt.plot(c3dFile[m2][coordinate1], label=m2)
    if m3:
        plt.plot(c3dFile[m3][coordinate1], label=m3)
    if m4:
        plt.plot(c3dFile[m4][coordinate1], label=m4)
    plt.title(title + coordinate1 + ' coordinate')
    plt.ylabel('position in [mm]')
    plt.xlabel('samples')

    ax_y = fig.add_subplot(312)
    if m1:
        plt.plot(c3dFile[m1][coordinate2], label=m1)
    if m2:
        plt.plot(c3dFile[m2][coordinate2], label=m2)
    if m3:
        plt.plot(c3dFile[m3][coordinate2], label=m3)
    if m4:
        plt.plot(c3dFile[m4][coordinate2], label=m4)
    plt.title(coordinate2 + ' coordinate')
    plt.ylabel('postion in [mm]')
    plt.xlabel('samples')

    ax_z = fig.add_subplot(313)
    if m1:
        plt.plot(c3dFile[m1][coordinate3], label=m1)
    if m2:
        plt.plot(c3dFile[m2][coordinate3], label=m2)
    if m3:
        plt.plot(c3dFile[m3][coordinate3], label=m3)
    if m4:
        plt.plot(c3dFile[m4][coordinate3], label=m4)

    plt.title(coordinate3 + ' coordinate')
    plt.ylabel('postion in [mm]')
    plt.xlabel('samples')
    plt.tight_layout(pad=0.5)
    plt.legend()
    plt.show()

    if change:
        cursor1 = Cursor(ax_x, useblit=True, linewidth=1, color='k')
        cursor2 = Cursor(ax_y, useblit=True, linewidth=1, color='k')
        cursor3 = Cursor(ax_z, useblit=True, linewidth=1, color='k')
        zoom_ok = False
        print('\nZoom or pan to view, \npress spacebar when ready to click:\n')
        while not zoom_ok:
            zoom_ok = plt.waitforbuttonpress()

        print('Give the number of ginputs you want to use (only even numbers allowed):')
        numberGinput = int(input())
        print('Click in the figure, in the middle of the switching moment, to select switching markers. You can use the right mouse click to remove a point. ')
        dataGinput = np.array(plt.ginput(numberGinput, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2))
        index = dataGinput[:, 0]
        value = dataGinput[:, 1]
        plt.close()
        dict = {1: m1, 2: m2, 3: m3, 4: m4}
        print('Which marker do you want to change? Choose the number voor marker1:')
        print(dict)
        changeMarker1 = dict.get(int(input()))
        print('and number marker 2:')
        changeMarker2 = dict.get(int(input()))
    if change:
        return index, value, changeMarker1, changeMarker2


def visual_check_smoothing_effect(markerName, coordinateName, markers, markersNew):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    plt.plot(markers[markerName][coordinateName], label='unfiltered data')
    plt.plot(markersNew[markerName][coordinateName], label='filtered data')
    plt.title('position of the  ' + coordinateName + ' coordinate from the ' + markerName)
    plt.ylabel('postion in [mm]')
    plt.xlabel('samples')
    plt.tight_layout(pad=0.5)
    plt.legend()

    ax = fig.add_subplot(312)
    plt.plot(np.gradient(markers[markerName][coordinateName]), label='unfiltered data')
    plt.plot(np.gradient(markersNew[markerName][coordinateName]), label='filtered data')
    plt.title('velocity of the  ' + coordinateName + ' coordinate')
    plt.ylabel('velocity')
    plt.xlabel('samples')
    plt.tight_layout(pad=0.5)

    ax = fig.add_subplot(313)
    plt.plot(np.gradient(np.gradient(markers[markerName][coordinateName])), label='unfiltered data')
    plt.plot(np.gradient(np.gradient(markersNew[markerName][coordinateName])), label='filtered data')
    plt.title('acceleration of the  ' + coordinateName + ' coordinate')
    plt.ylabel('acceleration')
    plt.xlabel('samples')
    plt.tight_layout(pad=0.5)

    plt.show()


def ball_pickup_indexs(m1=[], m2=[], m3=[], m4=[], markers=[]):
    """Determines the index of ball pickups for splitting throwing sets using RRS Y coordinate.

    Function is developed and written by Thomas van Hogerwou, master student TU-Delft
    Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

    Version 1.0 (2022-03-11)

    Arguments:
        markers: Marker dictionary
    Returns:
        ball_pickups: list of ball pickup indexes correlating to dictionary indexes
    """

    # Initialize variables
    ball_pickups = [0]
    tuples = []
    coordinate2 = 'Y'
    coordinate3 = 'Z'

    # Create figure
    fig = plt.figure()
    fig.add_subplot(311)
    if m1:
        plt.plot(markers[m1][coordinate3], label=m1)
    if m2:
        plt.plot(markers[m2][coordinate3], label=m2)
    if m3:
        plt.plot(markers[m3][coordinate3], label=m3)
    if m4:
        plt.plot(markers[m4][coordinate3], label=m4)
    plt.title(coordinate3 + ' coordinate of Lower Arm')
    plt.xlim(0,len(markers[m3][coordinate3]))
    plt.ylabel('position in [mm]')
    plt.legend()

    fig.add_subplot(312)
    if m1:
        plt.plot(markers[m1][coordinate2], label=m1)
    if m2:
        plt.plot(markers[m2][coordinate2], label=m2)
    if m3:
        plt.plot(markers[m3][coordinate2], label=m3)
    if m4:
        plt.plot(markers[m4][coordinate2], label=m4)
    plt.title(coordinate2 + ' coordinate of Lower Arm')
    plt.xlim(0,len(markers[m3][coordinate3]))
    plt.ylabel('position in [mm]')
    plt.xlabel('samples')
    plt.legend()

    fig.add_subplot(313)
    plt.plot(np.gradient(np.array(markers['VU_Baseball_R_C7']['Y'])))
    plt.title('Gradient of C7')
    plt.xlim(0,len(markers[m3][coordinate3]))

    # Use ginput to manually select cut points
    tuples = plt.ginput(15,-1,show_clicks= True, mouse_add=1, mouse_pop=3, mouse_stop=2)
    for i in range(len(tuples)):
        ball_pickups.append(np.round(tuples[i][0]))

    # Creat list of cut_points
    ball_pickups.append(len(markers[m1])+1)

    ball_pickups = [int(x) for x in ball_pickups]

    return ball_pickups


def cut_markers(markers=[], ball_pickups=[], inning = []):
    """Cuts marker data at indexs given by ball pickups

    Function is developed and written by Thomas van Hogerwou, master student TU-Delft
    Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

    Version 1.0 (2022-03-14)

    Arguments:
        markers: Marker dictionary
        ball_pickups: indexs of ball pickup
    Returns:
        cut_markers : dictionary contatining dictionarys of each individual pitch
    """
    markers_cut = {}

    # Give new name to each pitch based on inning number
    for i in range(len(ball_pickups)-1):
        markers_cut["pitch_{0}".format((10*(inning-1)) + i+1)] = markers.copy()

    i = 0

    # Select only the relevant data to remain in new markers_cut
    for pitch in markers_cut:
        for marker in markers_cut[pitch]:
            markers_cut[pitch][marker] = markers_cut[pitch][marker].iloc[ball_pickups[i]:ball_pickups[i+1]]
        i = i + 1
    return markers_cut


def trim_markers(markers, fs = 120, lead = .2, lag = .8):
    """trims marker data based on max derivative of C7

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-14)

       Arguments:
           markers: Marker dictionary
           lead: trim time in s before max V
           lag: lag time in s after max V
           wc: cuttoff frequency of filter
       Returns:
           trimmed_markers : dictionary contatining dictionarys of each individual pitch trimed to new indexes
       """
    index_offset = 0
    pitches_trimmed = copy.deepcopy(markers)
    for pitch in pitches_trimmed:
        # order: C7-int-cut-filt
        ## select C7 marker to define a window at which to cut the data
        backmark = np.array(pitches_trimmed[pitch]['VU_Baseball_R_C7']['Y'])
        knip = (np.gradient(backmark)) # take 1st derivative of C7
        knip_max = np.nanmax((knip))    # take the max of the 2nd derivative of C7 in the y-direction
        index_cut = index_offset + int(np.array(np.where(knip == knip_max)))  # select the index this event happens

        # improve the data by removing * elements
        keys = list(pitches_trimmed[pitch].keys())  # all the elements
        for j in range(len(pitches_trimmed[pitch])):
            in_keys = keys[j]
            # if a * is present in the element name, remove the element
            if '*' in in_keys:
                pitches_trimmed[pitch].pop(in_keys)

        # initialize dictionaries
        pitch_int = copy.deepcopy(pitches_trimmed[pitch])

        pitch_in = dict()
        keys_new = list(pitch_int.keys())   # all elements in the new marker data set

        for k in range(len(pitch_int)):
            key_new = keys_new[k]   # select element name
            # if key_new != 'RAC1' and key_new != 'LMM': # certain pitches could cause errors
            # cut the data
            pitch_in[key_new] = pitch_int[key_new].loc[(index_cut - lead*fs):(index_cut + lag*fs), :] # cut the data .2s before and .4s after the index
        pitches_trimmed[pitch] = pitch_in
        index_offset = index_offset + len(markers[pitch]['VU_Baseball_R_C7'])
    return pitches_trimmed


def orient_markers(markers):
    """Orients marker data to follow Bart's previous work

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-14)

       Arguments:
           markers: Marker dictionary
       Returns:
           oriented_markers : dictionary contatining dictionarys of each individual pitch oriented to new axis
       """
    oriented_markers = copy.deepcopy(markers)
    for marker in markers:
        oriented_markers[marker]['X'] = [markers[marker]['X'][index] * -1 for index in range(len(markers[marker]['X']))]
        oriented_markers[marker]['Y'] = [markers[marker]['Z'][index]  for index in range(len(markers[marker]['X']))]
        oriented_markers[marker]['Z'] = [markers[marker]['Y'][index]  for index in range(len(markers[marker]['X']))]
    return oriented_markers


def plot_inning_segment_moments(seg_M_joint,pitch_number,figure_number = 1):
    """Plots the moments of all pitches in an inning for a given segment name on the segment local frame

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-24)

       Arguments:
           seg_M_joint: Marker dictionary
       """
    seg_M_joint_norm = [np.linalg.norm(seg_M_joint['forearm'][:, index]) for index in range(len(seg_M_joint['forearm'][0,:]))]


    plt.figure(figure_number)
    plt.subplot(411)
    plt.plot(seg_M_joint['forearm'][0, :], label=pitch_number)
    plt.title('Moments Projected on Forearm Coordination System : Add(+)/Abd(-)')
    plt.ylabel('Moment [Nm]')
    plt.legend()

    plt.subplot(412)
    plt.title('Moments Projected on Forearm Coordination System : Pro(+)/Sup(-)')
    plt.plot(seg_M_joint['forearm'][1, :], label=pitch_number)
    plt.ylabel('Moment [Nm]')

    plt.subplot(413)
    plt.title('Moments Projected on Forearm Coordination System : Flex(+)/Ext(-)')
    plt.plot(seg_M_joint['forearm'][2, :], label=pitch_number)
    plt.xlabel('Samples')
    plt.ylabel('Moment [Nm]')

    plt.subplot(414)
    plt.title('Norm of Moments Projected on Forearm Coordination System')
    plt.plot(seg_M_joint_norm, label=pitch_number)
    plt.xlabel('Samples')
    plt.ylabel('Moment [Nm]')


def plot_inning_mean_moments(time,mean,pos_var,neg_var,figure_number = 1):
    """Plots the moments of all pitches in an inning for a given segment name on the segment local frame

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-24)

       Arguments:
           seg_M_joint: Marker dictionary
       """
    mean_norm = [np.linalg.norm(mean['forearm'][:, index]) for index in range(len(mean['forearm'][0,:]))]
    pos_var_norm = [np.linalg.norm(pos_var['forearm'][:, index]) for index in range(len(pos_var['forearm'][0,:]))]
    neg_var_norm = [np.linalg.norm(neg_var['forearm'][:, index]) for index in range(len(neg_var['forearm'][0,:]))]

    plt.figure(figure_number)
    plt.subplot(411)
    plt.plot(time,mean['forearm'][0, :], label='mean')
    plt.fill_between(time,neg_var['forearm'][0, :],pos_var['forearm'][0, :],alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('Moments Projected on Forearm Coordination System : Add(+)/Abd(-)')
    plt.ylabel('Moment [Nm]')
    plt.xlabel('Time [s]')
    plt.xlim(.5,1)
    plt.legend()

    plt.subplot(412)
    plt.title('Moments Projected on Forearm Coordination System : Pro(+)/Sup(-)')
    plt.plot(time,mean['forearm'][1, :], label='mean')
    plt.fill_between(time,neg_var['forearm'][1, :],pos_var['forearm'][1, :],alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.ylabel('Moment [Nm]')
    plt.xlabel('Time [s]')
    plt.xlim(.5,1)

    plt.subplot(413)
    plt.title('Moments Projected on Forearm Coordination System : Flex(+)/Ext(-)')
    plt.plot(time,mean['forearm'][2, :], label='mean')
    plt.fill_between(time,neg_var['forearm'][2, :],pos_var['forearm'][2, :],alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.xlabel('Time [s]')
    plt.xlim(.5,1)
    plt.ylabel('Moment [Nm]')

    plt.subplot(414)
    plt.title('Norm of Moments Projected on Forearm Coordination System')
    plt.plot(time,mean_norm, label='mean')
    plt.fill_between(time,neg_var_norm,pos_var_norm,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.xlabel('Time [s]')
    plt.ylabel('Moment [Nm]')
    plt.xlim(.5,1)
    plt.show(block=False)


def time_sync_moment_data(data, lag):
    """Time syncs model segments by time delay "lag"

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-24)

       Arguments:
           data: Segment model of single pitch event
           lag: amount of lag to add to model, indexes seperated from desired
                ex: [-1 -2 1 0 -1 2]

       Returns:
           synced_model: Segment model of same size, with laged data, padded with nans
       """
    synced_data = copy.deepcopy(data)
    for segment in data:
        synced_data[segment] = np.array([[data[segment][axis][index + lag] for index in range(len(data[segment][0]) - lag)]for axis in range(len(data['forearm']))])
        if lag < 0:
            for axis in range(len(data['forearm'])):
                for i in range(0, (-1 * lag)):
                    synced_data[segment][axis][i] = np.nan # remove the [-i] loop around python does, replace with nans

    return synced_data


def time_sync_force_data(data, lag):
    """Time syncs model segments by time delay "lag"

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-24)

       Arguments:
           data: Segment model of single pitch event
           lag: amount of lag to add to model

       Returns:
           synced_model: Segment model of same size, with laged data
       """
    synced_data = copy.deepcopy(data)
    for segment in data:
        for location in data[segment]:
            synced_data[segment][location] = np.array([[data[segment][location][axis][index + lag] for index in range(len(data[segment][location][0]) - lag)]for axis in range(len(data['forearm']['F_proximal']))])
            if lag < 0:
                for axis in range(len(data['forearm'])):
                    for i in range(0, (-1 * lag)):
                        synced_data[segment][location][axis][i] = 'NaN' # remove the [-i] loop around python does, replace with nans

    return synced_data


def calc_variability_seg_M_joint(Inning_seg_M_joint):
    """calculates mean and variability of an inning of pitches

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-25)

       Arguments:
            Inning_seg_M_joint: Moment data of all segments in an inning
            Inning_MER_events: List of MER events, only used to find shortest pitch
       Returns:
           Inning_mean_seg_M_joint: list of means based on time synced data
           Inning_var_seg_M_joint: list of variability based on time synced data
   """
    pitch_numbers = Inning_seg_M_joint.keys()
    # Determine seg names and a default shortest pitch
    for pitch_number in Inning_seg_M_joint:
        seg_names = Inning_seg_M_joint[pitch_number].keys()
        shortest_pitch = pitch_number
        break

    Inning_mean_seg_M_joint = dict.fromkeys(seg_names)
    Inning_var_seg_M_joint = dict.fromkeys(seg_names)
    Inning_mean_pos_var_seg_M_joint = dict.fromkeys(seg_names)
    Inning_mean_neg_var_seg_M_joint = dict.fromkeys(seg_names)

    for segment in seg_names:
        for pitch in pitch_numbers:
            if len(Inning_seg_M_joint[pitch]['pelvis'][0,:]) < len(Inning_seg_M_joint[shortest_pitch]['pelvis'][0,:]):
                shortest_pitch = pitch

        # Initiallize arrays
        seg_mean = np.empty([len(Inning_seg_M_joint[shortest_pitch]['pelvis'][0,:]),3])
        seg_mean[:] = np.NaN
        seg_var = np.empty([len(Inning_seg_M_joint[shortest_pitch]['pelvis'][0,:]),3])
        seg_var[:] = np.NaN

        # determine indexes with at least 3 values
        index_list = []
        for index in range(len(Inning_seg_M_joint[shortest_pitch]['forearm'][1])):
            nan_counter = 0
            for pitch in pitch_numbers:
                if np.isnan(np.mean(Inning_seg_M_joint[pitch]['forearm'][:,index])):
                    nan_counter = nan_counter + 1
            if nan_counter < 5:
                index_list.append(index)

        # calc mean and var for given indexes
        for index in index_list:
            seg_mean[index,:] = (np.nanmean([Inning_seg_M_joint[pitch]['forearm'][:,index] for pitch in Inning_seg_M_joint],0))
            seg_var[index,:] = (np.nanvar([Inning_seg_M_joint[pitch]['forearm'][:,index] for pitch in Inning_seg_M_joint],0))

        seg_mean = np.transpose(np.array(seg_mean))
        seg_var = np.transpose(np.array(seg_var))

        Inning_mean_seg_M_joint[segment] = seg_mean
        Inning_var_seg_M_joint[segment] = seg_var
        Inning_mean_pos_var_seg_M_joint[segment] = seg_mean + seg_var
        Inning_mean_neg_var_seg_M_joint[segment] = seg_mean - seg_var

    return Inning_mean_seg_M_joint, Inning_var_seg_M_joint, Inning_mean_pos_var_seg_M_joint, Inning_mean_neg_var_seg_M_joint

def Add_velocity_to_output(pitcher, inning, Fatigue_dictionary = []):
    """Adds velocity pitch data from an excel to a dictionary

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-25)

       Arguments:
            pitcher:
            inning:
       Returns:
           Output: Dictionary containing added data
    """
    data_key = 'Ball Speed'
    """
    Results Dictionary
    """

    if Fatigue_dictionary:
        Output_Dictionary = Fatigue_dictionary
    else:
        path = 'Results/Pitches/Unfiltered/' + pitcher + '/' + inning + '/'
        # --- Load data from pickle --- #
        filenameIn = path + 'Max_norm_moments'
        infile = open(filenameIn, 'rb')
        Output_Dictionary = pickle.load(infile)
        infile.close()

    """
    xlsx data
    """
    filename = "data/21_03_22_Overview UCL variability fatigue_graphs_BvT(1).xlsx"
    xlsx_data = openpyxl.load_workbook(filename=filename,data_only=True)

    df = pd.DataFrame(xlsx_data[pitcher].values) # make it a df


    segments = Output_Dictionary.keys()


    for segment in segments:
        Output_Dictionary[segment][data_key] = dict()
        for pitch in Output_Dictionary[segment]['max_abduction_moment']:
            xlsx_pitches = df[0].squeeze()
            rows = xlsx_pitches.str.find(pitch,end = len(pitch))
            for i in range(len(rows)):
                if rows[i] == 0:
                    row = i
                    break
            new_pitch_data = df[1][row]#grab from excel data
            Output_Dictionary[segment][data_key][pitch] = new_pitch_data

    return Output_Dictionary

def Add_fatigue_to_output(pitcher, inning, Fatigue_dictionary = []):
    """Adds velocity pitch data from an excel to a dictionary

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-25)

       Arguments:
            pitcher:
            inning:
       Returns:
           Output: Dictionary containing added data
    """
    data_key = 'Fatigue Reports'
    """
    Results Dictionary
    """

    if Fatigue_dictionary:
        Output_Dictionary = Fatigue_dictionary
    else:
        path = 'Results/Pitches/Unfiltered/' + pitcher + '/' + inning + '/'
        # --- Load data from pickle --- #
        filenameIn = path + 'Outputs'
        infile = open(filenameIn, 'rb')
        Output_Dictionary = pickle.load(infile)
        infile.close()

    """
    xlsx data
    """
    filename = "data/21_03_22_Overview UCL variability fatigue_graphs_BvT(1).xlsx"
    xlsx_data = openpyxl.load_workbook(filename=filename,data_only=True)

    df = pd.DataFrame(xlsx_data['VAS-scale'].values) # make it a df

    """"
    Find correct col
    """
    for j in range(20):
        if df[j][0] == pitcher:
            break

    segments = Output_Dictionary.keys()
    for segment in segments:
        Output_Dictionary[segment][data_key] = dict()
        Output_Dictionary[segment][data_key]['VAS-Scale 1 (Before US measurements)'] = df[j][1]
        Output_Dictionary[segment][data_key]['VAS-Scale 2 (Start pitching)'] = df[j][2]

        for pitch in Output_Dictionary[segment]['max_abduction_moment']:
            if pitch in ['pitch_1','pitch_2','pitch_3','pitch_4','pitch_5','pitch_6','pitch_7','pitch_8','pitch_9','pitch_10']:
                row = 3
            if pitch in ['pitch_11','pitch_12','pitch_13','pitch_14','pitch_15','pitch_16','pitch_17','pitch_18','pitch_19','pitch_20']:
                row = 4
            if pitch in ['pitch_21','pitch_22','pitch_23','pitch_24','pitch_25','pitch_26','pitch_27','pitch_28','pitch_29','pitch_30']:
                row = 5
            if pitch in ['pitch_31','pitch_32','pitch_33','pitch_34','pitch_35','pitch_36','pitch_37','pitch_38','pitch_39','pitch_40']:
                row = 6
            if pitch in ['pitch_41','pitch_42','pitch_43','pitch_44','pitch_45','pitch_46','pitch_47','pitch_48','pitch_49','pitch_50']:
                row = 7
            if pitch in ['pitch_51','pitch_52','pitch_53','pitch_54','pitch_55','pitch_56','pitch_57','pitch_58','pitch_59','pitch_60']:
                row = 8
            if pitch in ['pitch_61','pitch_62','pitch_63','pitch_64','pitch_65','pitch_66','pitch_67','pitch_68','pitch_69','pitch_70']:
                row = 9
            if pitch in ['pitch_71','pitch_72','pitch_73','pitch_74','pitch_75','pitch_76','pitch_77','pitch_78','pitch_79','pitch_80']:
                row = 10
            if pitch in ['pitch_81','pitch_82','pitch_83','pitch_84','pitch_85','pitch_86','pitch_87','pitch_88','pitch_89','pitch_90']:
                row = 11
            if pitch in ['pitch_91','pitch_92','pitch_93','pitch_94','pitch_95','pitch_96','pitch_97','pitch_98','pitch_99','pitch_100']:
                row = 12
            if pitch in ['pitch_101','pitch_102','pitch_103','pitch_104','pitch_105','pitch_106','pitch_107','pitch_108','pitch_109','pitch_110']:
                row = 13

            new_pitch_data = df[j][row]
            Output_Dictionary[segment][data_key][pitch] = new_pitch_data

    return Output_Dictionary

def max_moment_data(Fatigue_dictionary, seg_M_joint, segments, pitch_number, Polyfit = 0):
    """Adds velocity pitch data from an excel to a dictionary

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-25)

       Arguments:
            pitcher:
            inning:
       Returns:
           Output: Dictionary containing added data
    """

    # Max moment data for fatigue study
    if Polyfit == 0: #aka no polyfit solution based on max
        for segment in segments:
            Fatigue_dictionary[segment]['max_abduction_moment'][pitch_number] = np.nanmax([(seg_M_joint[segment][0, index]) for index in range(len(seg_M_joint[segment][0, :]))])

    if Polyfit == 1:
        for segment in segments:
            seg_index = np.nanargmax([(seg_M_joint[segment][0, index]) for index in range(len(seg_M_joint[segment][0, :]))])
            # --- Select window based on found indices for each segment --- #
            seg_window = np.linspace(seg_index - 2, seg_index + 2, ((seg_index + 2) - (seg_index - 2) + 1), endpoint=True).astype(int)
            # --- Calculate the 2nd order polynomial function coefficients for each segment --- #
            seg_fit = np.polyfit(seg_window, seg_M_joint[segment][0,seg_window], 2)
            # --- Analytical calculation of the exact point in time of the occurrence of the peak angular velocity for each segment --- #
            seg_analytical_time = -(seg_fit[1] / (2 * seg_fit[0]))

            if seg_analytical_time > 0:
                Fatigue_dictionary[segment]['max_abduction_moment'][pitch_number] = np.poly1d(seg_fit)(seg_analytical_time)
            else:
                Fatigue_dictionary[segment]['max_abduction_moment'][pitch_number] = np.nanmax([(seg_M_joint[segment][0, index]) for index in range(len(seg_M_joint[segment][0, :]))])

    return Fatigue_dictionary

def convert_bits2emg(dataframe, resolution=16):
    """Apply conversion factor to Biosignalsplux dataframe.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)

    Arguments:
        dataframe: original dataframe in bits
        resolution: number of bits
    Returns:
        dataframe: converted to proper units (e.g. EMG in mV)
    """

    # Initialise variables
    operating_voltage = 3  # Range: -1.5mV to +1.5mV
    gain = 506  # Standard unchangeable value (build-in sensor hardware)

    # Conversion formula presented in the Biosignalsplux factsheet
    dataframe = (((dataframe / (2 ** resolution)) - 0.5) * operating_voltage) / gain

    return dataframe


def read_plux(path, calibration_factors=np.vstack([28000, 38000]), conversion=1):
    """Reads the Biosignalsplux data with .txt file(s) into dictionary.

    Function is developed and written by Ton Leenen, PhD-Candidate Vrije Universiteit Amsterdam
    Contact E-Mail: a.j.r.leenen@vu.nl

    Version 1.0 (2020-03-19)
    Version 1.5 (2021-03-15) - Changed pandas data organisation to numpy for computational speed improvements
                               Adapted organisation of measured muscles in dictionary

    Arguments:
        path: path or folder containing the EMG data with .txt file(s)
        calibration_factors: accelerometer calibration factors derived from
                             calibration procedure (function: acc_calibration_factors)
        conversion: read raw data in bits (0 = raw, 1 = transfer bits to emg, 2 = transfer bits to acc)
    Returns:
        dictionary: containing (multiple) dataframe(s) with data from multiple measurements
        !!! EMG data is pre-filtered with a bandwidth filter of 25-500Hz !!!
    """

    # Initialise nested dictionary in a nested list
    dictionary = dict()

    # Load EMG data with .txt file(s) from path or folder
    for f in sorted_alphanumeric(os.listdir(path)):

        # Load the .txt file(s) in the path or folder only
        if f.endswith(".txt"):
            # Read .txt file(s) to extract metadata
            txt = open(f'{path}/{f}', "r")
            print("Loading " + str(f))

            # Read the string representation of a .txt file(s) and
            # convert only the second line of the file to a metadata list
            metadata = ast.literal_eval(txt.readlines()[1][24:-2])

            # Fix column names if broken
            metadata['column'] = ['nSeq', 'DI', 'FMP', 'BIC', 'TRI', 'DM', 'PS', 'PC', 'LD', 'ACC']

            # Split date and time from metadata to chunks
            date = metadata['date'][:4] + "0" + metadata['date'][4:].replace("-",
                                                                             "")  # Select measurement date (notation in yyyy:mm:dd)
            time = metadata['time'].replace(":", "").replace(".", "")[
                   :-1] + '0'  # Select measurement time (notation in hh:mm:ss)

            time_ms = metadata['time'].replace(":", "").replace(".",
                                                                "") + '00'  # Select measurement time with ms (notation in hh:mm:ss:ms)
            date_time_ms = date + time_ms  # Select measurement date and time with ms (notation in yyyy:mm:dd hh:mm:ss:ms)

            # Read .txt file(s) to extract the data
            dataframe = np.array(np.loadtxt(f'{path}/{f}'))

            # Drop the first and second columns containing index values
            dataframe = np.delete(dataframe, [0, 1], 1)

            # Convert bits to proper units
            if conversion == 1:  # Transfer raw values (bits) to emg
                dataframe = convert_bits2emg(dataframe, metadata["resolution"][0])

            elif conversion == 2:  # Transfer raw values (bits) to acc
                dataframe = convert_bits2acc(dataframe, calibration_factors, metadata["resolution"][0])

            # Construct timeline
            dataframe = EMG_timeline(dataframe, metadata["sampling rate"])

            # Define start time
            start_time = datetime.strptime(date_time_ms, "%Y%m%d%H%M%S%f")

            # Construct realtime timeline based on start time and created timeline based on sample frequency
            realtime = np.array(
                [start_time + timedelta(hours=0, minutes=0, seconds=i, microseconds=0) for i in dataframe[:, 1]])

            # Emerge realtime timeline into dataframe
            dataframe = np.column_stack((dataframe, realtime))

            # Convert numpy array to pandas dataframe with headers
            dataframe = pd.DataFrame(dataframe, columns=metadata['column'][2:] + ['time_s'] + ['real_time'])

            # Initialise all levels of the dictionary
            if date not in dictionary.keys():
                dictionary[date] = dict()

            if time not in dictionary[date].keys():
                dictionary[date][time] = dict()

            # Assign dataframe to sorted dictionary
            dictionary[date][time] = dataframe

            # for muscle in metadata["column"][-1:]:
            #     if muscle not in dictionary[date][time].keys():
            #         dictionary[date][time][muscle] = np.array([])
            #
            #         # Define index of muscle in metadata
            #         index = metadata["column"][-1:].index(muscle)
            #
            #         # Assign dataframe to sorted dictionary
            #         dictionary[date][time][muscle] = dataframe[:, index]

            print('Finished!\n')

    return dictionary

def Add_MVC_EMG_data(Dictionary,pitcher,fs_desired = 2000):
    """Adds MVC EMG data to EMG dictionary as part of preprocess_EMG

       Function is developed and written by Thomas van Hogerwou, master student TU-Delft
       Contact E-Mail: T.C.vanHogerwou@student.tudelft.nl

       Version 1.0 (2022-03-25)

       Arguments:
            Dictionary: Input dictionary
            pitcher: participant number "PP01" etc.
            fs_desired: Sampling frequency of output
       Returns:
            Output: Dictionary containing added data
    """

    ## input parameters ##

    fs_EMG = 2000
    fs_scaling = fs_desired / fs_EMG
    EMG_markers = ['ACC', 'BIC', 'DM', 'FMP', 'LD', 'PC', 'PS', 'TRI', 'real_time', 'time_s']
    Output = copy.deepcopy(Dictionary)

    path = "data/EMG_Data/01 Raw Data/{}/MVC/".format(pitcher)
    plux = read_plux(path, conversion=1)
    # Extract keys from dictionary plux
    date_plux = list(plux.keys())[0]
    time_plux = list(plux[date_plux].keys())[0]

    EMG_MVC_2000Hz = plux[date_plux][time_plux]

    """
    If fs_desired != fs_EMG, downsample to fs_desired
    """
    if fs_scaling != 1:
        EMG_MVC_downsampled = dict.fromkeys(EMG_MVC_2000Hz)
        for EMG_marker in EMG_markers:
            EMG_MVC_downsampled[EMG_marker] = sp.resample(EMG_MVC_2000Hz[EMG_marker],int(len(EMG_MVC_2000Hz[EMG_marker]) * fs_scaling))
            Output[EMG_marker]['MVC'] = EMG_MVC_downsampled[EMG_marker]
    else:
        for EMG_marker in EMG_markers:
            Output[EMG_marker]['MVC'] = EMG_MVC_2000Hz[EMG_marker]
    return Output