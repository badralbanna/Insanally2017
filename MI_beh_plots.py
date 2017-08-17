############################################################################
# MI_beh_plots.py
#
# This is a set of python functions to analyize the data from behaving animals
#
# Created by Badr F. Albanna 4/19/14
# Last edited by Badr F. Albanna 1/31/15
# Copyright 2014 Badr F. Albanna
############################################################################

import os
import numpy
import cPickle
import pickle
import scipy as sp
from mpld3 import plugins
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm, kde
from scipy.integrate import quad
from statsmodels.nonparametric.kde import KDEUnivariate
from numpy.random import rand
from numbers import Number
from kkpandas.kkrs import is_auditory

#############
# Constants #
#############

AC_COLOR = 'red'
PFC_COLOR = '#4C4C4C'
SIZE = 6
LINEWIDTH = 1
PERCENT_FORMAT_STRING = '{:<}'

###################
# Basic Utilities #
###################


def find_choice(trace, times, at_best=True):
    diff = trace[:, 0] - trace[:, 1]
    max_ind = numpy.argmax(diff)
    min_ind = numpy.argmin(diff)
    if at_best:
        if diff[max_ind] > -diff[min_ind]:
            probs = trace[max_ind]
            count = numpy.array([1., 0.])
            time = times[max_ind]
        elif diff[max_ind] < -diff[min_ind]:
            probs = trace[min_ind]
            count = numpy.array([0., 1.])
            time = times[min_ind]
        else:
            probs = numpy.array([0.5, 0.5])
            count = numpy.array([0.5, 0.5])
            time = numpy.nan
    else:
        probs = trace[-1]
        time = numpy.nan
        if probs[0] > probs[1]:
            count = numpy.array([1., 0.])
        elif probs[0] < probs[1]:
            count = numpy.array([0., 1.])
        else:
            count = numpy.array([.5, .5])
    return(time, probs, count)


def smooth(x, window_len=3, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=numpy.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))

    if window == 'flat': #moving average
        w = numpy.ones(window_len,'d')
    else:
        w = getattr(numpy, window)(window_len)
    y = numpy.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def counts_iterator(directory):
    for file_name in os.listdir(directory):
        if file_name[-7:] == ".pickle":
            try:
                with open(directory+file_name, 'rb') as f:
                    counts = pickle.load(f)
            except:
                print("Problem with {0}".format(file_name))
            yield(file_name[:-7], counts)


def counts_load(directory, seperate_files=False, reformat_name=True, max_size=None, seperator='-'):
    counts = {}
    for file_name in os.listdir(directory):
        if file_name[-7:] == ".pickle" and (file_name[0] == 'A' or file_name[0] == 'P'):
            if not seperate_files:
                animal = file_name[:-7]
                try:
                    with open(directory+file_name, 'rb') as f:
                        animal_dict = pickle.load(f)
                        counts[animal] = {}
                        for neuron, data in animal_dict.iteritems():
                            print("Loading {0}, {1}".format(animal, neuron))
                            if reformat_name:
                                counts[animal][tuple([neuron])] = data
                            else:
                                counts[animal][neuron] = data
                except:
                    raise
                    print("Problem with {0}".format(file_name))
            if seperate_files:
                try:
                    animal, neurons = file_name[:-7].split(seperator)
                except:
                    print(file_name)
                    raise
                if animal not in counts:
                    counts[animal] = {}
                neurons = tuple(eval(neurons))
                if (max_size is None) or (max_size >= len(neurons)):
                    try:
                        with open(directory + file_name, 'rb') as f:
                            data = pickle.load(f)
                            print("Loading {0}, {1}".format(animal, neurons))
                            if reformat_name:
                                counts[animal][neurons] = data[neurons]
                            else:
                                counts[animal][neurons] = data
                    except:
                        #raise
                        print("Problem with {0}".format(file_name))
                else:
                    continue
    return(counts)

def counts_load_chris(directory, seperate_files=False, reformat_name=True, max_size=None, seperator='-'):
    counts = {}
    for file_name in os.listdir(directory):
        if file_name[-7:] == ".pickle" and (file_name[0] == 'C'):
            if not seperate_files:
                animal = file_name[:-7]
                try:
                    with open(directory+file_name, 'rb') as f:
                        animal_dict = pickle.load(f)
                        counts[animal] = {}
                        for neuron, data in animal_dict.iteritems():
                            print("Loading {0}, {1}".format(animal, neuron))
                            if reformat_name:
                                counts[animal][tuple([neuron])] = data
                            else:
                                counts[animal][neuron] = data
                except:
                    raise
                    print("Problem with {0}".format(file_name))
            if seperate_files:
                try:
                    animal, neurons = file_name[:-7].split(seperator)
                except:
                    print(file_name)
                    raise
                if animal not in counts:
                    counts[animal] = {}

                neurons = eval(neurons)

                if (max_size is None) or (max_size >= len(neurons)):
                    try:
                        with open(directory + file_name, 'rb') as f:
                            data = pickle.load(f)
                            print("Loading {0}, {1}".format(animal, neurons))
                            if reformat_name:
                                counts[animal][(neurons,)] = data[neurons]
                            else:
                                counts[animal][neurons] = data
                    except:
                        raise
                        print("Problem with {0}".format(file_name))
                else:
                    continue
    return(counts)


def responses_load(directory):
    responses = {}
    for file_name in os.listdir(directory):
        if file_name[-7:] == ".pickle":
            animal = file_name[:-7]
            try:
                with open(directory+file_name, 'rb') as f:
                    print(animal)
                    animal_dict = pickle.load(f)
                    print(animal_dict.keys())
                    responses[animal] = {}
                    for neuron, data in animal_dict.iteritems():
                        responses[animal][(neuron, )] = data
            except:
                raise
                print("Problem with {0}".format(file_name))
    return(responses)


def rewrite_cell_list(selective_cells):
    selective_cell_dict = {}
    for cell in selective_cells:
        animal, neuron = cell.split('-')
        if animal in selective_cell_dict:
            selective_cell_dict[animal].append(tuple(eval(neuron)))
        else:
            selective_cell_dict[animal] = [tuple(eval(neuron))]
    return(selective_cell_dict)


def all_pairs(group):
    for i in range(len(group)):
        for j in range(len(group) - i - 1):
            yield((group[i], group[i + j + 1]))


def create_average_curve(values):
    return(numpy.mean(values, axis=0), numpy.std(values, axis=0))


def delta_log_liklihood(curve_1, curve_2, curve_std=None):
    difference = numpy.nan_to_num(numpy.log(curve_1)) - numpy.nan_to_num(numpy.log(curve_2))
    if curve_std is not None:
        difference_l = numpy.log(curve_1 - curve_std[0]) - numpy.log(curve_2 + curve_std[1])
        difference_h = numpy.log(curve_1 + curve_std[0]) - numpy.log(curve_2 - curve_std[1])
        return(difference, difference_l, difference_h)
    else:
        return(difference)


def inf_dot_product(difference_1, difference_2, cutoff=None, times=None):
    if cutoff is None:
        cutoff = (0, len(difference_1))
    if times is not None:
        cutoff_2 = (int(numpy.where(times >= cutoff[0])[0][0]), int(numpy.where(times <= cutoff[1])[0][-1]))
        cutoff = cutoff_2
    norm_value = numpy.product([numpy.sqrt(numpy.sum(x[cutoff_2[0]:cutoff_2[1]]**2)) for x in (difference_1, difference_2)], axis=0)
    dot = numpy.sum(numpy.product([x[cutoff_2[0]:cutoff_2[1]] for x in (difference_1, difference_2)], axis=0)) / norm_value
    if not numpy.isfinite(dot):
        print('oops!')
    return(dot)


#########################################
# Functions for processing joint counts #
#########################################


def calculate_MI(joint_counts):
    def entropy_term(x):
        if x == 0:
            return(0)
        else:
            return(-x*numpy.log2(x))
    vec_entropy_term = numpy.vectorize(entropy_term)

    p0 = numpy.sum(joint_counts, axis = 1) / numpy.sum(joint_counts)
    p1 = numpy.sum(joint_counts, axis = 0) / numpy.sum(joint_counts)
    p0g1 = joint_counts.copy()
    p0g1[:,0] = p0g1[:,0] / numpy.sum(p0g1[:,0])
    p0g1[:,1] = p0g1[:,1] / numpy.sum(p0g1[:,1])

    entropy0 = vec_entropy_term(p0).sum()
    entropy0g1 = vec_entropy_term(p0g1[:,0]).sum()*p1[0] + vec_entropy_term(p0g1[:,1]).sum()*p1[1]
    return( ( (entropy0 -  entropy0g1) / entropy0, ))


def percent_correct(joint_probability):
    total_counts = numpy.sum(joint_probability)
    correct = (joint_probability[0,0] + joint_probability[1,1] + 1.) / (total_counts + 2.)
    incorrect = (joint_probability[1,0] + joint_probability[0,1] + 1.) / (total_counts + 2.)
    return(correct, incorrect)


def d_prime(joint_counts):
    if numpy.any(numpy.sum(joint_counts, axis = 0) == 0):
        return((0.,))
    joint_counts = numpy.array(joint_counts)
    hit_rate = joint_counts[0,0] / (joint_counts[0,0] + joint_counts[0,1])
    false_alarm_rate = joint_counts[1,0] / (joint_counts[1,0] + joint_counts[1,1])
    if hit_rate == 1:
        hit_rate = 1 - 1/(2.*(joint_counts[0,0] + joint_counts[0,1]))
    if false_alarm_rate == 1:
        false_alarm_rate = 1 - 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))
    if hit_rate == 0:
        hit_rate = 1/(2*(joint_counts[0,0] + joint_counts[0,1]))
    if false_alarm_rate == 0:
        false_alarm_rate = 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))
    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(false_alarm_rate)
    return((z_hit - z_fa,))


def sensitivity_and_bias(joint_counts):
    joint_counts = numpy.array(joint_counts, dtype='float')
    hit_rate = joint_counts[0,0] / (joint_counts[0,0] + joint_counts[0,1])
    false_alarm_rate = joint_counts[1,0] / (joint_counts[1,0] + joint_counts[1,1])
    if hit_rate == 1:
        hit_rate = 1 - 1/(2.*(joint_counts[0,0] + joint_counts[0,1]))
    if false_alarm_rate == 1:
        false_alarm_rate = 1 - 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))
    if hit_rate == 0:
        hit_rate = 1/(2*(joint_counts[0,0] + joint_counts[0,1]))
    if false_alarm_rate == 0:
        false_alarm_rate = 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))

    return(hit_rate - false_alarm_rate, hit_rate + false_alarm_rate - 1)


def sensitivity_and_bias_rescale(joint_counts):
    joint_counts = numpy.array(joint_counts, dtype='float')
    hit_rate = joint_counts[0,0] / (joint_counts[0,0] + joint_counts[0,1])
    false_alarm_rate = joint_counts[1,0] / (joint_counts[1,0] + joint_counts[1,1])
    if hit_rate == 1:
        hit_rate = 1 - 1/(2.*(joint_counts[0,0] + joint_counts[0,1]))
    if false_alarm_rate == 1:
        false_alarm_rate = 1 - 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))
    if hit_rate == 0:
        hit_rate = 1/(2*(joint_counts[0,0] + joint_counts[0,1]))
    if false_alarm_rate == 0:
        false_alarm_rate = 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))

    return((hit_rate - false_alarm_rate + 1.) / 2., (hit_rate + false_alarm_rate) / 2.)


def hit_rates(joint_counts):
    joint_counts = numpy.array(joint_counts)
    hit_rate_1 = (joint_counts[0,0] + 1.) / (joint_counts[0,0] + joint_counts[0,1] + 2.)
    hit_rate_2 = (joint_counts[1,1] + 1.) / (joint_counts[1,0] + joint_counts[1,1] + 2.)

    if hit_rate_1 == 1:
        hit_rate_2 = 1 - 1/(2.*(joint_counts[0,0] + joint_counts[0,1]))
    if hit_rate_2 == 1:
        hit_rate_2 = 1 - 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))

    if hit_rate_1 == 0:
        hit_rate_1 = 1/(2*(joint_counts[0,0] + joint_counts[0,1]))
    if hit_rate_2 == 0:
        hit_rate_2 = 1/(2*(joint_counts[1,0] + joint_counts[1,1]))
    return(hit_rate_1, hit_rate_2)


def percent_correct_on_accurate_vs_inaccurate(count_list):
    if type(count_list) == tuple:
        all_joint_counts, correct_joint_counts = count_list
        per_correct_on_accurate = (correct_joint_counts[0,0] + correct_joint_counts[1,1] + 1.) / (all_joint_counts[0,0] + all_joint_counts[1,1] + 2.)
        per_correct_on_inaccurate = (correct_joint_counts[1,0] + correct_joint_counts[0,1] + 1. ) / (all_joint_counts[1,0] + all_joint_counts[0,1] + 2.)
        return(per_correct_on_accurate, per_correct_on_inaccurate)
    else:
        return(percent_correct(count_list))


def percent_correct_on_accurate_vs_inaccurate_2(count_list):
    if type(count_list) == tuple:
        correct_joint_counts, incorrect_joint_counts = count_list
        per_correct_on_accurate = (correct_joint_counts[0,0] + correct_joint_counts[1,1] + 1. ) / (correct_joint_counts[0,0] + correct_joint_counts[1,1] + incorrect_joint_counts[0,0] + incorrect_joint_counts[1,1] + 2.)
        per_correct_on_inaccurate = (correct_joint_counts[1,0] + correct_joint_counts[0,1] + 1.) / (correct_joint_counts[1,0] + correct_joint_counts[0,1] + incorrect_joint_counts[1,0] + incorrect_joint_counts[0,1] + 2.)
        return(per_correct_on_accurate, per_correct_on_inaccurate)
    else:
        return(percent_correct(count_list))


# FINISH
def auc(binary_list, probs_list):
    from sklearn.metrics import roc_auc_score
    try:
        return((roc_auc_score(binary_list, probs_list),))
    except:
        return((5,))
    pass

#################
# Prepping Data #
#################


def reformat_neuron_list(neuron_list):
    reformatted_dict = {}
    for neuron in neuron_list:
        neuron, cell = neuron.split('-')
        cell = eval(cell)[0]
        if neuron in reformatted_dict:
            reformatted_dict[neuron].append(cell)
        else:
            reformatted_dict[neuron] = [cell]
    return(reformatted_dict)


def check_if_in_list(group, neuron_list):
    animal, cells = group.split('-')
    cells = eval(cells)
    try:
        select_cells = neuron_list[animal]
    except:
        return(False)
    for cell in cells:
        if cell in select_cells:
            return(True)
    return(False)


def check_how_many_in_list(group, neuron_list):
    animal, cells = group.split('-')
    cells = eval(cells)
    number = 0
    try:
        select_cells = neuron_list[animal]
    except:
        return(number)
    for cell in cells:
        if cell in select_cells:
            number += 1
    return(number)


def process_counts(counts_iterator, process_function, null_counts=None, test_function=None, boundaries=None, exclude={}, correct='all', num_neurons=None, excep=False):
    AC = {'stim' : {'mean' : [], 'err' : [], 'sig' : []},
        'action' : {'mean' : [], 'err' : [], 'sig' : []},
        'labels' : []
        }
    PFC = {'stim' : {'mean' : [], 'err' : [], 'sig' : []},
        'action' : {'mean' : [], 'err' : [], 'sig' : []},
        'labels' : []
        }
    exclude = reformat_neuron_list(exclude)
    if boundaries != None:
        AC['stim']['per_above'] = []
        AC['action']['per_above'] = []
        PFC['stim']['per_above'] = []
        PFC['action']['per_above'] = []
    for animal, counts in counts_iterator.iteritems():
        for neuron in counts.keys():
            if (num_neurons is not None) and (len(neuron) not in num_neurons):
                continue
            if (animal in exclude) and len(set(neuron).intersection(set(exclude[animal]))) != 0:
                print("Excluding {0}-{1}".format(animal, neuron))
                continue
            try:
                if correct == 'all':
                    stim_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['all_stimulus'] ]
                    action_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['all_action'] ]
                elif correct == 'correct':
                    stim_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['correct_stimulus'] ]
                    action_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['correct_action'] ]
                elif correct == 'incorrect':
                    stim_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['incorrect_stimulus'] ]
                    action_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['incorrect_action'] ]
                elif correct == 'all_correct':
                    stim_list = [ process_function((all_joint_counts, correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(counts[neuron]['all_stimulus'], counts[neuron]['correct_stimulus'])]
                    action_list = [ process_function((all_joint_counts,  correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(counts[neuron]['all_action'], counts[neuron]['correct_action']) ]
                elif correct == 'incorrect_correct':
                    stim_list = [ process_function((all_joint_counts, correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(counts[neuron]['correct_stimulus'], counts[neuron]['incorrect_stimulus'])]
                    action_list = [ process_function((all_joint_counts,  correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(counts[neuron]['correct_action'], counts[neuron]['incorrect_action']) ]
                elif correct == 'probs_complete':
                    stim_list = [ process_function((stimulus == 'T')[probs[:,0] != 0], probs[:,0][probs[:,0] != 0]) for stimulus, probs in zip(counts[neuron]['stimulus'], counts[neuron]['stimulus_probs']) ]
                    action_list = [ process_function((action == 'NP')[probs[:,0] != 0], probs[:,0][probs[:,0] != 0]) for action, probs in zip(counts[neuron]['action'], counts[neuron]['action_probs']) ]

                if null_counts is not None:
                    null_count = null_counts[animal]
                    if correct == 'all':
                        null_stim_list = [ process_function(joint_counts) for joint_counts in null_count[neuron]['all_stimulus'] ]
                        null_action_list = [ process_function(joint_counts) for joint_counts in null_count[neuron]['all_action'] ]
                    elif correct == 'correct':
                        null_stim_list = [ process_function(joint_counts) for joint_counts in null_count[neuron]['correct_stimulus'] ]
                        null_action_list = [ process_function(joint_counts) for joint_counts in null_count[neuron]['correct_action'] ]
                    elif correct == 'incorrect':
                        null_stim_list = [ process_function(all_joint_counts - correct_joint_counts) for all_joint_counts, correct_joint_counts in zip(null_count[neuron]['all_stimulus'], null_count[neuron]['correct_stimulus'])]
                        null_action_list = [ process_function(all_joint_counts - correct_joint_counts) for all_joint_counts, correct_joint_counts in zip(null_count[neuron]['all_action'], null_count[neuron]['correct_action']) ]
                    elif correct == 'both':
                        null_stim_list = [ process_function((all_joint_counts, correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(null_count[neuron]['all_stimulus'], null_count[neuron]['correct_stimulus'])]
                        null_action_list = [ process_function((all_joint_counts,  correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(null_count[neuron]['all_action'], null_count[neuron]['correct_action']) ]
                    elif correct == 'probs_complete':
                        null_stim_list = [ process_function((stimulus == 'T')[probs[:,0] != 0], probs[:,0][probs[:,0] != 0]) for stimulus, probs in zip(null_count[neuron]['stimulus'], null_count[neuron]['stimulus_probs']) ]
                        null_action_list = [ process_function((action == 'NP')[probs[:,0] != 0], probs[:,0][probs[:,0] != 0]) for action, probs in zip(null_count[neuron]['action'], null_count[neuron]['action_probs'])]

                stim_list_conditions = zip(*stim_list)
                stim_means = [ numpy.nanmean(condition) for condition in stim_list_conditions ]
                stim_err = [ numpy.nanstd(condition) / numpy.sqrt(len(condition)) for condition in stim_list_conditions ]
                if null_counts is not None:
                    null_stim_list_conditions = zip(*null_stim_list)
                    stim_sig = [test_function(null_stim, stim) for null_stim, stim in zip(null_stim_list_conditions, stim_list_conditions)]

                action_list_conditions = zip(*action_list)
                action_means = [ numpy.nanmean(condition) for condition in action_list_conditions ]
                action_err = [ numpy.nanstd(condition) / numpy.sqrt(len(condition)) for condition in action_list_conditions ]
                if null_counts is not None:
                    null_action_list_conditions = zip(*null_action_list)
                    action_sig = [test_function(null_action, action) for null_action, action in zip(null_action_list_conditions, action_list_conditions)]

                if boundaries != None:
                    stim_above = [ numpy.sum(numpy.array(stims) > boundaries[i]) / float(len(stims)) for i, stims in enumerate(stim_list_conditions) ]
                    action_above = [ numpy.sum(numpy.array(actions) > boundaries[i]) / float(len(actions)) for i, actions in enumerate(action_list_conditions) ]

                if animal[0] == 'A':
                    AC['stim']['mean'].append(stim_means)
                    AC['stim']['err'].append(stim_err)
                    AC['action']['mean'].append(action_means)
                    AC['action']['err'].append(action_err)
                    AC['labels'].append("{0}-{1}".format(animal, neuron))
                    if null_counts is not None:
                        AC['stim']['sig'].append(stim_sig)
                        AC['action']['sig'].append(action_sig)
                    if boundaries != None:
                        AC['stim']['per_above'].append(stim_above)
                        AC['action']['per_above'].append(action_above)

                else: #if animal[0] == 'P':
                    PFC['stim']['mean'].append(stim_means)
                    PFC['stim']['err'].append(stim_err)
                    PFC['action']['mean'].append(action_means)
                    PFC['action']['err'].append(action_err)
                    PFC['labels'].append("{0}-{1}".format(animal, neuron))
                    if null_counts is not None:
                        PFC['stim']['sig'].append(stim_sig)
                        PFC['action']['sig'].append(action_sig)
                    if boundaries != None:
                        PFC['stim']['per_above'].append(stim_above)
                        PFC['action']['per_above'].append(action_above)
            except:
                if excep:
                    raise
                else:
                    #raise
                    print("problem with animal {0}, neuron {1}".format(animal, neuron))
    for region in [AC, PFC]:
        for task_var in ['stim', 'action']:
            for key, values in region[task_var].iteritems():
                region[task_var][key] = zip(*values)
    return(AC, PFC)

def process_counts_chris(counts_iterator, process_function, null_counts=None, test_function=None, boundaries=None, exclude={}, correct='all', num_neurons=None, excep=False):
    AC = {'block' : {'mean' : [], 'err' : [], 'sig' : []},
        'labels' : []
        }
    PFC = {'block' : {'mean' : [], 'err' : [], 'sig' : []},
        'labels' : []
        }
    exclude = reformat_neuron_list(exclude)
    if boundaries != None:
        AC['block']['per_above'] = []
        PFC['block']['per_above'] = []
    for animal, counts in counts_iterator.iteritems():
        for neuron in counts.keys():
            if (num_neurons is not None) and (len(neuron) not in num_neurons):
                continue
            if (animal in exclude) and len(set(neuron).intersection(set(exclude[animal]))) != 0:
                print("Excluding {0}-{1}".format(animal, neuron))
                continue
            try:
                if correct == 'all':
                    block_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['all_block'] ]
                elif correct == 'correct':
                    stim_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['correct_stimulus'] ]
                    action_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['correct_action'] ]
                elif correct == 'incorrect':
                    stim_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['incorrect_stimulus'] ]
                    action_list = [ process_function(joint_counts) for joint_counts in counts[neuron]['incorrect_action'] ]
                elif correct == 'all_correct':
                    stim_list = [ process_function((all_joint_counts, correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(counts[neuron]['all_stimulus'], counts[neuron]['correct_stimulus'])]
                    action_list = [ process_function((all_joint_counts,  correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(counts[neuron]['all_action'], counts[neuron]['correct_action']) ]
                elif correct == 'incorrect_correct':
                    stim_list = [ process_function((all_joint_counts, correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(counts[neuron]['correct_stimulus'], counts[neuron]['incorrect_stimulus'])]
                    action_list = [ process_function((all_joint_counts,  correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(counts[neuron]['correct_action'], counts[neuron]['incorrect_action']) ]
                elif correct == 'probs_complete':
                    stim_list = [ process_function((stimulus == 'T')[probs[:,0] != 0], probs[:,0][probs[:,0] != 0]) for stimulus, probs in zip(counts[neuron]['stimulus'], counts[neuron]['stimulus_probs']) ]
                    action_list = [ process_function((action == 'NP')[probs[:,0] != 0], probs[:,0][probs[:,0] != 0]) for action, probs in zip(counts[neuron]['action'], counts[neuron]['action_probs']) ]

                if null_counts is not None:
                    null_count = null_counts[animal]
                    if correct == 'all':
                        null_block_list = [ process_function(joint_counts) for joint_counts in null_count[neuron]['all_block'] ]
                    elif correct == 'correct':
                        null_stim_list = [ process_function(joint_counts) for joint_counts in null_count[neuron]['correct_stimulus'] ]
                        null_action_list = [ process_function(joint_counts) for joint_counts in null_count[neuron]['correct_action'] ]
                    elif correct == 'incorrect':
                        null_stim_list = [ process_function(all_joint_counts - correct_joint_counts) for all_joint_counts, correct_joint_counts in zip(null_count[neuron]['all_stimulus'], null_count[neuron]['correct_stimulus'])]
                        null_action_list = [ process_function(all_joint_counts - correct_joint_counts) for all_joint_counts, correct_joint_counts in zip(null_count[neuron]['all_action'], null_count[neuron]['correct_action']) ]
                    elif correct == 'both':
                        null_stim_list = [ process_function((all_joint_counts, correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(null_count[neuron]['all_stimulus'], null_count[neuron]['correct_stimulus'])]
                        null_action_list = [ process_function((all_joint_counts,  correct_joint_counts)) for all_joint_counts, correct_joint_counts in zip(null_count[neuron]['all_action'], null_count[neuron]['correct_action']) ]
                    elif correct == 'probs_complete':
                        null_stim_list = [ process_function((stimulus == 'T')[probs[:,0] != 0], probs[:,0][probs[:,0] != 0]) for stimulus, probs in zip(null_count[neuron]['stimulus'], null_count[neuron]['stimulus_probs']) ]
                        null_action_list = [ process_function((action == 'NP')[probs[:,0] != 0], probs[:,0][probs[:,0] != 0]) for action, probs in zip(null_count[neuron]['action'], null_count[neuron]['action_probs'])]

                block_list_conditions = zip(*block_list)
                block_means = [ numpy.nanmean(condition) for condition in block_list_conditions ]
                block_err = [ numpy.nanstd(condition) / numpy.sqrt(len(condition)) for condition in block_list_conditions ]
                if null_counts is not None:
                    null_block_list_conditions = zip(*null_block_list)
                    block_sig = [test_function(null_block, block) for null_block, block in zip(null_block_list_conditions, block_list_conditions)]

                if boundaries != None:
                    block_above = [ numpy.sum(numpy.array(block) > boundaries[i]) / float(len(block)) for i, block in enumerate(block_list_conditions) ]

                ulabel = "{0}-{1}".format(animal, neuron[0])
                key = "{0}-{1}".format(animal, neuron)
                if is_auditory(ulabel):
                    AC['block']['mean'].append(block_means)
                    AC['block']['err'].append(block_err)
                    AC['labels'].append(key)
                    if null_counts is not None:
                        AC['block']['sig'].append(block_sig)
                    if boundaries != None:
                        AC['block']['per_above'].append(block_above)

                else:
                    PFC['block']['mean'].append(block_means)
                    PFC['block']['err'].append(block_err)
                    PFC['labels'].append(key)
                    if null_counts is not None:
                        PFC['block']['sig'].append(block_sig)
                    if boundaries != None:
                        PFC['block']['per_above'].append(block_above)
            except:
                if excep:
                    raise
                else:
                    #raise
                    print("problem with animal {0}, neuron {1}".format(animal, neuron))
    for region in [AC, PFC]:
        for task_var in ['block']:
            for key, values in region[task_var].iteritems():
                region[task_var][key] = zip(*values)
    return(AC, PFC)

def load_certainty(counts, exclude=None):
    probs_dict = {}
    for animal in counts.keys():
        probs_dict[animal] = {}
        for neuron in counts[animal].keys():
            if "{0}-{1}".format(animal, neuron) in exclude:
                print("{0}-{1} excluded".format(animal, neuron))
                continue
            probs_dict[animal][neuron] = {}
            probs_dict[animal][neuron]['all_stimulus'] = counts[animal][neuron]['all_stimulus_probs']
            probs_dict[animal][neuron]['all_action'] = counts[animal][neuron]['all_action_probs']
            probs_dict[animal][neuron]['correct_stimulus'] = counts[animal][neuron]['correct_stimulus_probs']
            probs_dict[animal][neuron]['correct_action'] = counts[animal][neuron]['correct_action_probs']
            probs_dict[animal][neuron]['incorrect_stimulus'] = counts[animal][neuron]['incorrect_stimulus_probs']
            probs_dict[animal][neuron]['incorrect_action'] = counts[animal][neuron]['incorrect_action_probs']
    return(probs_dict)


def load_certainty_chris(counts, exclude=None):
    probs_dict = {}
    for animal in counts.keys():
        probs_dict[animal] = {}
        for neuron in counts[animal].keys():
            if "{0}-{1}".format(animal, neuron) in exclude:
                print("{0}-{1} excluded".format(animal, neuron))
                continue
            probs_dict[animal][neuron] = {}
            probs_dict[animal][neuron]['all_block'] = counts[animal][neuron]['all_block_probs']
    return(probs_dict)


def calc_certainty(counts, exclude=None):
    probs_dict = {}
    for animal in counts.keys():
        probs_dict[animal] = {}
        for neuron in counts[animal].keys():
            probs_dict[animal][neuron] = {}
            stimulus_probs_total = []
            action_probs_total = []
            correct_action_probs_total = []
            correct_stimulus_probs_total = []
            incorrect_stimulus_probs_total = []
            incorrect_action_probs_total = []
            if not "{0}-{1}".format(animal, neuron) in exclude:
                current_neuron = counts[animal][neuron]
                if current_neuron == {}:
                    continue
                for correct, stimulus, action, stimulus_probs, action_probs in zip(current_neuron['correct'], current_neuron['stimulus'], current_neuron['action'], current_neuron['stimulus_probs'], current_neuron['action_probs']):
                    stimulus_probs = numpy.array(stimulus_probs)
                    action_probs = numpy.array(action_probs)
                    stimulus_probs_total.append(numpy.array([stimulus_probs[stimulus == 'T'].sum(axis = 0),
                        stimulus_probs[stimulus == 'F'].sum(axis = 0)]))
                    correct_stimulus_probs_total.append(numpy.array([stimulus_probs[(stimulus == 'T')*(correct == '+')].sum(axis = 0),
                        stimulus_probs[(stimulus == 'F')*(correct=='+')].sum(axis = 0)]))
                    incorrect_stimulus_probs_total.append(numpy.array([stimulus_probs[(stimulus == 'T')*(correct == '-')].sum(axis = 0),
                        stimulus_probs[(stimulus == 'F')*(correct=='-')].sum(axis = 0)]))
                    action_probs_total.append(numpy.array([action_probs[action == 'NP'].sum(axis = 0),
                        action_probs[action == 'W'].sum(axis = 0)]))
                    correct_action_probs_total.append(numpy.array([action_probs[(action == 'NP')*(correct=='+')].sum(axis = 0),
                        action_probs[(action == 'W')*(correct=='+')].sum(axis = 0)]))
                    incorrect_action_probs_total.append(numpy.array([action_probs[(action == 'NP')*(correct=='-')].sum(axis = 0),
                        action_probs[(action == 'W')*(correct=='-')].sum(axis = 0)]))
                probs_dict[animal][neuron]['all_stimulus'] = stimulus_probs_total
                probs_dict[animal][neuron]['all_action'] = action_probs_total
                probs_dict[animal][neuron]['correct_stimulus'] = correct_stimulus_probs_total
                probs_dict[animal][neuron]['correct_action'] =  correct_action_probs_total
                probs_dict[animal][neuron]['incorrect_stimulus'] = incorrect_stimulus_probs_total
                probs_dict[animal][neuron]['incorrect_action'] =  incorrect_action_probs_total
            else:
                print("{0}-{1} excluded".format(animal, neuron))
    return(probs_dict)


def calc_correct(counts):
    new_counts = {}
    for animal, neuron_data in counts.iteritems():
        new_counts[animal] = {}
        for neuron, data in neuron_data.iteritems():
            new_counts[animal][neuron] = data
            try:
                correct_stimulus_counts_total = []
                correct_action_counts_total = []
                incorrect_stimulus_counts_total = []
                incorrect_action_counts_total = []
                for correct, stimulus, action, stimulus_counts, action_counts in zip(data['correct'], data['stimulus'], data['action'], data['stimulus_choices'], data['action_choices']):
                    stimulus_counts = numpy.array(stimulus_counts)
                    action_counts= numpy.array(action_counts)
                    correct_stimulus_counts_total.append(numpy.array([stimulus_counts[(stimulus == 'T')*(correct == '+')].sum(axis = 0),
                        stimulus_counts[(stimulus == 'F')*(correct=='+')].sum(axis = 0)]))
                    correct_action_counts_total.append(numpy.array([action_counts[(action == 'NP')*(correct=='+')].sum(axis = 0),
                        action_counts[(action == 'W')*(correct=='+')].sum(axis = 0)]))
                    incorrect_stimulus_counts_total.append(numpy.array([stimulus_counts[(stimulus == 'T')*(correct == '-')].sum(axis = 0),
                        stimulus_counts[(stimulus == 'F')*(correct=='-')].sum(axis = 0)]))
                    incorrect_action_counts_total.append(numpy.array([action_counts[(action == 'NP')*(correct=='-')].sum(axis = 0),
                        action_counts[(action == 'W')*(correct=='-')].sum(axis = 0)]))
                new_counts[animal][neuron]['correct_stimulus'] = correct_stimulus_counts_total
                new_counts[animal][neuron]['correct_action'] =  correct_action_counts_total
                new_counts[animal][neuron]['incorrect_stimulus'] = incorrect_stimulus_counts_total
                new_counts[animal][neuron]['incorrect_action'] =  incorrect_action_counts_total
            except:
                print("Problem with {0}, {1}".format(animal, neuron))
    return(new_counts)


def find_significance(values, labels, sig=.001):
    is_significant = []
    not_significant = []
    for label, value in zip(labels, values):
        if value <= sig:
            is_significant.append(label)
            print(label + ' is significant')
        else:
            not_significant.append(label)
    return(set(is_significant), set(not_significant))


def sig_filter(sig_list, sig_threshhold):
    sig_mask = numpy.array(sig_list) < sig_threshhold
    return(sig_mask)


######################
# Plotting Functions #
######################


def plot_scatter_w_density(x_points, y_points, colors = 'b', markers = 'o', filled = True, linestyles='-', labels = None, error_bars = None, bounds = ([0,1], [0,1]), guides = (.5,.5), x_ticks = [0, .5, 1.], y_ticks = [0, .5, 1.], format_percent = True, fig_size = (3, 3)):

    # Check if we have multiple data sets
    if not isinstance(x_points[0], Number):
        multiple = True
        num_sets = len(x_points)
        if type(linestyles) is not list:
            linestyles = num_sets*[linestyles]
        if type(colors) is not list:
            colors = num_sets*[colors]
        if type(markers) is not list:
            markers = num_sets*[markers]
        if type(filled) is not list:
            filled = num_sets*[filled]
        if type(error_bars) is not list:
            error_bars = num_sets*[error_bars]
    else:
        multiple = False

    # Set up the plot
    x_guide, y_guide = guides
    x_bounds, y_bounds = bounds

    fig = plt.figure(figsize=fig_size)
    fig_h, fig_w = fig_size
    density_x = .2 / fig_h
    density_y = .2 / fig_w
    left, width = 0.3 / fig_w, (fig_w - 0.3)/ fig_w - density_y
    bottom, height = 0.3 / fig_h, (fig_w - 0.3) / fig_h - density_x

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height, width, density_x]
    rect_histy = [left + width, bottom, density_y, height]

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx, sharex = axScatter)
    axHisty = plt.axes(rect_histy, sharey = axScatter)

    axHistx.plot([x_guide, x_guide], [-10, 50], color = 'k', linewidth=.5, zorder=0)
    axHisty.plot([-10, 50], [y_guide, y_guide], color = 'k', linewidth=.5, zorder=0)
    axScatter.plot([x_guide,x_guide], [-10, 10], color = 'k', linewidth=.5, zorder=0)
    axScatter.plot([-10, 10], [y_guide, y_guide], color = 'k', linewidth=.5, zorder=0)

    X = numpy.arange(x_bounds[0]-1,x_bounds[1]+1,.001)
    Y = numpy.arange(y_bounds[0]-1,y_bounds[1]+1,.001)

    def plot_points(x_points, y_points, color, marker, filled, linestyle, error_bar):
        assert len(x_points) == len(y_points)
        if len(x_points) == 0:
            return(None, 0, 0)
        elif len(x_points) == 1:
            x_max = 0
            y_max = 0
        elif len(x_points) > 1:
            x_density = kde.gaussian_kde(x_points)
            y_density = kde.gaussian_kde(y_points)
            axHistx.plot(X, x_density(X), color = color, linestyle = linestyle, linewidth = LINEWIDTH)
            x_max = numpy.max(x_density(X))
            axHisty.plot(y_density(Y), Y, color = color, linestyle = linestyle, linewidth = LINEWIDTH)
            y_max = numpy.max(y_density(Y))

        if not filled:
            facecolor = 'white'
        else:
            facecolor = color
        if error_bar != None:
            points = axScatter.errorbar(x_points, y_points, xerr = error_bar[0] , yerr = error_bar[1], mfc = facecolor, mec = color, ecolor = color, fmt = marker, mew = LINEWIDTH, ms = SIZE, capsize = 2)
        else:
            points = axScatter.scatter(x_points, y_points, facecolor = facecolor, edgecolor = color, linewidth = LINEWIDTH, marker = marker, s = SIZE**2)
        return(points, x_max, y_max)

    if multiple:
        points = []
        x_max = 0
        y_max = 0
        for i, x_data_set, y_data_set, color, marker, filled, linestyle, error_bar in zip(range(num_sets), x_points, y_points, colors, markers, filled, linestyles, error_bars):
            points_temp, x_max_temp, y_max_temp = plot_points(x_data_set, y_data_set, color, marker, filled, linestyle, error_bar = error_bar)
            if points_temp is not None:
                points.append(points_temp)
            if linestyle is not 'None':
                if x_max_temp > x_max:
                    x_max = x_max_temp
                if y_max_temp > y_max:
                    y_max = y_max_temp
    else: # not multiple
        points, x_max, y_max = plot_points(x_points, y_points, colors, markers, filled, linestyles, error_bar = error_bars)

    axScatter.set_xticks(x_ticks)
    axScatter.set_yticks(y_ticks)
    if format_percent:
        axScatter.set_xticklabels([PERCENT_FORMAT_STRING.format(int(tick*100)) for tick in x_ticks])
        axScatter.set_yticklabels([PERCENT_FORMAT_STRING.format(int(tick*100)) for tick in x_ticks])
    axScatter.set_xlim(x_bounds)
    axScatter.set_ylim(y_bounds)


    axHistx.spines['top'].set_visible(False)
    axHistx.spines['right'].set_visible(False)
    axHistx.spines['left'].set_visible(False)
    axHistx.spines['bottom'].set_visible(False)
    axHistx.yaxis.set_major_formatter(plt.NullFormatter())
    axHistx.set_yticks([])
    axHistx.xaxis.set_ticks_position("none")
    plt.setp(axHistx.get_yticklabels(), visible = False)
    plt.setp(axHistx.get_xticklabels(), visible = False)
    axHistx.set_ylim([-.002, x_max*1.1])

    axHisty.spines['top'].set_visible(False)
    axHisty.spines['right'].set_visible(False)
    axHisty.spines['left'].set_visible(False)
    axHisty.spines['bottom'].set_visible(False)
    axHisty.xaxis.set_major_formatter(plt.NullFormatter())
    axHisty.set_xticks([])
    axHisty.yaxis.set_ticks_position("none")
    plt.setp(axHisty.get_yticklabels(), visible = False)
    plt.setp(axHisty.get_xticklabels(), visible = False)
    axHisty.set_xlim([-.002, y_max*1.1])

    if labels != None:
        if multiple:
            tooltips = []
            for point, label in zip(points, labels):
                if error_bar is not None:
                    point = point[0]
                tooltips.append(plugins.PointLabelTooltip(point, list(label)))
            for tooltip in tooltips:
                plugins.connect(fig, tooltip)
        else:
            tooltip = plugins.PointLabelTooltip(points[0], list(labels))
            plugins.connect(fig, tooltip)
    return(axScatter)


def plot_density(points, boundaries = None, colors = 'k', linestyles = '-', x_lim = [0,1], y_lim = None, guides = [], ticks = [0,.25,.5,.75,1.], y_ticks = None, format_percent = True,  fig_size = (6,2), show_quartiles = True):
    # Check if we have multiple data sets
    if not isinstance(points[0], Number):
        multiple = True
        num_sets = len(points)
        if type(linestyles) is not list:
            linestyles = num_sets*[linestyles]
        if type(colors) is not list:
            colors = num_sets*[colors]
    else:
        multiple = False

    fig, ax = plt.subplots(figsize = fig_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([])
    if ticks is not None:
        ax.set_xticks(ticks)
        if format_percent:
            ax.set_xticklabels([PERCENT_FORMAT_STRING.format(int(tick*100)) for tick in ticks])

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    else:
        ax.spines['left'].set_visible(False)

    for guide in guides:
        ax.axvline(guide, color = 'k')

    X = numpy.arange(x_lim[0], x_lim[1], .001)

    def plot_points(points, color, linestyle, boundaries):
        num_points = len(points)
        if boundaries is not None:
            bottom, top = boundaries
            diff = top - bottom
            points_copy = [ ((diff*i+bottom) + (datum-bottom)) for i in xrange(--10, 12, 2) for datum in points] + [ ((diff*i+bottom) - (datum - bottom)) for i in xrange(-10, 12, 2) for datum in points]
            density = kde.gaussian_kde(points_copy)
        else:
            density = kde.gaussian_kde(points)

        ax.plot(X, num_points*density(X), color = color, linestyle = linestyle, linewidth = LINEWIDTH)

        for point in points:
            ax.plot([point, point], [0, num_points*density(point)], color = color,  linewidth = LINEWIDTH / 3., alpha = .2)
        if show_quartiles:
            median = numpy.percentile(points, 50)
            Q1 = numpy.percentile(points, 25)
            Q3 = numpy.percentile(points, 75)
            ax.plot([median, median], [0, num_points*density(median)], color = color,  linewidth = LINEWIDTH)
            ax.plot([Q1, Q1], [0, num_points*density(Q1)], color = color, linewidth = LINEWIDTH/2.)
            ax.plot([Q3, Q3], [0, num_points*density(Q3)], color = color, linewidth = LINEWIDTH/2.)
        return(numpy.max(num_points*density(X)))

    if multiple:
        points = []
        max_value = 0
        for i, data_set, color, linestyle in zip(range(num_sets), points, colors, linestyles):
            max_temp = plot_points(data_set, color, linestyle, boundaries)
            if max_temp > max_value:
                max_value = max_temp
    else: # not multiple
        max_value = plot_points(points, colors, linestyles, boundaries)

    if y_lim is None:
        ax.set_ylim([0,max_value*1.15])
    else:
        ax.set_ylim([0,y_lim*1.15])
    return(ax)


def plot_legend(point_styles, labels = None, points_filled = None, colors = 'k', line_styles = None):
    num_items = len(point_styles)
    if line_styles is not None:
        width = .6
        ax_width = 1.5
    else:
        width = .2
        ax_width = 1
    fig = plt.figure(figsize = (width, .3*num_items))
    ax = fig.add_axes([0,0,ax_width,1])
    plt.axis('off')
    spacing = 1. / (num_items + 1.)
    for i, point_style, color in zip(range(num_items), point_styles, colors):
        if points_filled is not None:
            if points_filled[i] is True:
                facecolor = color
            else:
                facecolor = 'white'
        ax.scatter([.5], [spacing*(i+1)], marker = point_style, edgecolor = color, facecolor = facecolor, s = SIZE**2)
        if line_styles is not None:
            ax.plot([.75,1.5], [spacing*(i+1), spacing*(i+1)], color = color, linestyle = line_styles[i], linewidth = LINEWIDTH)
    ax.set_xlim([0,ax_width])
    ax.set_ylim([0,1])


def plot_histogram(values, low, high, bin_size, sig_filter=None, color='k', linestyle='solid', non_sig_color='white', sig_color='0.8', lock_to=None, guides=None, ticks=[0,.25,.5,.75,1.], y_max=None, y_ticks=None, format_percent=True, fig_size= (2,1)):
    #fig, ax = plt.subplots(figsize = fig_size)
    fig = plt.figure(figsize=fig_size)
    fig_w, fig_h = fig_size
    rect = [.3 / fig_w, .2 / fig_h, (fig_w - .4 ) / fig_w, (fig_h - .3) / fig_h]
    ax = plt.axes(rect)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks(ticks)
    if format_percent:
        ax.set_xticklabels([PERCENT_FORMAT_STRING.format(int(100*tick)) for tick in ticks])

    if guides is not None:
        for guide in guides:
            ax.axvline(guide, color = 'k')

    #new_high = numpy.ceil((high - low) / bin_size)*bin_size + low
    if lock_to is None:
        bins = numpy.arange(low, high, bin_size)
    else:
        bins = numpy.concatenate((numpy.arange(lock_to - bin_size, low - bin_size, -bin_size)[::-1], numpy.arange(lock_to, high, bin_size)))

    def make_counts(values):
        counts = [numpy.sum((values >= l)*(values < (l+bin_size))) for l in bins]
        return(counts)

    if sig_filter is not None:
        counts = make_counts(values)
        sig_counts = make_counts(numpy.array(values)[sig_filter])
        for left, count in zip (bins, counts):
            ax.add_patch(Rectangle((left, 0), bin_size, count, facecolor = non_sig_color, alpha = .5, linestyle = linestyle, edgecolor=color))
        for left, count in zip (bins, sig_counts):
            ax.add_patch(Rectangle((left, 0), bin_size, count, facecolor = sig_color, linestyle = linestyle, edgecolor=color))
    else:
        counts = make_counts(values)
        for left, count in zip (bins, counts):
            ax.add_patch(Rectangle((left, 0), bin_size, count, facecolor = non_sig_color, linestyle = linestyle, edgecolor=color))

    ax.set_xlim([low, bins[-1]+bin_size])
    if y_max is None:
        ax.set_ylim([0,max(counts)*1.15])
    else:
        ax.set_yticks(y_ticks)
        ax.set_ylim([0,y_max])
    return(ax, counts)


def plot_boxplot(locations, values, markers='o', colors='k', centercolors='k', linestyles='solid', filled=False, draw_type='jitter', labels=None, y_ticks=None, y_ticklabels=None, y_bounds=None, width=.9, draw_sig=False, sig_pairs=None, sig_levels=None, spacer=.02, fig_size=(3, 2), size=SIZE, mean=False):
    if centercolors is None:
        if type(colors) is not list:
            centercolors = colors
        else:
            centercolors = []
            for color in colors:
                if type(color) is list:
                    centercolors.append(color[0])
                else:
                    centercolors.append(color)

    if isinstance(locations, list):
        multiple = True
        num_sets = len(locations)
        if type(linestyles) is not list:
            linestyles = num_sets * [linestyles]
        if type(colors) is not list:
            colors = num_sets * [colors]
        if type(markers) is not list:
            markers = num_sets * [markers]
        if type(filled) is not list:
            filled = num_sets * [filled]
        if type(centercolors) is not list:
            centercolors = num_sets * [centercolors]
    else:
        multiple = False

    fig_w, fig_h = fig_size
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([.3 / fig_w, .2 / fig_h, (fig_w - .4) / fig_w, (fig_h - .3) / fig_h])
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks(locations)
    ax.set_xlim([min(locations) - width, max(locations) + width])

    def draw_boxplot(location, (smallest, Q1, median, Q3, largest), color, filled, linestyle):
        if draw_type is 'box':
            ax.add_patch(Rectangle((location - width / 2, Q1), width,  median - Q1, facecolor = 'none', edgecolor = color, linestyle = linestyle))
            ax.add_patch(Rectangle((location - width / 2, median), width,  Q3 - median, facecolor = 'none', edgecolor = color, linestyle = linestyle))
            ax.plot([location - width / 2., location + width / 2.], [median, median], linewidth = 2, color = color)
            ax.plot([location, location], [Q3, largest], linewidth = 1, color = color, linestyle = linestyle)
            ax.plot([location, location], [Q1, smallest], linewidth = 1, color = color, linestyle = linestyle)
        elif draw_type is 'jitter':
            if filled:
                facecolor = color
            else:
                facecolor = 'white'
            ax.plot([location, location], [Q1, Q3], linewidth=4, color='white', zorder=8)
            ax.scatter([location], [median], s=1.5 * size**2, facecolor='white', color='white', linewidth=4, zorder=8)
            ax.plot([location, location], [Q1, Q3], linewidth=2, color=color, zorder=10)
            ax.scatter([location], [median], s=1.5 * size**2, facecolor=facecolor, color=color, linewidth=2, zorder=11)
        elif draw_type is 'bar':
            if fill is False:
                current_color='white'
            else:
                current_color=color
            ax.bar(location, median, align='center', color=current_color)
            ax.plot([location, location], [Q1, Q3], linewidth = 1, color = color, linestyle = linestyle)
            return(Q1, Q3)
        return(smallest, largest)


    def draw_jitter_plot(location, width, values, marker, color, filled):
        #num_values = len(values)
        #x_values = .6*width*(rand(num_values) - .5) + location
        if len(values) > 1:
            jitter_width = kde.gaussian_kde(values)
            kde_norm = numpy.max([jitter_width(x) for x in numpy.arange(numpy.min(values), numpy.max(values), .001)])
            x_values = [.8*width*jitter_width(value)*(rand() - .5) / kde_norm + location for value in values]
        else:
            x_values = [location]
        if filled is True:
            facecolor = color
        else:
            facecolor = 'white'
        ax.scatter(x_values, values, facecolor=facecolor, color=color, linewidth=LINEWIDTH, alpha=.4, s=size**2)
        pass

    def draw_significance(location_1, location_2, height, level):
        ax.plot([location_1, location_1], [height, height + spacer], color = 'k', linewidth=.5)
        ax.plot([location_2, location_2], [height, height + spacer], color = 'k', linewidth=.5)
        ax.plot([location_1, location_2], [height + spacer, height + spacer], color = 'k', linewidth=.5)
        if level == 0:
            level_text = 'n.s.'
        else:
            level_text = level*'*'
        ax.text((location_1 + location_2) / 2., height + 1.3*spacer , level_text, ha='center')

    if multiple:
        smallest_a = 0
        largest_a = 0
        space_stack = []
        for location, value, color, centercolor, marker, linestyle, fill in zip(locations, values, colors, centercolors, markers, linestyles, filled):
            if len(value) == 0:
                continue
            smallest = numpy.min(value)
            largest =  numpy.max(value)
            if mean is False:
                Q1 = numpy.percentile(value, 25)
                median = numpy.median(value)
                Q3 = numpy.percentile(value, 75)
            else:
                stddev = numpy.std(value)
                median = numpy.mean(value)
                Q1 = median - stddev
                Q3 = median + stddev
            smallest, largest = draw_boxplot(location, (smallest, Q1, median, Q3, largest), centercolor, fill, linestyle)
            space_stack.append(largest)
            if largest > largest_a:
                largest_a = largest
            if smallest > smallest_a:
                smallest_a = smallest
            if draw_type is 'jitter':
                draw_jitter_plot(location, width, value, marker, color, fill)
    else:
        smallest = numpy.min(values)
        Q1 = numpy.percentile(values, 25)
        median = numpy.median(values)
        Q3 = numpy.percentile(values, 75)
        largest =  numpy.max(values)
        smallest_a, largest_a = draw_boxplot(locations,  (smallest, Q1, median, Q3, largest))
        if draw_type is 'jitter':
            draw_jitter_plot(locations, width, values, marker, colors, centercolors, fill)

    if draw_sig:
        MULTI = 2
        num_sig = len(sig_pairs)
        space_stack = []
        for value in values:
            if len(value) != 0:
                space_stack.append(numpy.max(value))
        if y_ticks is not None:
            ax.set_ylim([min(y_ticks), max(y_ticks) + spacer*MULTI*num_sig])
        for pair, level in zip(sig_pairs, sig_levels):
            height = numpy.max([space_stack[i] for i in range(pair[0], pair[1]+1)])
            lift = pair[0] + numpy.argmax([space_stack[i] for i in range(pair[0], pair[1]+1)])
            space_stack[lift] = space_stack[lift] + MULTI*spacer
            draw_significance(locations[pair[0]], locations[pair[1]], height + 2*spacer, level)
    if y_ticks is not None:
        ax.set_ylim([min(y_ticks), max(y_ticks)])
        ax.yaxis.set_ticks(y_ticks)
        if y_ticklabels is not None:
            ax.yaxis.set_ticklabels(y_ticklabels)
    if y_bounds is not None:
        ax.set_ylim(y_bounds)
    if labels is None:
        ax.xaxis.set_ticklabels(len(locations)*[''])
    return(ax)


def plot_receptive_field(receptive_field, range=(.25, 64), tick_labels=[.5, 1, 2, 4, 8, 16, 32], figsize=(3, 2)):
    freqs, rates, variation = zip(*receptive_field)
    rates = numpy.array(rates)
    freqs = numpy.array(freqs)
    variation = numpy.array(variation)

    index = numpy.argsort(freqs)
    rates = rates[index]
    freqs = freqs[index]
    variation = variation[index]
    print(freqs)
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(freqs, rates + variation, rates - variation, color='.8')
    ax.plot(freqs, rates, color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # remove unneeded ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xscale('log', basex=2)
    ax.set_xticks(freqs)
    ax.set_xticklabels(tick_labels)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    #ax.set_xlim(range)


def plot_bean(data_1, data_2, colors = ['k', 'k'], alpha = .2, ylim = [0,1], yticks = [0,.25,.5,.75,1.], boundaries = None):
    fig = plt.figure(figsize=(4, 6))
    rect = [.2, .05, .7, .90]
    ax = plt.axes(rect)
    ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.set_yticks(yticks)
    #ax.set_yticklabels(["{:.0%}".format(tick) for tick in yticks])
    ax.set_ylim([ylim[0] - .001, ylim[1] + .001])

    if boundaries is not None:
        bottom, top = boundaries
        diff = top - bottom
        data_1_copy = [ ((diff*i+bottom) + (datum-bottom)) for i in xrange(--10, 12, 2) for datum in data_1] + [ ((diff*i+bottom) - (datum - bottom)) for i in xrange(-10, 12, 2) for datum in data_1]
        data_2_copy = [ ((diff*i+bottom) + (datum-bottom)) for i in xrange(-10, 12, 2) for datum in data_2] + [ ((diff*i+bottom) - (datum - bottom)) for i in xrange(-10, 12, 2) for datum in data_2]
        density_1 = kde.gaussian_kde(data_1_copy, bw_method = diff / 300.)
        density_2 = kde.gaussian_kde(data_2_copy, bw_method = diff / 300.)
    else:
        data_1_copy = data_1[:]
        data_2_copy = data_2[:]

        density_1 = kde.gaussian_kde(data_1_copy)
        density_2 = kde.gaussian_kde(data_2_copy)

    X = numpy.arange(ylim[0],ylim[1],.01)
    ax.plot(-density_1(X), X, color = colors[0], linewidth = LINEWIDTH)
    ax.plot(density_2(X), X, color = colors[1], linewidth = LINEWIDTH)

    for data in data_1:
        ax.plot([-density_1(data),0], [data, data], color = colors[0], alpha = alpha)
    data_1_average = numpy.median(data_1)
    ax.plot([-density_1(data_1_average),0], [data_1_average, data_1_average], color = colors[0], linewidth = 3)

    for data in data_2:
        ax.plot([0, density_2(data)], [data , data], color = colors[1], alpha = alpha)
    data_2_average = numpy.median(data_2)
    ax.plot([0,density_2(data_2_average)], [data_2_average, data_2_average], color = colors[1], linewidth = 3)

    return(ax)


def line_of_unity_scatter(x_points, y_points, colors = 'b', markers = 'o',filled = True, labels = None, error_bars = None, bounds = ([0,1], [0,1]), guides = (.5,.5), x_ticks = [0, .5, 1.], y_ticks = [0, .5, 1.], size=SIZE, format_percent = True, fig_size = (3,3), unity = True):

    # Check if we have multiple data sets
    if not isinstance(x_points[0], Number):
        multiple = True
        num_sets = len(x_points)
        if type(colors) is not list:
            colors = num_sets*[colors]
        if type(markers) is not list:
            markers = num_sets*[markers]
        if type(filled) is not list:
            filled = num_sets*[filled]
        if type(error_bars) is not list:
            error_bars = num_sets*[error_bars]
    else:
        multiple = False

    # Set up the plot
    x_guide, y_guide = guides
    x_bounds, y_bounds = bounds

    fig_w, fig_h = fig_size
    fig = plt.figure(figsize=fig_size)
    axScatter = fig.add_axes([.3 / fig_w, .3 / fig_h, (fig_w - .4) / fig_w, (fig_h - .4) / fig_h])

    axScatter.plot([x_guide,x_guide], [-10, 10], color = 'k', linewidth=.5, zorder=0)
    axScatter.plot([-10, 10], [y_guide, y_guide], color = 'k', linewidth=.5, zorder=0)
    if unity:
        axScatter.plot([-10, 10], [-10, 10], color = 'k', linewidth=.5, zorder=0)

    def plot_points(x_points, y_points, color, marker, filled, error_bar):
        if not filled:
            facecolor = 'white'
        else:
            facecolor = color
        if error_bar is not None:
            points = axScatter.errorbar(x_points, y_points, xerr = error_bar[0] , yerr = error_bar[1], mfc = facecolor, mec = color, ecolor = color, fmt = marker, mew = LINEWIDTH, ms = size, capsize = 2)
        else:
            points = axScatter.scatter(x_points, y_points, facecolor = facecolor, edgecolor = color, linewidth = LINEWIDTH, marker = marker, s = size**2)
        x_max = max(x_points)
        y_max = max(y_points)
        return(points, x_max, y_max)

    if multiple:
        points = []
        x_max = 0
        y_max = 0
        for i, x_data_set, y_data_set, color, marker, filled, error_bar in zip(range(num_sets), x_points, y_points, colors, markers, filled, error_bars):
            points_temp, x_max_temp, y_max_temp = plot_points(x_data_set, y_data_set, color, marker, filled, error_bar = error_bar)
            points.append(points_temp)
            if x_max_temp > x_max:
                x_max = x_max_temp
            if y_max_temp > y_max:
                y_max = y_max_temp
    else: # not multiple
        points, x_max, y_max = plot_points(x_points, y_points, colors, markers, filled, error_bars)

    axScatter.set_xticks(x_ticks)
    axScatter.set_yticks(y_ticks)
    if format_percent:
        axScatter.set_xticklabels([PERCENT_FORMAT_STRING.format(int(tick*100)) for tick in x_ticks])
        axScatter.set_yticklabels([PERCENT_FORMAT_STRING.format(int(tick*100)) for tick in x_ticks])
    axScatter.set_xlim(x_bounds)
    axScatter.set_ylim(y_bounds)


    if labels is not None:
        if multiple:
            tooltips = []
            for point, label in zip(points, labels):
                if error_bar is not None:
                    point = point[0]
                tooltips.append(plugins.PointLabelTooltip(point, list(label)))
            for tooltip in tooltips:
                plugins.connect(fig, tooltip)
        else:
            tooltip = plugins.PointLabelTooltip(points, list(labels))
            plugins.connect(fig, tooltip)
    return(axScatter)


def plot_bar_comparison(AC, PFC, labels = ["accurate", "inaccurate"]):
    TICKS = [0, .25, .5, .75, 1.0]

    AC_behavior_on_disagree, AC_behavior_on_agree = zip(*AC)
    PFC_behavior_on_disagree, PFC_behavior_on_agree = zip(*PFC)

    fig = plt.figure(figsize=(4,6))
    axes_rect = [.20, .1, .75, .85]
    ax = plt.axes(axes_rect)

    bAC_disag = ax.bar([-.4],
            [numpy.mean(AC_behavior_on_disagree)], width = .4,
            color = AC_COLOR, alpha = .2, edgecolor = AC_COLOR, label = 'AC on disagree')

    bAC_ag = ax.bar([0],
            [numpy.mean(AC_behavior_on_agree)], width = .4,
            color = AC_COLOR, alpha = .5, edgecolor = AC_COLOR, label = 'AC on agree')

    bPFC_disag = ax.bar([.6],
            [ numpy.mean(PFC_behavior_on_disagree) ], width = .4,
            color = PFC_COLOR, alpha = .2, edgecolor = PFC_COLOR, label = 'PFC on disagree')

    bPFC_ag = ax.bar([1],
            [numpy.mean(PFC_behavior_on_agree) ], width = .4,
            color = PFC_COLOR, alpha = .5, edgecolor = PFC_COLOR, label = 'PFC on agree')

    ax.errorbar([.2, -.2], [numpy.mean(AC_behavior_on_agree), numpy.mean(AC_behavior_on_disagree)],
                yerr = [ numpy.std(AC_behavior_on_agree) / numpy.sqrt(len(AC_behavior_on_agree)),
                        numpy.std(AC_behavior_on_disagree) / numpy.sqrt(len(AC_behavior_on_disagree))],
                fmt = '.', color = AC_COLOR)
    ax.errorbar([1.2, .8], [numpy.mean(PFC_behavior_on_agree), numpy.mean(PFC_behavior_on_disagree)],
                yerr = [ numpy.std(PFC_behavior_on_agree) / numpy.sqrt(len(PFC_behavior_on_agree)),
                        numpy.std(PFC_behavior_on_disagree) / numpy.sqrt(len(PFC_behavior_on_disagree))],
                fmt = '.', color = PFC_COLOR)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()   # remove unneeded ticks
    ax.get_yaxis().tick_left()
    ax.set_xticks([0,1])
    ax.set_xticklabels(['AC', 'FC'])
    ax.set_xlim([-.5, 1.5])

    ax.set_yticks([0,.25,.5,.75,1.])
    ax.set_yticklabels(["{:.0%}".format(tick) for tick in TICKS])
    #leg = ax.legend([bAC_ag, bAC_disag ], ["", ""], loc = 9)
    #leg.draw_frame(False)

    leg2 = ax.legend([ bAC_ag,  bAC_disag, bPFC_ag, bPFC_disag ], ["",  "", labels[0], labels[1]], ncol = 2, columnspacing = 0, loc = 1)
    leg2.draw_frame(False)
    #plt.gca().add_artist(leg)
    return(ax)


def plot_trial(trace, trace_a, SIs, SI_as, times, spike_set, nosepoke, smooth_window = 50, times_c = numpy.arange(0, 2.5,.001), bounds = None, show_SI = False, withhold = True):
    fig = plt.figure(figsize = (3, 2))

    trace_c = [ trace[times > time][0] for time in times_c if time < times[-1]]
    trace_c = numpy.array(trace_c + (len(times_c) - len(trace_c))*[trace_c[-1]] )

    trace_ac = [ trace_a[times > time][0] for time in times_c if time < times[-1] ]
    trace_ac = numpy.array(trace_ac + (len(times_c) - len(trace_ac))*[trace_c[-1]] )

    SI_c = numpy.zeros(len(times_c))
    SI_ac = numpy.zeros(len(times_c))
    index = [ numpy.nonzero((times_c < time))[0][-1] for time in times[:-1] ]
    SI_c[index] = SIs
    SI_ac[index] = SI_as

    raster_rect = [.1, .1, .8, .1]
    global_rect = [.1, .1, .8, .9]

    if show_SI:
        stim_SI_rect =  [.1, .60, .8, .15]
        stim_probability_rect =  [.1, .80, .8, .15]
        action_SI_rect = [.1, .15, .8, .15]
        action_probability_rect = [.1, .35, .8, .15]
    else:
        stim_probability_rect =  [.1, .60, .8, .30]
        action_probability_rect = [.1, .20, .8, .30]

    axGlobal = plt.axes(global_rect)
    axRaster = plt.axes(raster_rect, axisbg = 'none')
    axStimProb = plt.axes(stim_probability_rect, axisbg = 'none')
    axActionProb = plt.axes(action_probability_rect, axisbg = 'none')

    if show_SI:
        axStimSI = plt.axes(stim_SI_rect, axisbg = 'none')
        axActionSI = plt.axes(action_SI_rect, axisbg = 'none')


    if show_SI:
        all_axes = [axGlobal, axRaster, axStimProb, axStimSI, axActionProb, axActionSI]
    else:
        all_axes = [axGlobal, axRaster, axStimProb, axActionProb]

    for axis in all_axes:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['right'].set_visible(False)

    if bounds is not None:
        minimum, maximum = bounds
    else:
        minimum = times_c[0]
        maximum = times_c[-1]

    axGlobal.set_xlim(minimum,maximum)
    axGlobal.axvspan(0,.1, color = 'k', linewidth = 1.5, alpha= .1)

    if withhold == False:
        axGlobal.axvline(nosepoke, color = 'k', linewidth = 1.5, linestyle = 'solid')
    elif withhold == True:
        axGlobal.axvline(nosepoke, color = 'k', linewidth = 1.5, linestyle = 'dotted')


    #axStimProb.axvspan(0,.1, color = 'k', linewidth = 1.5, alpha= .1)
    #axStimSI.axvspan(0,.1, color = 'k', linewidth = 1.5, alpha = .1)
    #axActionProb.axvspan(0,.1, color = 'k', linewidth = 1.5, alpha = .1)
    #axActionSI.axvspan(0,.1, color = 'k', linewidth = 1.5, alpha = .1)
    #axRaster.axvspan(0,.1, color = 'k', linewidth = 1.5, alpha = .1)

    #axStimProb.axvline(nosepoke, color = 'k', linewidth = 1.5)
    #axStimSI.axvline(nosepoke, color = 'k', linewidth = 1.5)
    #axActionProb.axvline(nosepoke, color = 'k', linewidth = 1.5)
    #axActionSI.axvline(nosepoke, color = 'k', linewidth = 1.5)

    if show_SI:
        smooth_SI_a = smooth(numpy.array(SI_ac), window_len = smooth_window)
        max_SI_a = max(smooth_SI_a)
        min_SI_a = min(smooth_SI_a)
        axActionSI.plot(times_c, smooth_SI_a, color = 'k')

        smooth_SI = smooth(numpy.array(SI_c), window_len = smooth_window)
        max_SI = max(smooth_SI)
        min_SI = min(smooth_SI)
        axStimSI.plot(times_c, smooth_SI, color = 'k')

    num_neurons = float(len(spike_set))
    for i, spikes in enumerate(spike_set):
        axRaster.scatter(spikes, [i / num_neurons]*len(spikes), linewidth = 1.5, marker = "|", color = 'k', s = 300 / num_neurons)

    smooth_stim_prob_NP = smooth(numpy.array(trace_ac[:,0]), window_len = smooth_window)
    smooth_stim_prob_W = smooth(numpy.array(trace_ac[:,1]), window_len = smooth_window)
    actionProb = axActionProb.plot(times_c,  smooth_stim_prob_NP, color = 'green', label = "G", linewidth = 2)
    actionProb = axActionProb.plot(times_c,  smooth_stim_prob_W, color = 'purple', label = "NG", linewidth = 2)
    #actionProb = axActionProb.imshow([ smooth(numpy.array(trace_a[:,'NP'])[1:], window_len = 10) ],
    #           extent = [trace.axes['time'][1],trace.axes['time'][-1], 0, 1 ],
    #           aspect = 'auto', cmap = 'PiYG', vmin = 0, vmax = 1)

    smooth_stim_prob_T = smooth(numpy.array(trace_c[:,0]), window_len = smooth_window)
    smooth_stim_prob_F = smooth(numpy.array(trace_c[:,1]), window_len = smooth_window)
    stimProb = axStimProb.plot(times_c,  smooth_stim_prob_T, color = 'r', label = "T", linewidth = 2)
    stimProb = axStimProb.plot(times_c,  smooth_stim_prob_F, color = 'b', label = "NT", linewidth = 2)
    #stimProb = axStimProb.imshow([ smooth(numpy.array(trace[:,'T'])[1:], window_len = 10) ],
    #           extent = [trace.axes['time'][1],trace.axes['time'][-1], 0, 1 ],
    #           aspect = 'auto', cmap = 'bwr', vmin = 0, vmax = 1)

    axStimProb.set_xlim(minimum, maximum)
    axActionProb.set_xlim(minimum, maximum)
    axRaster.set_xlim(minimum, maximum)
    if show_SI:
        axStimSI.set_xlim(minimum, maximum)
        axActionSI.set_xlim(minimum, maximum)

    axRaster.get_xaxis().set_visible(True)
    axRaster.get_xaxis().tick_bottom()
    axRaster.spines['bottom'].set_visible(True)

    axStimProb.get_yaxis().set_visible(True)
    axStimProb.get_yaxis().tick_left()
    axStimProb.spines['left'].set_visible(True)
    StimProb_labels = [0,.5,1]
    axStimProb.set_yticks( StimProb_labels )
    axStimProb.set_yticklabels(["{:.0%}".format(label) for label in StimProb_labels ])
    axStimProb.set_ylim(-.05, 1.05)

    axActionProb.get_yaxis().set_visible(True)
    axActionProb.get_yaxis().tick_left()
    axActionProb.spines['left'].set_visible(True)
    ActionProb_labels = [0,.5,1]
    axActionProb.set_yticks( StimProb_labels )
    axActionProb.set_yticklabels(["{:.0%}".format(label) for label in ActionProb_labels ])
    axActionProb.set_ylim(-.05, 1.05)

    if show_SI:
        axActionSI.get_yaxis().set_visible(True)
        axActionSI.get_yaxis().tick_left()
        axActionSI.spines['left'].set_visible(True)
        ActionSI_labels = [min_SI_a,0,max_SI_a]
        axActionSI.set_yticks( ActionSI_labels )
        axActionSI.set_yticklabels(["{: .3f}".format(label) for label in ActionSI_labels ])

        axStimSI.get_yaxis().set_visible(True)
        axStimSI.get_yaxis().tick_left()
        axStimSI.spines['left'].set_visible(True)
        StimSI_labels = [min_SI,0,max_SI]
        axStimSI.set_yticks(StimSI_labels)
        axStimSI.set_yticklabels(["{: .3f}".format(label) for label in StimSI_labels ])

    #TICKS = [0, .5, 1]
    #barStim = fig.colorbar(stimProb, cax = axStimProbColor, ticks = TICKS)
    #barStim.ax.set_yticklabels(["F", "", "T"])
    #barAction = fig.colorbar(actionProb, cax = axActionProbColor, ticks = TICKS)
    #barAction.ax.set_yticklabels(["W", "", "NP" ])

    # axRaster.set_xlabel("Time from tone onset (s)")
    # axActionProb.set_ylabel("$\\rm{p(choice)}$", size = 'large')
    # axStimProb.set_ylabel("$\\rm{p(stimulus)}$", size = 'large')
    # if show_SI:
    #     axStimSI.set_ylabel("SI (bits)")
    #     axActionSI.set_ylabel("SI (bits)")

    # Legend
    # leg1 = axStimProb.legend(frameon = False, fontsize = 12, loc = (1, .2))
    # leg2 = axActionProb.legend(frameon = False, fontsize = 12, loc = (1, .2))
    # plt.gca().add_artist(leg1)
    return(axStimProb)


def plot_trial_2(traces, times, spike_set, nosepoke, colors=[['r', 'b']], smooth_window=50, times_c=numpy.arange(0, 2.5,.001), bounds=None, show_SI=False, withhold=True, fig_size=(3, 1.5)):
    fig = plt.figure(figsize = fig_size)
    fig_w, fig_h = fig_size
    num_traces = len(traces)
    num_spikes = len(spike_set)

    trace_c = []
    for trace, tim in zip(traces, times):
        trace_c_temp = [ trace[tim > time][0] for time in times_c if time < tim[-1]]
        trace_c_temp = numpy.array(trace_c_temp + (len(times_c) - len(trace_c_temp))*[trace_c_temp[-1]])
        trace_c.append(trace_c_temp)

    l_mar = .4
    r_mar = .1
    t_mar = .1
    b_mar = .2

    raster_h = num_spikes* .2
    spacer = .2
    raster_rect = [l_mar/fig_w, (b_mar)/fig_h, 1.- (l_mar + r_mar)/fig_w, raster_h/fig_h]

    global_rect = [l_mar/fig_w, b_mar/fig_h, 1.- (l_mar + r_mar)/fig_w, 1. - (b_mar + t_mar)/fig_h]

    trace_h = (fig_h - t_mar - b_mar - raster_h) / num_traces - spacer
    prob_rects = [[l_mar/fig_w, (b_mar + raster_h + (i+1)*spacer + i*trace_h)/fig_h, 1.- (l_mar + r_mar)/fig_w, trace_h/fig_h] for i in range(num_traces)]

    axGlobal = fig.add_axes(global_rect)
    axRaster = fig.add_axes(raster_rect, axisbg = 'none')
    axsProb = [fig.add_axes(rect, axisbg = 'none') for rect in prob_rects]

    all_axes = [axGlobal, axRaster]
    all_axes.extend(axsProb)

    for axis in all_axes:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['right'].set_visible(False)

    if bounds is not None:
        minimum, maximum = bounds
    else:
        minimum = times_c[0]
        maximum = times_c[-1]

    axGlobal.set_xlim(minimum, maximum)
    axGlobal.axvspan(0,.1, color = 'k', linewidth = 1.5, alpha= .1)
    if withhold == False:
        axGlobal.axvline(nosepoke, color = 'k', linewidth = 1.5, linestyle = 'solid')
    elif withhold == True:
        axGlobal.axvline(nosepoke, color = 'k', linewidth = 1.5, linestyle = 'dotted')

    num_neurons = float(len(spike_set))
    for i, spikes in enumerate(spike_set):
        axRaster.scatter(spikes, [i + .5
]*len(spikes), linewidth = 1, marker = "|", color = 'k', s = 160)
    axRaster.set_ylim(0, num_neurons)
    axRaster.set_xlim(minimum, maximum)
    axRaster.get_xaxis().set_visible(True)
    axRaster.get_xaxis().tick_bottom()
    axRaster.spines['bottom'].set_visible(True)

    for trace, color, ax in zip(trace_c, colors, axsProb):
        smooth_stim_prob_T = smooth(numpy.array(trace[:,0]), window_len = smooth_window)
        smooth_stim_prob_F = smooth(numpy.array(trace[:,1]), window_len = smooth_window)
        stimProb = ax.plot(times_c,  smooth_stim_prob_T, color = color[0], label = "T", linewidth = 2)
        stimProb = ax.plot(times_c,  smooth_stim_prob_F, color =color[1], label = "NT", linewidth = 2)
    for ax in axsProb:
        ax.set_xlim(minimum, maximum)
        ax.get_yaxis().set_visible(True)
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_visible(True)
        StimProb_labels = [0,.5, 1]
        ax.set_yticks(StimProb_labels)
        ax.set_yticklabels(["{:.0%}".format(label) for label in StimProb_labels ])
        ax.set_ylim(0, 1.0)

    return()


def plot_raster(responses, nosepokes=None, cond_variable=None, condition=None, color='k', marker='o', num_trials=None, guides=True, draw_hist=True, stim_highlight=[0, .1], beginning_time=-1, ending_time=2.5, bin_width=.050, num_labels=2., width_per_sec=.7, trial_per_in=120, lw=1, size=.25):
    if cond_variable is not None:
        filtered_responses = responses[cond_variable == condition]
        if nosepokes is not None:
            filtered_nosepokes = nosepokes[cond_variable == condition]
    else:
        filtered_responses = responses
        if nosepokes is not None:
            filtered_nosepokes = nosepokes

    if num_trials is None:
        num_trials = len(filtered_responses)
    num_trials_rate = len(filtered_responses)

    fig_h = .3 + float(num_trials) / trial_per_in
    if draw_hist:
        fig_h += .5
    fig_w = width_per_sec*(ending_time - beginning_time) + .4
    fig = plt.figure(figsize=(fig_w, fig_h))

    if draw_hist:
        raster_rect = [.3 / fig_w, .7 / fig_h, (fig_w - .4) / fig_w, (fig_h - .8) / fig_h]
        gap_rect = [.3 / fig_w, .6 / fig_h, (fig_w - .4) / fig_w, .1 / fig_h]
        raster_hist_rect = [.3 / fig_w, .2 / fig_h, (fig_w - .4) / fig_w, 0.4 / fig_h]
    else:
        raster_rect = [.3 / fig_w, .2 / fig_h, (fig_w - .4) / fig_w, (fig_h - .3) / fig_h]
    ax = plt.axes(raster_rect)
    if draw_hist:
        ax_gap = plt.axes(gap_rect)
        ax1 = plt.axes(raster_hist_rect)

    for i in range(num_trials):
        ax.scatter(filtered_responses[i], [i + .5] * len(filtered_responses[i]), marker="o", s=size , color=color, edgecolor='none')
        if nosepokes is not None:
            ax.scatter(filtered_nosepokes[i], [i + .5], marker=marker, s=size*5, color='k')
    ax.set_xlim([beginning_time, ending_time])
    ax.set_ylim([0, num_trials])
    # ax.set_ylabel('Trial #')
    # ax.axvspan(START, START+WINDOW, facecolor = 'g', alpha=.2)

    steps = int(numpy.floor(num_trials / (10. * num_labels)) * 10.)
    ax.set_yticks(range(0, num_trials + 1, steps))
    if guides:
        for guide in numpy.arange(0, num_trials, 10):
            ax.axhline(guide, color='k', alpha=.1, linewidth=.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    if stim_highlight is not None:
        ax.axvspan(stim_highlight[0], stim_highlight[1], facecolor='k', alpha=.2)

    # total_responses = [item for subarray in filtered_responses[0:NUM_TRIALS] for item in subarray ]
    if draw_hist:
        ax.set_xticks([])
        ax.spines['bottom'].set_color('none')
        total_responses = numpy.array([item for subarray in filtered_responses for item in subarray])
        total_responses = total_responses[(total_responses > beginning_time)*(total_responses < ending_time)]
        diff = ending_time - beginning_time
        #response_copy = [((diff * i + beginning_time) + (datum - beginning_time)) for i in xrange(-10, 12, 2) for datum in total_responses] + [((diff * i + beginning_time) - (datum - beginning_time)) for i in xrange(-10, 12, 2) for datum in total_responses]
        response_copy = numpy.hstack((total_responses, -total_responses + 2*beginning_time + 2*diff, -total_responses + 2*beginning_time))
        kde = KDEUnivariate(response_copy)
        kde.fit(bw=.02)
        # pdf = kde.gaussian_kde(response_copy)
        # pdf.set_bandwidth(.02)
        average_rate = len(total_responses) / (num_trials_rate * (ending_time - beginning_time))
        pre_factor = len(total_responses) / (num_trials_rate * quad(lambda x: kde.evaluate([x])[0], beginning_time, ending_time)[0])
        firing_rate = lambda x: pre_factor * kde.evaluate([x])[0]
        t = sp.linspace(beginning_time, ending_time, 200)
        ax1.set_xlim([beginning_time, ending_time])
        rates = numpy.array([firing_rate(i) for i in t])
        ax1.plot(t, rates, color=color, linewidth=lw)
        cur_ylim = ax1.get_ylim();

        #ax1.set_ylim([0, numpy.max(rates)*1.2])
        #ax1.set_yticks(numpy.arange(0,  numpy.max(rates)*1.2, int(average_rate)))
        ax_gap.set_xlim([beginning_time, ending_time])
        ax_gap.spines['top'].set_color('none')
        ax_gap.spines['bottom'].set_color('none')
        ax_gap.spines['right'].set_color('none')
        ax_gap.spines['left'].set_color('none')
        ax_gap.set_xticks([])
        ax_gap.set_yticks([])
        if stim_highlight is not None:
            ax1.axvspan(stim_highlight[0], stim_highlight[1], facecolor='k', alpha=.2)
            ax_gap.axvspan(stim_highlight[0], stim_highlight[1], facecolor='k', alpha=.2)

        bins = numpy.arange(beginning_time, ending_time + bin_width, bin_width)
        num_bins = len(bins) - 1
        counts, bin_edges = numpy.histogram(total_responses, bins=bins)
        counts = numpy.array(counts) / (num_trials_rate * bin_width)
        ax1.bar(bin_edges[1:] - bin_width/2., counts, bin_width, color=color, edgecolor=color, linewidth=.5, alpha=.2)
        #ax1.hist(total_responses, bins=int(num_bins), normed=40, alpha=.2, color=color)
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')
        #ax1.spines['left'].set_color('none')
        ax1.xaxis.set_ticks_position('bottom')

        # if stim_highlight == None:
        #     ax1.set_xlabel("Time before nosepoke (s)")
        # else:
        #     ax1.set_xlabel("Time from tone onset (s)")


def plot_certainty(AC, PFC, var = 0, bins = numpy.arange(0,1.05,.05)):
    fig, ((axACstim, axPFCstim),(axACaction, axPFCaction)) = plt.subplots(nrows = 2, ncols= 2, sharex = True, sharey = True, figsize = (8,4))

    axACstim.hist(AC['stim']['per_above'][var], bins = bins, color = AC_COLOR)
    axACaction.hist(AC['action']['per_above'][var], bins = bins, color = AC_COLOR)
    axACstim.set_ylabel("stimulus")
    axACstim.set_title("AC")
    axACaction.set_ylabel("action")
    axACaction.set_xticklabels(["{:.0%}".format(tick) for tick in axACaction.get_xticks()])
    axACaction.set_xlabel("significance (%)")

    axPFCstim.hist(PFC['stim']['per_above'][var], bins = bins, color = PFC_COLOR)
    axPFCaction.hist(PFC['action']['per_above'][var], bins = bins, color = PFC_COLOR)
    axPFCstim.set_title("FC")
    axPFCaction.set_xticklabels(["{:.0%}".format(tick) for tick in axACaction.get_xticks()])
    axPFCaction.set_xlabel("significance (%)")

    plt.tight_layout()
    pass
