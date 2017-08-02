import numpy
from numpy.random import choice, random
from .PSTH import PSTH
from .single_ISI import calc_ISIs
from .multiple_ISI import calc_ISIs_multiple
from .tools import flatten

################################################
# Using the ISI distributions to generate data #
################################################

def generate_spike_train_using_ISI(ISIs, (min_time, max_time)):
    current_time = min_time
    spike_times = []
    while current_time < max_time:
        if current_time != 0:
            spike_times.append(current_time.copy())
        current_time = current_time + choice(ISIs)
    return(numpy.array(spike_times))


def generate_random_data_using_ISI(spikes, num_trials, durations):
    ISIs = flatten([calc_ISIs(spike_train)[0] for spike_train in spikes])
    responses = [generate_spike_train_using_ISI(ISIs, duration) for duration in durations]
    return(numpy.array(responses))


def generate_random_data_within_class_using_ISI(spikes, variables, durations):
    ISIs = {condition: flatten([calc_ISIs(spike_train) for spike_train in spike_list]) for condition, spike_list in spikes.iteritems()}
    responses = []
    for var, duration in zip(variables, durations):
        responses.append(generate_spike_train_using_ISI(ISIs[var], duration))
    return(numpy.array(responses))


#################################################
# Using the PSTH distributions to generate data #
#################################################


def generate_spike_train_using_PSTH(psth, (min_time, max_time), resolution=.001):
    time_bins = numpy.arange(min_time, max_time, resolution)
    spikes = []
    for time in time_bins:
        prob = psth(time) * resolution
        if random() <= prob:
            spikes.append(time)
    return(numpy.array(spikes))


def generate_random_data_using_PSTH(spikes, num_trials, durations):
    firing_rate = PSTH(spikes, mirror=True, bw=None)
    responses = [generate_spike_train_using_PSTH(firing_rate, duration) for duration in durations]
    return(numpy.array(responses))


def generate_random_data_within_class_using_PSTH(spikes, variables, durations):
    firing_rates = {condition: PSTH(spike_list, mirror=True, bw=None) for condition, spike_list in spikes.iteritems()}
    responses = []
    for var, duration in zip(variables, durations):
        responses.append(generate_spike_train_using_PSTH(firing_rates[var], duration))
    return(numpy.array(responses))
