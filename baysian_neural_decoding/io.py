import numpy
import h5py
from copy import deepcopy

TRIAL_DURATION = 10
PRE_TRIAL_DURATION = 2.5

############
# File I/O #
############
# The Plexon .nex file was imported into Octave using the readexFile.m
# script and then saved as a HD5 file using Ocatve's
# "save -hdf5 file_name.hd5 variable". This was then imported into
# python using h5py.
#
# In that format, the commands below should work. This is obviously
# ideosyncratic to our system.


def extract_spikes(hd5_file, neuron_num=0):
    """Extracts the spiking data from the hdf5 file. Returns an array of
    spike times.

    Keyword arguments:
    neuron_num -- the index of the neuron you would like to access.
    """
    with h5py.File(hd5_file, "r+") as f:
        neuron_list = f['NF']['value']['neurons']['value']
        if len(neuron_list) <= 10:
            neuron_str = "_" + str(neuron_num)
        else:
            neuron_str = "_" + "0" * (2 - len(str(neuron_num))) + str(neuron_num)
        timestamps = numpy.array(neuron_list[neuron_str]['value']['timestamps']['value'][0])
    return(timestamps)


def extract_events(hd5_file):
    """Extracts the timestamps of the events stored in the hdf5 file."""
    events = {}
    with h5py.File(hd5_file, "r+") as f:
        event_list = f['NF']['value']['events']['value']
        for key in event_list.iterkeys():
            if key == 'dims':
                continue
            name = event_list[key]['value']['name']['value']
            # The hdf5 that results file contains the names not as strings
            # but as an array of integers which code for the string in
            # ASCII format.
            name_str = ''.join(map(chr, name))
            try:
                timestamps = numpy.array(event_list[key]['value']['timestamps']['value'][0])
            except:
                timestamps = numpy.array([], dtype='float64')
            events[name_str] = timestamps
    return(events)


def load_events_spikes_script(neuron_num=0, spike_files=None, event_files=None, exception=None, variables=None, **kwargs):
    """Extracts spikes and events
    """
    event_set = [extract_events(f) for f in event_files]
    if type(neuron_num) is int:
        if exception is not None:
            spike_set = [event_set[i][exception[neuron_num]] for i in range(len(event_set))]
        else:
            spike_set = [extract_spikes(f, neuron_num) for f in spike_files]
    elif (type(neuron_num) is list) or (type(neuron_num) is tuple):
        spike_set = []
        for num in neuron_num:
            if exception is not None:
                spike_set_temp = [event_set[i][exception[num]] for i in range(len(event_set))]
            else:
                spike_set_temp = [extract_spikes(f, num) for f in spike_files]
            spike_set.append(spike_set_temp)
    return(event_set, spike_set)


######################
# Basic Calculations #
######################


def create_complete_table(event_set, spike_set, variable_maps, pre_trial_duration=PRE_TRIAL_DURATION, trial_duration=TRIAL_DURATION, stim_variables=['T', 'F'], action_variables=['NPT', 'NPF']):
    assert len(action_variables) == len(stim_variables)
    stimuli = []
    stimuli_time = []
    actions = []
    correctness = []
    nose_pokes = []
    num_neurons = len(spike_set)
    responses = [[] for i in xrange(num_neurons)]
    all_trial_times = []
    for events, spikes_list, variable_map in zip(event_set, zip(*spike_set), variable_maps):
        for stim_variable, action_variable in zip(stim_variables, action_variables):
            trial_times = events[variable_map[stim_variable]]
            try:
                nose_poke_times = numpy.array(events[variable_map[action_variable]])
            except:
                nose_poke_times = None
            try:
                correct_times = events[variable_map[stim_variable + '+']]
            except:
                correct_times = None

            all_trial_times.extend(trial_times)

            for i, trial_time in enumerate(trial_times):
                stimuli.append(stim_variable)
                stimuli_time.append(trial_time)

                # finding correctness of trial
                if correct_times is None:
                    correctness.append('U')
                elif trial_time in correct_times:
                    correctness.append('+')
                else:
                    correctness.append('-')

                # finding nosepoke time
                if nose_poke_times is None:
                    nose_pokes.append(numpy.nan)  # This conflates no response with unkown response.  There is probably a better systemc
                    actions.append('U')
                else:
                    if i == len(trial_times) - 1:
                        index = (nose_poke_times > trial_time)
                    else:
                        index = (nose_poke_times > trial_time)*(nose_poke_times < trial_times[i+1])
                    if sum(index) == 0:
                        nose_poke_time = numpy.nan
                        actions.append('W')
                    else:
                        nose_poke_time = nose_poke_times[index][0] - trial_time
                        actions.append('NP')
                    nose_pokes.append(nose_poke_time)

                for i, spikes in enumerate(spikes_list):
                    current_response = spikes[(spikes >= trial_time - pre_trial_duration)*(spikes < (trial_time + trial_duration))] - trial_time
                    responses[i].append(deepcopy(current_response))

    sort_index = numpy.argsort(all_trial_times)
    for i, response in enumerate(responses):
        responses[i] = numpy.array(response)[sort_index]

    return(numpy.array(stimuli_time)[sort_index], numpy.array(stimuli)[sort_index], numpy.array(actions)[sort_index], numpy.array(nose_pokes)[sort_index], responses)
