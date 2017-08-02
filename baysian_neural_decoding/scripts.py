import numpy
import itertools
from copy import deepcopy
from numpy.random import permutation
from .controls import *
from .inference import *
from .tools import flatten, filter_times


def break_into_folds(index, num_folds):
    index = permutation(index)
    fold_size = float(len(index)) / num_folds
    folded_index = [index[int(round(i * fold_size)): int(round((i + 1) * fold_size))] for i in xrange(num_folds)]
    return(folded_index)


def break_into_subgroups(variable_dict):
    variable_combinations = itertools.product(*[values[0].keys() for values in variable_dict.values()])
    subgroups = {}
    for combination in variable_combinations:
        subgroups[combination] = numpy.product([values[1] == value for values, value in zip(variable_dict.values(), combination)], axis=0, dtype='bool')
    return(subgroups)


def format_into_subgroups(variable_dict):
    variable_names, values = variable_dict[variable_dict.keys()[0]]
    subgroups = {}
    for name in variable_names.keys():
        subgroups[name] = (values == name)
    return(subgroups)


def index_to_bool(index, size):
    bool_index = numpy.zeros(size).astype('bool')
    bool_index[index] = True
    return(bool_index)


def break_into_folds_balanced(variable_dict, num_folds, cross_groups=True):
    if cross_groups:
        subgroups = break_into_subgroups(variable_dict)
    else:
        subgroups = format_into_subgroups(variable_dict)
    num_trials = len(variable_dict.values()[0][1])
    trial_numbers = numpy.arange(num_trials)
    folded_subgroups = {}
    for combination, subgroup in subgroups.iteritems():
        folded_subgroups[combination] = break_into_folds(trial_numbers[subgroup], num_folds)
    # Ugly code, fix if possible
    merged_folds = [list(itertools.chain(*(list(fold[i]) for fold in folded_subgroups.values()))) for i in xrange(num_folds)]
    merged_folds_bool = [index_to_bool(fold, num_trials) for fold in merged_folds]
    return(merged_folds_bool)


def convert_responses(spike_list, response_function, response_times, multiple, **kwargs):
    if multiple:
        responses_all = []
        times_all = []
        spike_list = zip(*spike_list)
        for spikes, response_time in zip(spike_list, response_times):
            responses = []
            times = []
            for i, ind_spikes in enumerate(spikes):
                response, time = response_function(ind_spikes, response_time=response_time, **kwargs)
                responses_labeled = zip(len(response) * [i], response)
                responses.extend(deepcopy(responses_labeled))
                times.extend(deepcopy(time))
            sort_index = numpy.argsort(times)
            times = numpy.array(times)[sort_index]
            responses = numpy.array(responses)[sort_index]
            times_all.append(deepcopy(times))
            responses_all.append(deepcopy(responses))
        return(numpy.array(responses_all), numpy.array(times_all))
    else:
        responses = []
        times = []
        for spikes, response_time in zip(spike_list, response_times):
            response, time = response_function(spikes, response_time=response_time, **kwargs)
            responses.append(deepcopy(response))
            times.append(deepcopy(time))
        return(numpy.array(responses), numpy.array(times))


def contingency_table(counts, variable_map, variable_list):
    num_variables = len(variable_map)
    contingency_table = numpy.zeros((num_variables, num_variables))
    for value, index in variable_map.iteritems():
        value_filter = numpy.array(variable_list) == value
        contingency_table[index] = numpy.nansum(counts[value_filter], axis=0)
    return(contingency_table)


def spike_cutoff_script(spikes, multiple=False, spike_cutoff=3):
    if multiple:
        cutoff_filters = [numpy.array([len(spike) for spike in ind_spikes]) >= spike_cutoff for ind_spikes in spikes]
        cutoff_filter = numpy.product(cutoff_filters, axis=0, dtype="bool")
    else:
        cutoff_filter = numpy.array([len(spike) for spike in spikes]) >= spike_cutoff
    return(cutoff_filter)


def generate_random_data_script(spikes, trial_durations, variable_dict, multiple=False, within_class=False, use_PSTH=False, **kwargs):
    num_trials = len(spikes)
    # THIS MUST BE FIXED
    if within_class:
        spikes_sorted = {}
        subgroups = break_into_subgroups(variable_dict)
        if multiple:
            pass
            # if multiple:
            #     new_spikes = []
            #     for ind_spikes in zip(*spikes):
            #         for combination, index in subgroups:
            #             spikes_sorted[combination] = ind_spikes[index]
            #         variable_values = zip([values[1] for values in variables_dict.items()])
            #         ind_spikes_new = generate_random_data_within_class(spikes_sorted, variable_values, inference_times)
            #         new_spikes.append(deepcopy(ind_spikes_new))
            #     spikes = zip(*new_spikes)
        else:  # if not multiple
            for combination, index in subgroups.iteritems():
                spikes_sorted[combination] = numpy.array(spikes)[index]
            variable_values = zip(*[values[1] for values in variable_dict.values()])
            if use_PSTH:
                spikes = generate_random_data_within_class_using_PSTH(spikes_sorted, variable_values, trial_durations)
            else:  # not use PSTH
                spikes = generate_random_data_within_class_using_ISI(spikes_sorted, variable_values, trial_durations)
    else:  # if not within_class
        if multiple:
            pass
            # spikes = [generate_random_data_using_ISI(ind_spikes, num_trials, trial_durations) for ind_spikes in spikes]
        else:
            if use_PSTH:
                spikes = generate_random_data_using_PSTH(spikes, num_trials, trial_durations)
            else:
                spikes = generate_random_data_using_ISI(spikes, num_trials, trial_durations)
    return(spikes)


def find_all_params_script(responses, times, trial_times, variable_dict, pre_function, multiple, num_cells, training_set=None, **kwargs):
    if training_set is None:
        if responses is None and multiple:
            num_trials = len(times[0])
        else:
            num_trials = len(times)
        training_set = numpy.ones(num_trials, dtype=bool)
    params = {}
    for variable_name, values in variable_dict.iteritems():
        params[variable_name] = {}
        if pre_function is not None:
            offsets = numpy.array(trial_times[variable_name][0].T)[0]
            inf_times = numpy.array(trial_times[variable_name][1])
            if (responses is None) and multiple:
                new_sep_times = [filter_times(time, offsets, inf_times, responses=responses)[0] for time in times]
                new_sep_responses = {i: None for i in range(num_cells)}
            else:
                new_times, new_responses = filter_times(times, offsets, inf_times, responses=responses)
        for variable_value in values[0].keys():
            current_index = training_set * (values[1] == variable_value)
            if pre_function is not None:
                if multiple:
                    params[variable_name][variable_value] = {}
                    if responses is not None:
                        new_sep_responses, new_sep_times = separate_multiple(new_responses, new_times, num_cells)
                    for index in range(num_cells):
                        new_sep_time = new_sep_times[index]
                        new_sep_response = new_sep_responses[index]
                        if responses is None:
                            params[variable_name][variable_value][index] = pre_function(new_sep_time[current_index], inf_times=inf_times[current_index], **kwargs)
                        else:
                            params[variable_name][variable_value][index] = pre_function(new_sep_response[current_index], new_sep_time[current_index], inf_times=inf_times[current_index], **kwargs)
                else:
                    if responses is None:
                        params[variable_name][variable_value] = pre_function(new_times[current_index], inf_times=inf_times[current_index], **kwargs)
                    else:
                        params[variable_name][variable_value] = pre_function(new_responses[current_index], new_times[current_index], inf_times=inf_times[current_index], **kwargs)
            else:
                params[variable_name][variable_value] = {}
    return(params)


def find_all_probs_script(responses, times, trial_times, variable_dict, probability_function, multiple, num_cells, training_set=None, params=None, **kwargs):
    if training_set is None:
        if responses is None and multiple:
            num_trials = len(times[0])
        else:
            num_trials = len(times)
        training_set = numpy.ones(num_trials, dtype=bool)
    probabilities = {}
    for variable_name, values in variable_dict.iteritems():
        probabilities[variable_name] = {}
        offsets = numpy.array(trial_times[variable_name][0].T)[0]
        total_times = numpy.array(trial_times['total'])
        inf_times = numpy.array(trial_times[variable_name][1])
        if responses is None and multiple:
            new_sep_times = numpy.array([filter_times(time, offsets, inf_times, responses=responses)[0] for time in times])
            new_sep_responses = {i: None for i in range(num_cells)}
        else:
            new_times, new_responses = filter_times(times, offsets, inf_times, responses=responses)
        for variable_value in values[0].keys():
            current_index = training_set * (values[1] == variable_value)
            if multiple:
                ind_probs = {}
                if responses is not None:
                    new_sep_responses, new_sep_times = separate_multiple(new_responses, new_times, num_cells)
                for index in range(num_cells):
                    kwargs.update(params[variable_name][variable_value][index])
                    new_sep_time = new_sep_times[index]
                    new_sep_response = new_sep_responses[index]
                    if responses is None:
                        ind_probs[index] = probability_function(new_sep_time[current_index], total_times=total_times, **kwargs)
                    else:
                        ind_probs[index] = probability_function(new_sep_response[current_index], new_sep_time[current_index], total_times=total_times, **kwargs)
                probabilities[variable_name][variable_value] = multiple_prob_func_factory(ind_probs)
            else:
                kwargs.update(params[variable_name][variable_value])
                if responses is None:
                    probabilities[variable_name][variable_value] = probability_function(new_times[current_index], total_times=total_times, **kwargs)
                else:
                    probabilities[variable_name][variable_value] = probability_function(new_responses[current_index], new_times[current_index], total_times=total_times, **kwargs)
    return(probabilities)


def find_all_choices_script(responses, times, trial_times, variable_dict, probabilities, test_set=None, response_cutoff=1, at_best=True, track_traces=False):
    if test_set is None:
        num_trials = len(responses)
        test_set = numpy.ones(num_trials, dtype=bool)
    response_holdouts = responses[test_set]
    time_holdouts = times[test_set]
    num_trials = len(response_holdouts)
    choices = {}
    for variable_name, values in variable_dict.iteritems():
        offsets = numpy.array(trial_times[variable_name][0].T)[0][test_set]
        inf_times = numpy.array(trial_times[variable_name][1][test_set])
        trial_lengths = numpy.array(trial_times[variable_name][2][test_set])
        new_time_holdouts, new_response_holdouts = filter_times(time_holdouts, offsets, trial_lengths, responses=response_holdouts)
        num_values = len(values[0])
        null_array = numpy.zeros((num_trials, num_values))
        choices[variable_name] = {
            'probs': deepcopy(null_array),
            'counts': deepcopy(null_array),
            'times': numpy.zeros(num_trials)}
        if track_traces:
            choices[variable_name]['traces'] = []
            choices[variable_name]['trace_times'] = []
        value_map = values[0]
        num_values = len(value_map)
        null_prob = 1. / num_values
        prior = null_prob * numpy.ones(num_values)
        probability = probabilities[variable_name]
        for i, (response, time) in enumerate(zip(new_response_holdouts, new_time_holdouts)):
            trace = probability_trace(response, time, prior, probability, value_map)
            t, probs, count = find_choice(trace, time, at_best=at_best)
            if track_traces:
                choices[variable_name]['traces'].append(trace)
                choices[variable_name]['trace_times'].append(time)
            choices[variable_name]['times'][i] = t
            choices[variable_name]['probs'][i] = probs
            choices[variable_name]['counts'][i] = count
        if track_traces:
            choices[variable_name]['traces'] = numpy.array(choices[variable_name]['traces'])
            choices[variable_name]['trace_times'] = numpy.array(choices[variable_name]['trace_times'])
    return(choices)


def pre_script(variable_dict, spikes, trial_times, response_function, pre_function, prob_from_spikes=False, condition_variables=None, at_best=True, num_folds=10, multiple=False, use_false=False, shuffle=False, **kwargs):
    if multiple:
        num_cells = len(spikes)
        num_trials = len(spikes[0])
    else:
        num_cells = 1
        num_trials = len(spikes)
    total_trial = num_trials * [trial_times['total']]

    # Generating Random Data
    if use_false:
        spikes = generate_random_data_script(spikes, total_trial, variable_dict, multiple=multiple, **kwargs)
    if shuffle:
        for name in variable_dict.keys():
            variable_dict[name][1] = permutation(variable_dict[name][1])

    # Convert spikes to responses
    responses, times = convert_responses(spikes, response_function, total_trial, multiple, **kwargs)

    # Find conditional probabilities for each variable
    if prob_from_spikes:
        params = find_all_params_script(None, spikes, trial_times, variable_dict, pre_function, multiple, num_cells, training_set=None, **kwargs)
    else:
        params = find_all_params_script(responses, times, trial_times, variable_dict, pre_function, multiple, num_cells, training_set=None, **kwargs)
    return(params)


def main_script(variable_dict, spikes, trial_times, response_function, probability_function, prob_from_spikes=False, condition_variables=None, at_best=True, num_folds=10, multiple=False, use_false=False, within_class=False, shuffle=False, params=None, track_traces=False, **kwargs):
    if multiple:
        num_cells = len(spikes)
        num_trials = len(spikes[0])
    else:
        num_cells = 1
        num_trials = len(spikes)
    total_trial = num_trials * [trial_times['total']]

    # Shuffling data
    if use_false:
        spikes = generate_random_data_script(spikes, total_trial, variable_dict, multiple=multiple, **kwargs)
    if shuffle:
        for name in variable_dict.keys():
            variable_dict[name][1] = permutation(variable_dict[name][1])

    # Convert spikes to responses
    responses, times = convert_responses(spikes, response_function, total_trial, multiple, params=params, **kwargs)

    # Make an empty results dictionary
    results = {}
    for variable_name, values in variable_dict.iteritems():
        null_array = numpy.zeros((num_trials, len(values[0])))
        results[variable_name] = {
            'probs': deepcopy(null_array),
            'counts': deepcopy(null_array),
            'times': numpy.zeros(num_trials)}
        if track_traces:
            results[variable_name]['traces'] = num_trials*[None]
            results[variable_name]['trace_times'] = num_trials*[None]


    # Break data into folds
    folds = break_into_folds_balanced(variable_dict, num_folds)
    for test_set in folds:
        training_set = ~test_set
        # Find conditional probabilities for each variable
        if prob_from_spikes:
            probabilities = find_all_probs_script(None, spikes, trial_times, variable_dict, probability_function, multiple, num_cells, training_set=training_set, params=params, **kwargs)
        else:
            probabilities = find_all_probs_script(responses, times, trial_times, variable_dict, probability_function, multiple, num_cells, training_set=training_set, params=params, **kwargs)

        # Run on holdouts
        holdout_results = find_all_choices_script(responses, times, trial_times, variable_dict, probabilities, test_set=test_set, at_best=at_best, track_traces=track_traces)

        # Fill in values in the results array
        for variable_name, holdout_result in holdout_results.iteritems():
            results[variable_name]['probs'][test_set] = holdout_result['probs']
            results[variable_name]['counts'][test_set] = holdout_result['counts']
            results[variable_name]['times'][test_set] = holdout_result['times']
            if track_traces:
                ind = numpy.arange(len(test_set))
                for i, trace, trace_times in zip(ind[test_set], holdout_result['traces'], holdout_result['trace_times']):
                    results[variable_name]['traces'][i] = trace
                    results[variable_name]['trace_times'][i] = trace_times

    # Create contingency tables
    for variable_name, values in variable_dict.iteritems():
        results[variable_name]['probs_summary'] = contingency_table(results[variable_name]['probs'], values[0], values[1])
        results[variable_name]['counts_summary'] = contingency_table(results[variable_name]['counts'], values[0], values[1])
        if condition_variables is not None:
            for condition_name, condition_filter in condition_variables.iteritems():
                results[variable_name][condition_name + '_probs_summary'] = contingency_table(results[variable_name]['probs'][condition_filter], values[0], values[1][condition_filter])
                results[variable_name][condition_name + '_counts_summary'] = contingency_table(results[variable_name]['counts'][condition_filter], values[0], values[1][condition_filter])
    return(results)


def multiple_prob_func_factory(ind_probs):
    def probs(x, y):
        return(ind_probs[x[0]](x[1], y))
    return(probs)


def separate_multiple(responses, times, num_cells):
    separate_responses = {i: [] for i in range(num_cells)}
    separate_times = {i: [] for i in range(num_cells)}
    for response, time in zip(responses, times):
        for index in range(num_cells):
            try:
                sel_time = deepcopy(time[response[:, 0] == index])
                sel_response = deepcopy(response[response[:, 0] == index, 1])
            except:
                sel_time = numpy.array([])
                sel_response = numpy.array([])
            separate_times[index].append(sel_time)
            separate_responses[index].append(sel_response)
    for index, responses in separate_responses.iteritems():
        separate_responses[index] = numpy.array(responses)
    for index, times in separate_times.iteritems():
        separate_times[index] = numpy.array(times)
    return(separate_responses, separate_times)


def main_responses_script(variable_dict, spikes, trial_times, response_function, response_values, response_times, probability_function, prob_from_spikes=False, condition_variables=None, at_best=True, num_folds=10, multiple=False, use_false=False, within_class=False, shuffle=False, cross_groups=True, params=None, **kwargs):
    num_trials = len(spikes)
    total_trial = num_trials * [trial_times['total']]

    # Shuffling data
    if use_false:
        spikes = generate_random_data_script(spikes, total_trial, variable_dict, **kwargs)
    if shuffle:
        for name in variable_dict.keys():
            variable_dict[name][1] = permutation(variable_dict[name][1])

    # Convert spikes to responses
    responses, times = convert_responses(spikes, response_function, total_trial, multiple, params=params, **kwargs)

    # Make an empty results dictionary
    results = []
    # for variable_name, values in variable_dict.iteritems():
    #     null_array = numpy.zeros((num_trials, len(values[0])))
    #     results[variable_name] = {
    #         'probs': deepcopy(null_array),
    #         'counts': deepcopy(null_array),
    #         'times': numpy.zeros(num_trials)}

    # Break data into folds
    folds = break_into_folds_balanced(variable_dict, num_folds, cross_groups=cross_groups)
    for test_set in folds:
        training_set = ~test_set
        # Find conditional probabilities for each variable
        if prob_from_spikes:
            probabilities = find_all_probs_script(None, spikes, trial_times, variable_dict, probability_function, False, None, training_set=training_set, params=params, **kwargs)
        else:
            probabilities = find_all_probs_script(responses, times, trial_times, variable_dict, probability_function, False, None, training_set=training_set, params=params, **kwargs)
        current_probs = {}
        for var in probabilities.keys():
            current_probs[var] = {key: numpy.array([[prob(i, j) for i in response_values] for j in response_times[var]]) for key, prob in probabilities[var].iteritems()}
        results.append(current_probs)
    return(results)


def main_responses_script_OLD(variable_dict, spikes, nosepoke_times, response_function, response_values, probability_function, prob_from_spikes=False, trial_duration=2.5, condition_variables=None, shuffle=False, multiple=False, use_false=False, within_class=False, num_folds=10, spike_cutoff=3, **kwargs):

    # Screening out trials with low spike counts
    if spike_cutoff is not None:
        variable_dict = deepcopy(variable_dict)
        condition_variables = deepcopy(condition_variables)
        spikes = numpy.array(spikes)
        nosepoke_times = numpy.array(nosepoke_times)
        if multiple:
            cutoff_filter = numpy.array([len(flatten(spike)) for spike in spikes]) >= spike_cutoff
        else:
            cutoff_filter = numpy.array([len(spike) for spike in spikes]) >= spike_cutoff
        for variable_name, values in variable_dict.iteritems():
            variable_dict[variable_name][1] = numpy.array(values[1])[cutoff_filter]
        if condition_variables is not None:
            for condition_name, condition_filter in condition_variables.iteritems():
                condition_variables[condition_name] = condition_filter[cutoff_filter]
        spikes = numpy.array(spikes)[cutoff_filter]
        nosepoke_times = numpy.array(nosepoke_times)[cutoff_filter]
    else:
        spikes = numpy.array(spikes)
        nosepoke_times = numpy.array(nosepoke_times)
    num_trials = len(spikes)

    if use_false:
        if within_class:
            spikes_sorted = {}
            subgroups = break_into_subgroups(variable_dict)
            if multiple:
                new_spikes = []
                for ind_spikes in zip(*spikes):
                    for combination, index in subgroups:
                        spikes_sorted[combination] = ind_spikes[index]
                    variable_values = zip([values[1] for values in variables_dict.items()])
                    ind_spikes_new = generate_random_data_within_class(spikes_sorted, variable_values, nosepoke_times)
                    new_spikes.append(deepcopy(ind_spikes_new))
                spikes = zip(*new_spikes)
            else:
                for combination, index in subgroups:
                    spikes_sorted[combination] = spikes[index]
                variable_values = zip([values[1] for values in variables_dict.items()])
                spikes = generate_random_data_within_class(spikes_sorted, variable_values, nosepoke_times)
        else:  # if not within_class
            if multiple:
                spikes = zip(*[generate_random_data(ind_spikes, num_trials, nosepoke_times) for ind_spikes in zip(*spikes)])
            else:
                spikes = generate_random_data(spikes, num_trials, nosepoke_times)
    if shuffle:
        variable_dict = deepcopy(variable_dict)
        for variable_name, values in variable_dict.iteritems():
            shuffle_func(values[1])

    responses, times = convert_responses(spikes, response_function, nosepoke_times, **kwargs)
    folds = break_into_folds_balanced(variable_dict, num_folds)

    # make an empty results dictionary

    results = []

    for test_set in folds:
        training_set = ~test_set
        # find conditional probabilities for each variable
        probabilities = {}
        for variable_name, values in variable_dict.iteritems():
            probabilities[variable_name] = {}
            for variable_value in values[0].keys():
                probabilities[variable_name][variable_value] = {}
                if not prob_from_spikes:
                    current_prob = probability_function(responses[training_set * (values[1] == variable_value)], **kwargs)
                    probabilities[variable_name][variable_value] = numpy.array([current_prob(i) for i in response_values])
                else:
                    current_prob = probability_function(spikes[training_set * (values[1] == variable_value)], **kwargs)
                    probabilities[variable_name][variable_value] = numpy.array([current_prob(i) for i in response_values])
        results.append(probabilities)
    return(results)
