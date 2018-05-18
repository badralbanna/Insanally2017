import numpy
from copy import deepcopy


def probability_trace(responses, times, prior, probability, variable_map):
    num_responses = len(responses)
    num_variables = len(variable_map)
    prob_trace = numpy.zeros((num_responses + 1, num_variables))
    prob_trace[0] = deepcopy(prior)
    new_prob = deepcopy(prior)
    for i, (response, time) in enumerate(zip(responses, times)):
        old_prob = deepcopy(new_prob)
        try:
            for variable_value, index in variable_map.iteritems():
                cur_prob = probability[variable_value](response, time)
                if numpy.isnan(cur_prob):
                    raise ValueError
                new_prob[index] = cur_prob * old_prob[index]
            new_prob = new_prob / numpy.sum(new_prob)
        except:
            new_prob = deepcopy(old_prob[:])
        prob_trace[i + 1] = deepcopy(new_prob[:])
    return(prob_trace)

# FIX TO INCLUDE TIMES


def probability_trace_multiple_ISI(response, times, prior, pRgX, cutoff=10, crosstalk=True):
    pS = prior
    pS_list = []
    cond_neuron, spike = find_next_spike(response, 0)
    while True:
        next_pS = pS[:]
        if crosstalk:
            search_responses = response
        else:
            search_responses = {cond_neuron: response[cond_neuron]}

        for search_neuron, search_spike_times in search_responses.iteritems():
            key = '{0}|{1}'.format(search_neuron, cond_neuron)
            current_response = find_last_ISI(search_spike_times, spike)
            if not numpy.isnan(current_response):
                next_pS = numpy.array([prob[key](current_response) * prior_prob for prob, prior_prob in zip(pRgX, next_pS)])
                next_pS = next_pS / numpy.sum(next_pS)
        if not numpy.isnan(next_pS[0]):
            pS_list.append(list(next_pS))
            pS = next_pS[:]
        else:
            pS_list.append(list(pS))
        cond_neuron, spike = find_next_spike(response, spike)
        if numpy.isnan(cond_neuron):
            return(numpy.array(pS_list))
    return(numpy.array(pS_list))


def find_choice(trace, times, at_best=True):
    try:
        if at_best:
            time_index, count_index = numpy.unravel_index(trace.argmax(), trace.shape)
            count = numpy.zeros(trace.shape[1])
            count[count_index] = 1.
            probs = trace[time_index]
            time = times[time_index - 1]
        else:
            probs = trace[-1]
            if any(numpy.isnan(prob) for prob in probs):
                raise ValueError
            count = numpy.zeros(trace.shape[1])
            count[probs.argmax()] = 1.
            time = numpy.nan
    except:
        count = numpy.zeros(trace.shape[1])
        probs = numpy.zeros(trace.shape[1])
        time = numpy.nan
    return(time, probs, count)
