import numpy
from .tools import flatten
from single_words import bins, calc_bin_counts, calc_words
from copy import copy, deepcopy

######################################
# Calculating discreet probabilities #
######################################

def discrete_probability_in_time(responses, times, gauranteed_set=None, remove_set=None, total_times=(0.0, 2.5), **kwargs):
    """
    """
    responses = flatten(responses)
    times = flatten(times)
    time_bins = set(times)
    pR_times = {}
    for time in time_bins:
        pR = {}
        if gauranteed_set is not None:
            for response in gauranteed_set:
                pR[response] = 1.
        for response in responses[times == time]:
            if response in pR:
                pR[response] += 1.
            else:
                # All start with 1. count in accordance with Baysian inference.
                pR[response] = 2.
        pR = dict(zip(pR.keys(), numpy.array(pR.values()) / numpy.sum(pR.values())))
        if remove_set is not None:
            for response in remove_set:
                if response in pR:
                    # This will remove the influence of this response on baysian
                    # inference.
                    pR[response] = 1.
        pR_times[time] = copy(pR)

    def prob_time(x, y):
        if x in pR_times[y]:
            return(pR_times[y][x])
        else:
            return(1.)
    return(prob_time)



