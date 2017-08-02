import numpy
from .tools import flatten

WINDOW = .030
NUM_BINS = 1


#################################
# Calculating spike count words #
#################################


def bins(window=WINDOW, num=NUM_BINS):
    return(numpy.array([(float(i) / num) * window for i in xrange(num + 1)]))


def calc_bin_counts(response, window=WINDOW, num_bins=NUM_BINS):
    bin = bins(window, num_bins)
    return(str(tuple([numpy.sum((response >= bin[i]) * (response < bin[i + 1])) for i in xrange(num_bins)])))


def calc_words(responses, window=WINDOW, num_bins=NUM_BINS, **kwargs):
    if len(responses) == 0:
        return(numpy.array([]), numpy.array([]))
    else:
        max_response = numpy.max(responses)
        min_response = numpy.min(responses)
    delta = float(window) / num_bins
    epoch = numpy.arange(min_response, max_response + delta, delta)
    processed_responses = []
    for time in epoch:
        index = (responses >= time) * (responses < time + window)
        response = responses[index] - time
        processed_response = calc_bin_counts(response, window=window, num_bins=num_bins)
        processed_responses.append(processed_response)
    processed_responses = numpy.array(processed_responses)
    return(processed_responses, epoch)


######################################
# Calculating discreet probabilities #
######################################

def discrete_probability(responses, times, gauranteed_set=None, remove_set=None, **kwargs):
    """
    """
    responses = flatten(responses)
    pR = {}
    if gauranteed_set is not None:
        for response in gauranteed_set:
            pR[response] = 1.
    for response in responses:
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

    def prob(x, y):
        if x in pR:
            return(pR[x])
        else:
            return(1.)
    return(prob)
