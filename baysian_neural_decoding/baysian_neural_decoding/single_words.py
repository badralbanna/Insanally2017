import numpy
from .tools import flatten

BIN_SIZE = .030
NUM_BINS = 1


#################################
# Calculating spike count words #
#################################


def bins(bin_size=BIN_SIZE, num=NUM_BINS):
    return(numpy.array([(float(i) / num) * bin_size for i in xrange(num + 1)]))


def calc_bin_counts(response, bin_size=BIN_SIZE, num_bins=NUM_BINS):
    bin = bins(bin_size, num_bins)
    return(str(tuple([numpy.sum((response >= bin[i]) * (response < bin[i + 1])) for i in xrange(num_bins)])))


def calc_words(responses, bin_size=BIN_SIZE, num_bins=NUM_BINS, total_times=(0.0, 2.5), **kwargs):
    if len(responses) == 0:
        return(numpy.array([]), numpy.array([]))
    #else:
        #max_response = numpy.max(responses)
        #min_response = numpy.min(responses)
    delta = float(bin_size) / num_bins
    epochs = numpy.arange(total_times[0], total_times[1] + delta, delta)
    processed_responses = []
    for time in epochs:
        index = (responses >= time) * (responses < time + bin_size)
        response = responses[index] - time
        processed_response = calc_bin_counts(response, bin_size=bin_size, num_bins=num_bins)
        processed_responses.append(processed_response)
    processed_responses = numpy.array(processed_responses)
    return(processed_responses, epochs)


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
