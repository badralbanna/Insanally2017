import numpy
from .single_ISI_in_time import current_bin, generate_time_bins


########################
# Calculating the ISIs #
########################


def calc_ISIs_multiple(spikes_list):
    ISIs_all = []
    times_all = []
    for i, spikes in enumerate(spikes_list):
        num_spikes = len(spikes)
        ISIs, times = calc_ISIs(spikes)
        ISIs_labeled = zip(num_spikes * [i], ISIs)
        ISIs_all.extend(ISIs_labeled)
        times_all.extend(times)
    sort_index = numpy.argsort(times_all)
    times_all = numpy.array(times_all)[sort_index]
    ISIs_all = numpy.array(ISIs_all)[sort_index]
    return(ISIs_all, times_all)

###########################
# Estimating Proabilities #
###########################


def kde_wrapper_multiple(ISI_array, times):
    probabilities = {}
    for index in set(ISI_array[:, 0]):
        probabilities[index] = kde_wrapper(ISI_array[ISI_array[:, 0] == index, 1])

    def prob(pair, time):
        return(probabilities[pair[0]](pair[1]))
    return(prob)
