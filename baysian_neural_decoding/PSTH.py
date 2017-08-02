import numpy
from statsmodels.nonparametric.kde import KDEUnivariate
# from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
# from scipy.interpolate import interp1d
from .tools import flatten, filter_times

BIN_SIZE = .030
BW_METHOD = 'cv_ml'
BW = .030

###############################
# Calculating the spike_times #
###############################


def calc_spike_times(response, **kwargs):
    return(numpy.array(response), numpy.array(response))


def calc_time_words(responses, response_time=None, bin_size=BIN_SIZE, **kwargs):
    if response_time is None:
        response_time = (min(responses), max(responses))
    times = numpy.arange(response_time[0], response_time[1], bin_size)
    processed_responses = numpy.array([numpy.sum((responses >= time) * (responses < time + bin_size)) for time in times])
    return(processed_responses, times)


###########################
# Estimating Proabilities #
###########################

# def set_bw_and_bin_size(variable_dict, spikes, trial_times, response_function, probability_function, bw=BW, **kwargs):
#     bws = []
#     for variable_name, values in variable_dict.iteritems():
#         offsets = numpy.array(trial_times[variable_name][0].T)[0]
#         inf_times = numpy.array(trial_times[variable_name][1])
#         new_times, new_responses = filter_times(spikes, offsets, inf_times, responses=None)
#         for variable_value in values[0].keys():
#             current_index = (values[1] == variable_value)
#             bws.append(PSTH(new_times[current_index], bw=bw, **kwargs)[1])
#     mean_bw = numpy.mean(bws)
#     bin_size = 2 * numpy.sqrt(3) * mean_bw
#     return({'bin_size': bin_size, 'bw': mean_bw})


def set_bw_PSTH(spike_times, bw_method=BW_METHOD, mirror=False, trial_time=None, norm=True, num_folds=10, trial_duration=2.5, **kwargs):
    spike_times_flat = flatten(spike_times)

    if trial_time is None:
        trial_time = (numpy.min(spike_times_flat), numpy.max(spike_times_flat))
    if mirror:
        spike_times_flat = numpy.hstack((-spike_times_flat - 2 * trial_time[0], spike_times_flat, spike_times_flat + trial_time[1]))

    if bw_method == 'cv_ml':
        grid = GridSearchCV(KernelDensity(), {'bandwidth': numpy.linspace(.001, .200, 20)}, cv=num_folds)  # 10-fold cross-validation
        grid.fit(spike_times_flat.reshape(-1, 1))
        bw_value = grid.best_params_['bandwidth']

    return({'bw': bw_value})


def PSTH(spike_times, bw=BW, mirror=False, trial_time=None, norm=True, trial_duration=2.5, **kwargs):
    num = len(spike_times)
    spike_times_flat = flatten(spike_times)
    total_spikes = len(spike_times_flat)
    if trial_time is None:
        trial_time = (numpy.min(spike_times_flat), numpy.max(spike_times_flat))
    if mirror:
        spike_times_flat = numpy.hstack((-spike_times_flat - 2 * trial_time[0], spike_times_flat, spike_times_flat + trial_time[1]))
        # pre_factor = 3.
    # else:
    #     pre_factor = 1.
    # if norm:
    #     pre_factor = pre_factor * (total_spikes / num)
    # Using KDEMultivariate
    # if type(bw) is float:
    #     bw = [bw]
    # fit = KDEMultivariate(spike_times_flat, 'c', bw=bw)
    # return(lambda x: pre_factor * fit.pdf[x])
    #
    # Using KDEUnivariate (FFT)
    kde = KDEUnivariate(spike_times_flat)
    if bw is not None:
        kde.fit(bw=bw)
    else:
        kde.fit()
    if norm:
        pre_factor = total_spikes / (num * quad(lambda x: kde.evaluate([x])[0], trial_time[0], trial_time[1])[0])
    return(lambda x: pre_factor * kde.evaluate([x])[0])


def PSTH_poisson_prob(spike_times, bin_size=BIN_SIZE, bw=BW, mirror=False, trial_time=None, norm=True, trial_duration=2.5, **kwargs):
    expected = PSTH(spike_times, bw, mirror, trial_time, norm, trial_duration, **kwargs)

    def poisson_prob(count, time):
        ex = bin_size * expected(time + bin_size / 2.)
        # note does not divide by the factorial of the total count since this will not effect inference.
        return((ex ** count) * numpy.exp(-ex))
    return(poisson_prob)
