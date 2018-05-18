import numpy
from statsmodels.nonparametric.kde import KDEUnivariate
# from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
# from scipy.interpolate import interp1d
from .tools import flatten

BIN_SIZE = .030
BW_METHOD = 'cv_ml'
BW_PSTH = .030

###############################
# Calculating the spike_times #
###############################


def calc_spike_times(response, **kwargs):
    return(numpy.array(response), numpy.array(response))


def calc_spike_times_2(response, **kwargs):
    current = numpy.array(response)
    beginning = 0.
    if len(current) >= 1:
        previous = [beginning] + list(response[:-1])
    else:
        previous = []
    pair = zip(previous, current)
    return(numpy.array(pair), numpy.array(response))


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
        spike_times_flat = numpy.hstack((-1 * spike_times_flat - 2 * trial_time[0], spike_times_flat, spike_times_flat + trial_time[1]))

    if bw_method == 'cv_ml':
        grid = GridSearchCV(KernelDensity(), {'bandwidth': numpy.linspace(.001, .200, 20)}, cv=num_folds)  # 10-fold cross-validation
        grid.fit(spike_times_flat.reshape(-1, 1))
        bw_value = grid.best_params_['bandwidth']

    return({'bw_psth': bw_value})


def PSTH(spike_times, bw_psth=BW_PSTH, mirror=False, trial_time=None, norm=True, trial_duration=2.5, **kwargs):
    num = len(spike_times)
    spike_times_flat = flatten(spike_times)
    total_spikes = len(spike_times_flat)
    if trial_time is None:
        trial_time = (numpy.min(spike_times_flat), numpy.max(spike_times_flat))
    if mirror:
        spike_times_flat = numpy.hstack((-1 * spike_times_flat + 2 * trial_time[0], spike_times_flat, -1 * spike_times_flat + 2 * trial_time[1]))
    kde = KDEUnivariate(spike_times_flat)
    if bw_psth is not None:
        kde.fit(bw=bw_psth)
    else:
        kde.fit()
    if norm:
        pre_factor = total_spikes / (num * quad(lambda x: kde.evaluate([x])[0], trial_time[0], trial_time[1])[0])
    else: 
        pre_factor = 1.
    return(lambda x: pre_factor * kde.evaluate([x])[0])


def PSTH_poisson_prob(spike_times, bin_size=BIN_SIZE, bw_psth=BW_PSTH, mirror=False, trial_time=None, norm=True, trial_duration=2.5, **kwargs):
    expected = PSTH(spike_times, bw_psth, mirror, trial_time, norm, trial_duration, **kwargs)

    def poisson_prob(count, time):
        ex = bin_size * expected(time + bin_size / 2.)
        # note does not divide by the factorial of the total count since this will not effect inference.
        return((ex ** count) * numpy.exp(-ex))
    return(poisson_prob)


def PSTH_poisson_prob_2(spike_times, bw_psth=BW_PSTH, trial_time=None, norm=True, trial_duration=2.5, total_times=(0, 2.5), log=False, **kwargs):
    PSTH_func = PSTH(spike_times, bw_psth=bw_psth, mirror=True, trial_time=total_times, norm=True, trial_duration=trial_duration)

    def prob_of_spike(ISI, time):
        if log:
            ISI = 10.**ISI
        int_of_interval = quad(PSTH_func, time - ISI, time, maxp1=100)[0]
        r = PSTH_func(time)
        prob = numpy.exp(-int_of_interval) * r
        # print('PSTH int {0}, rate {1}'.format(int_of_interval, r))
        return(prob)

    return(prob_of_spike)
