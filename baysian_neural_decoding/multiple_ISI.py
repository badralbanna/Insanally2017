import numpy
from .single_ISI import calc_ISIs
from .single_ISI import kde_wrapper
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from .tools import flatten

BW_METHOD = 'cv_ml'
BW = .030
LOG = False

########################
# Calculating the ISIs #
########################


def calc_ISIs_multiple(spikes_list, log=LOG, **kwargs):
    ISIs_all = []
    times_all = []
    for i, spikes in enumerate(spikes_list):
        num_spikes = len(spikes)
        ISIs, times = calc_ISIs(spikes, log=log)
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


def kde_wrapper_multiple(ISI_array, times, log=LOG, bw=BW, **kwargs):
    probabilities = {}
    for index in set(ISI_array[:, 0]):
        probabilities[index] = kde_wrapper(ISI_array[ISI_array[:, 0] == index, 1], log=log, bw=bw[index], **kwargs)

    def prob(pair, time):
        return(probabilities[pair[0]](pair[1]))
    return(prob)


def set_bw_multiple(ISI_array, times, log=LOG, bw_method=BW_METHOD, num_folds=10, **kwargs):
    bw_value = {}
    ISI_array = flatten(ISI_array)
    for index in set(ISI_array[:, 0]):
        ISIs = ISI_array[ISI_array[:, 0] == index, 1]
        # Need to mirror data if not using log
        if not log:
            ISIs = numpy.hstack((-ISIs, ISIs))

        if bw_method is 'cv_ml':
            grid = GridSearchCV(KernelDensity(), {'bandwidth': numpy.linspace(.001, 1.00, 100)}, cv=num_folds)  # 10-fold cross-validation
            grid.fit(ISIs.reshape(-1, 1))
            bw_value[index] = grid.best_params_['bandwidth']
            # fit = KDEMultivariate(ISIs, 'c', bw=bw_method)
            # bw_value[index] = fit.bw[0]
        elif bw_method is 'None':
            bw_value[index] = None
    return({'bw': bw_value})
