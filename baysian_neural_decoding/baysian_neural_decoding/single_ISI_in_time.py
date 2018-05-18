import numpy
# from scipy.fftpack import dct, idct
# from scipy.optimize import brentq
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.base.model import GenericLikelihoodModel
# from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from .tools import flatten
from .PSTH import PSTH, BW_PSTH
from scipy.integrate import quad, cumtrapz, trapz
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d

BW_ISI_METHOD = 'cv_ml'
BW_ISI = .200
WINDOW = 1.0
STEP = .100

###########################
# Estimating Proabilities #
###########################

############################
# ISIs Generated from PSTH #
############################

# Use with set_bw_ISI_PSTH, prob_from_spikes=True
def ISI_from_poisson(times, log=False, bw_PSTH=BW_PSTH, window=WINDOW, step=STEP, total_times=(0, 2.5), res=500, res_2=40, **kwargs):
    start, end = total_times
    time_bins = generate_time_bins(start, end, window=window, step=step)

    PSTH_func = PSTH(times, bw_PSTH=bw_PSTH, mirror=True, trial_time=total_times,  res=res)

    delta_1 = (end - start) / (res - 1.)
    PSTH_list = [PSTH_func(y) for y in numpy.linspace(start, end, res)]
    int_firing_list = cumtrapz(PSTH_list, dx=delta_1)

    def prob(ISI, a, b):
        start_bin = int(a / delta_1)
        end_bin = int(b / delta_1)
        step = int(ISI / delta_1)
        Y = [PSTH_list[i] * PSTH_list[i + step] * numpy.exp(-int_firing_list[i + step] + int_firing_list[i]) for i in range(start_bin, end_bin - step)]
        value = trapz(Y, dx=delta_1) * ISI
        return value

    #num_trials = len(times)
    times = flatten(times)

    if log:
        ISI_short = -3.
        ISI_long = numpy.log10(window)
    else:
        ISI_short = .001
        ISI_long = window
    X = numpy.linspace(ISI_short, ISI_long, res_2)
    delta_2 = (ISI_long - ISI_short) / (res_2 - 1.)

    prob_dict = {}
    for i, time_bin in enumerate(time_bins):
        if log:
            Y = numpy.array([prob(10**log_ISI, time_bin[0], time_bin[1]) * 10**log_ISI for log_ISI in X])
        else:
            Y = numpy.array([prob(ISI, time_bin[0], time_bin[1]) for ISI in X])
        norm = trapz(Y, dx=delta_2)
        Y = Y / norm
        prob_dict[tuple(time_bin)] = ius(X, Y)

    def prob_time_wrapper(ISI, time):
        try:
            value = prob_dict[current_bin(time, time_bins)](ISI)
            if value > 0:
                return(value)
            else:
                return(numpy.nan)
        except:
            return(numpy.nan)

    return prob_time_wrapper

#################
# Mixture Model #
#################

def log_poisson(x, tau):
    return(numpy.log(10) * 10**(x - tau) * numpy.exp(-10**(x - tau)))


def gaussian(x, mu, sigma):
    return((1. / (sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(x - mu)**2 / (2 * sigma**2)))


def logit(x):
    return(1. / (1. + numpy.exp(-x)))


def log_poisson_gaussian_mix_gen(tau, mu, sigma, p):
    p1 = logit(p)
    p2 = 1. - p1

    def log_poisson_gaussian_mix(x):
        return(p1 * log_poisson(x, tau) + p2 * gaussian(x, mu, sigma))

    return(log_poisson_gaussian_mix)


class PoissonLogGaussian(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        self.param_names = ['tau', 'mu', 'sigma', 'p']
        if exog is None:
            exog = numpy.zeros_like(endog)
        super(PoissonLogGaussian, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        tau, mu, sigma, p = params
        p1 = logit(p)
        p2 = 1. - p1
        return -numpy.log(p1 * log_poisson(self.endog, tau) + p2 * gaussian(self.endog, mu, sigma))

    def fit(self, start_params=None, maxiter=1000000, maxfun=1000000, **kwds):
        if start_params is None:
            mean_value = self.endog.mean()
            median_value = numpy.median(self.endog)
            start_params = numpy.array([mean_value, median_value, .3, 2.])

        return super(PoissonLogGaussian, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)


def poisson_log_gaussian_mixture(values, times, log=True, bw_ISI=BW_ISI, window=WINDOW, step=STEP, total_times=(0, 2.5), res=1000, **kwargs):
    time_bins = generate_time_bins(total_times[0], total_times[1], window=window, step=step)
    values = flatten(values)
    times = flatten(times)

    prob_dict = {}
    for i, time_bin in enumerate(time_bins):
        current_values = values[(time_bin[0] < times) * (times <= time_bin[1])]
        if not log:
            current_values = numpy.hstack((-current_values, current_values))

        if len(current_values) > 0:
            model = PoissonLogGaussian(current_values)
            results = model.fit(disp=0, method='lbfgs')
            prob_dict[tuple(time_bin)] = log_poisson_gaussian_mix_gen(*results.params)

    def prob_time_wrapper(ISI, time):
        try:
            return(prob_dict[current_bin(time, time_bins)](ISI))
        except:
            return(numpy.nan)

    return prob_time_wrapper


########################
# KDE Model of the ISI #
########################

def current_bin(time, time_bins):
    max_distance = 0.
    for time_bin in time_bins:
        if time > time_bin[0] and time <= time_bin[1]:
            current_distance = min(time_bin[1] - time, time - time_bin[0])
            if current_distance > max_distance:
                max_distance = current_distance
                max_time_bin = tuple(time_bin)
    return(tuple(max_time_bin))


def generate_time_bins(beginning_time, final_time, window=WINDOW, step=STEP):
    return([(i - window / 2., i + window / 2.) for i in numpy.arange(beginning_time, final_time + step, step)])


def timed_prob(values, times, log=False, bw_ISI=BW_ISI, window=WINDOW, step=STEP, total_times=(0, 2.5), model='kde', **kwargs):
    time_bins = generate_time_bins(total_times[0], total_times[1], window=window, step=step)
    values = flatten(values)
    times = flatten(times)
    # if latency:
    #     times = times[~numpy.isnan(values)]
    #     values = values[~numpy.isnan(values)]

    kde_dict = {}
    for i, time_bin in enumerate(time_bins):
        current_values = values[(time_bin[0] < times) * (times <= time_bin[1])]
        if not log:
            current_values = numpy.hstack((-current_values, current_values))

        if len(current_values) > 0:
            if model == 'kde':
                kde_dict[tuple(time_bin)] = KDEUnivariate(current_values)
                kde_dict[tuple(time_bin)].fit(bw=bw_ISI)
            elif model == 'plg':
                PLG = PoissonLogGaussian(current_values)
                results = PLG.fit(disp=False)
                tau, mu, sigma, p = results.params
                kde_dict[tuple(time_bin)] = log_poisson_gaussian_mix_gen(tau, mu, sigma, p)

    if model == 'kde':
        def prob_time_wrapper(ISI, time):
            try:
                return(kde_dict[current_bin(time, time_bins)].evaluate([ISI])[0])
            except:
                return(1.)
    elif model == 'plg':
        def prob_time_wrapper(ISI, time):
            try:
                return(kde_dict[current_bin(time, time_bins)](ISI))
            except:
                return(1.)
    return(prob_time_wrapper)


def set_bw_ISI_in_time(ISIs, times, log=False, bw_ISI_method=BW_ISI_METHOD, inf_times=None, num_folds=10, window=WINDOW, **kwargs):
    new_ISIs = []
    for inf_time, ISI, time in zip(inf_times, ISIs, times):
        # if latency:
        #     time = time[~numpy.isnan(ISI)]
        #     ISI = ISI[~numpy.isnan(ISI)]
        # removed 2*window and window 
        new_ISIs.append(ISI[(time <= inf_time[0] + window) * (time > inf_time[0])])
    new_ISIs = flatten(new_ISIs)

    # Need to mirror data if not using log
    if not log:
        new_ISIs = numpy.hstack((-1 * new_ISIs, new_ISIs))

    if bw_ISI_method == 'cv_ml':
        grid = GridSearchCV(KernelDensity(), {'bandwidth': numpy.linspace(.010, .500, 50)}, cv=num_folds)  # 10-fold cross-validation
        grid.fit(new_ISIs.reshape(-1, 1))
        bw_ISI_value = grid.best_params_['bandwidth']

    return({'bw_ISI': bw_ISI_value})
