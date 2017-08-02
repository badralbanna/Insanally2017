import numpy
from math import ceil
from scipy.fftpack import dct, idct
from scipy.optimize import brentq
from statsmodels.nonparametric.kde import KDEUnivariate
# from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d
# from scipy.interpolate import UnivariateSpline
from .tools import flatten

BW_METHOD = 'cv_ml'
BW = .030
LOG = False

########################
# Calculating the ISIs #
########################


def calc_ISIs(response, log=False, **kwargs):
    if len(response) >= 2:
        ISI = response[1:] - response[:-1]
        times = response[1:]
    else:
        ISI = numpy.array([])
        times = numpy.array([])
    if not log:
        return(numpy.array(ISI), numpy.array(times))
    else:
        return(numpy.log10(numpy.array(ISI)), numpy.array(times))


###########################
# Estimating Proabilities #
###########################


def kde_Botev2010(data, n=2**14, data_range=None):
    n = 2**ceil(numpy.log2(n))
    if data_range is None:
        minimum = min(data)
        maximum = max(data)
        length = maximum - minimum
        minimum = minimum - length / 2
        maximum = maximum + length / 2
        data_range = maximum - minimum
    else:
        minimum, maximum = data_range
        data_range = maximum - minimum

    def fixed_point(t, N, I, a2, l=7):
        f = 2 * numpy.pi * (2 * l) * numpy.sum(I ** l * a2 * numpy.exp(-I * numpy.pi ** 2 * t))
        for s in xrange(l - 1, 1, -1):
            K0 = numpy.prod(numpy.arange(1, s, 2)) / numpy.sqrt(2 * numpy.pi)
            const = (1. + (1. / 2) ** (s + 1. / 2)) / 3.
            time = (2 * const * K0 / N / f)**(2. / (3. + 2. * s))
            f = 2 * numpy.pi * (2 * l) * numpy.sum(I ** s * a2 * numpy.exp(-I * numpy.pi ** 2 * time))
        return(t - (2 * N * numpy.sqrt(numpy.pi) * f) ** (-2. / 5))

    dx = data_range / n
    x_mesh = numpy.arange(minimum, maximum + dx, dx)
    N = float(len(set(data)))
    hist_data = numpy.histogram(data, bins=x_mesh)[0] / N
    hist_data = hist_data / numpy.sum(hist_data)
    a = dct(hist_data)
    I = numpy.arange(1, n)**2
    a2 = (a[1:] / 2.)**2

    try:
        t_star = brentq(lambda t: fixed_point(t, N, I, a2), 0, .1)
    except:
        t_star = .28 * N ** (-2. / 5)

    bandwidth = numpy.sqrt(t_star * data_range)
    a_t = numpy.exp(-numpy.arange(0, n) ** 2 * numpy.pi ** 2 * t_star / 2.) * a
    density = idct(a_t, norm='ortho')
    density = density / (dx * numpy.sum(density))  # normalizing
    # density_function = UnivariateSpline(x_mesh[:-1],density)
    density_function = interp1d(x_mesh[:-1], density)

    return(density_function, bandwidth)


def kde_Botev2010_2d(data, n=2**10, data_range=None):
    n = 2**ceil(numpy.log2(n))

    if data_range is None:
        minimum = numpy.min(data, axis=0)
        maximum = numpy.max(data, axis=0)
        data_range = maximum - minimum
        minimum_xy = minimum - data_range / 4.
        maximum_xy = maximum + data_range / 4.
        scaling = maximum_xy - minimum_xy
    else:
        minimum_xy = numpy.array(data_range[0])
        maximum_xy = numpy.array(data_range[1])
        scaling = maximum_xy - minimum_xy

    def K(s):
        return((-1)**s*numpy.prod(numpy.arange(1, 2*s, 2)) / numpy.sqrt(2*numpy.pi))

    def psi(s, time, A2, I):
        w = numpy.exp(-I*numpy.pi**2*time)*numpy.append(numpy.array([1]), .5*numpy.ones((1, len(I)-1)))
        w_x = numpy.matrix(w*(I**s[0]))
        w_y = numpy.matrix(w*(I**s[1]))
        A2 = numpy.matrix(A2)
        out = (-1)**numpy.sum(s)*(w_y*A2*w_x.T)*numpy.pi**(2*numpy.sum(s)) / N**2
        return(out[0, 0])

    def func(s, t, N, A2, I):
        if numpy.sum(s) <= 4:
            sum_func = func([s[0] + 1, s[1]], t, N, A2, I) + func([s[0], s[1] + 1], t, N, A2, I)
            const = (1. + 1./2.**(numpy.sum(s)+1.)) / 3.
            time = (-2*const*K(s[0])*K(s[1]) / (N * sum_func))**(1./(2.+numpy.sum(s)))
            return(psi(s, time, A2, I))
        else:
            return(psi(s, t, A2, I))

    def evolve(t, func, N, A2, I):
        sum_func = func([0, 2], t, N, A2, I) + func([2, 0], t, N, A2, I) + 2*func([1, 1], t, N, A2, I)
        time = (2*numpy.pi*N*sum_func)**(-1./3)
        return(time)

    N = len(data)
    transformed_data = (data - minimum_xy.reshape(1, 2).repeat(N, 0)) / scaling.reshape(1, 2).repeat(N, 0)
    data_X, data_Y = zip(*transformed_data)
    dx = 1. / n
    bins = numpy.arange(0, 1+dx, dx)
    initial_data = numpy.histogram2d(data_X, data_Y, bins=bins)[0]
    a = dct(dct(initial_data).T)
    A2 = a**2
    I = numpy.arange(0, n)**2

    t_star = brentq(lambda t: t - evolve(t, func, N, A2, I), 0, .1, maxiter=2000)

    p_02 = func([0, 2], t_star, N, A2, I)
    p_20 = func([2, 0], t_star, N, A2, I)
    p_11 = func([1, 1], t_star, N, A2, I)

    t_y = (p_02**(3./4) / (4*numpy.pi*N*p_20**(3./4)*(p_11 + numpy.sqrt(p_20*p_02))))**(1./3)
    t_x = (p_20**(3./4) / (4*numpy.pi*N*p_02**(3./4)*(p_11 + numpy.sqrt(p_20*p_02))))**(1./3)

    a_tx = numpy.matrix(numpy.exp(- numpy.arange(0, n)**2*numpy.pi**2*t_x / 2.)).T
    a_ty = numpy.matrix(numpy.exp(- numpy.arange(0, n)**2*numpy.pi**2*t_y / 2.))
    a_t = numpy.array(a_tx * a_ty) * a

    density = idct(idct(a_t).T)
    density = density / numpy.sum(density) * (numpy.product(a_t.shape) / numpy.product(scaling))
    x = numpy.linspace(minimum_xy[0], maximum_xy[0], n)
    y = numpy.linspace(minimum_xy[1], maximum_xy[1], n)
    density_function = interpolate.RectBivariateSpline(x, y, density)

    def density_function_2(xy, **kwargs):
        return(density_function(*xy, **kwargs)[0, 0])
    bandwidth = numpy.sqrt([t_x, t_y]) * scaling
    return(density_function_2, bandwidth)


def kde_wrapper(values, times, log=LOG, bw=BW, **kwargs):
    """A wrapper function for your KDE function of choice. Returns a
    function that will be used by future code in this module.

    Currently it uses statsmodels.nonparametric.kernel_density import KDEMultivariate
    """
    values = flatten(values)

    # Need to mirror data if not using log
    if not log:
        values = numpy.hstack((-values, values))

    # Using KernelDensity
    # kde = KernelDensity(bandwidth=bw)
    # kde.fit(values.reshape(-1, 1))
    # return(lambda x, y: 10**kde.score_samples([[x]])[0])

    # Using KDSUnivariate (FFT)
    kde = KDEUnivariate(values)
    if bw is None:
        kde.fit()
    else:
        kde.fit(bw=bw)
    return(lambda x, y: kde.evaluate([x])[0])

    # Using KDEMultivariate
    # kde = KDEMultivariate(values, 'c', bw=bw)
    # return(lambda x, y: kde.pdf([x])[0])


def set_bw(ISIs, times, log=LOG, bw_method=BW_METHOD, num_folds=10, **kwargs):
    ISIs = flatten(ISIs)

    # Need to mirror data if not using log
    if not log:
        ISIs = numpy.hstack((-ISIs, ISIs))

    if bw_method is 'cv_ml':
        grid = GridSearchCV(KernelDensity(), {'bandwidth': numpy.linspace(.001, 1.00, 100)}, cv=num_folds)  # 10-fold cross-validation
        grid.fit(ISIs.reshape(-1, 1))
        bw_value = grid.best_params_['bandwidth']
        # fit = KDEMultivariate(ISIs, 'c', bw=bw_method)
        # bw_value = fit.bw[0]
    elif bw_method is 'None':
        bw_value = None

    return({'bw': bw_value})
