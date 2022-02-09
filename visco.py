from numpy import nonzero, arange, exp, pi, mean, real, imag, sqrt, sum
from numpy.fft import fft, fftshift
from scipy.special import gamma
from scipy.optimize import minimize


def get_r(x):
    '''
    get the time decay constant needed for a given signal
    :param x: some signal (listlike)
    :return: optimal time decay constant (float)
    '''
    first_nonzero = nonzero(x)[0][0]
    return abs(x[-1] / x[first_nonzero]) ** (1 / (x.size - first_nonzero))


def mdft(x, r, length=None):
    '''
    perform the modified discrete fourier transform of a signal at a given radial distance
    :param x: some signal (listlike)
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param length: number of elements in the transformed signal (int)
    (default is None, which leaves length=len(x); recommended to set equal to the length of shortest signal in batch)
    :return: modified discrete fourier transform of x at r (numpy array)
    '''
    n = arange(0, x.size, 1)
    return fftshift(fft(x * r ** -n, n=length))


def qsp(X, r, freq):
    '''
    relaxance (Q) of the spring-pot (fractional order strain derivative) model in the Z-domain
    :param X: model parameters (array of two floats) [C, b] where C is the fractional modulus in units of Pa - s^-b
    and b is the exponential constant which ranges between 0 and 1 (bounds inclusive). 0 defines pure spring-like
    behavior, 1 defines pure dashpot-like (fluid) behavior.
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param freq: frequency array in Hz (listlike) [-Nyquist freq, Nyquist freq]
    :return: relaxance of the spring-pot model in the Z-domain along the circle of radius r (numpy array)
    '''
    C, b = X
    return C * (1 - exp(-1j * freq * pi / freq[-1]) / r) ** b


def qplr(X, r, freq):
    '''
    relaxance (Q) of the power law model in the Z-domain
    :param X: model parameters (array of two floats) [E, n] where E is the modulus in units of Pa
    and n is the exponential constant which ranges between 0 and 1 (1 not included). 0 defines pure spring-like
    behavior, 1 defines pure dashpot-like (fluid) behavior.
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param freq: frequency array in Hz (listlike) [-Nyquist freq, Nyquist freq]
    :return: relaxance of the power law model in the Z-domain along the circle of radius r (numpy array)
    '''
    E, n = X
    return E * gamma(1 - n) * (1 - exp(-1j * freq * pi / freq[-1]) / r) ** n


def qmax(X, r, freq):
    '''
    relaxance (Q) of a single arm maxwell (standard linear solid) model in the Z-domain
    :param X: model parameters (array of three floats) [Ge, G1, T1] where Ge is the equilibrium modulus in units of Pa,
    G1 is the modulus of the maxwell arm in units of Pa, and T1 is the relaxation time in units of s^-1. The sum of
    Ge and G1 is known as Gg, the glassy modulus or instantaneous response (analogous to young's modulus)
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param freq: frequency array in Hz (listlike) [-Nyquist freq, Nyquist freq]
    :return: relaxance of a single arm maxwell model in the Z-domain along the circle of radius r (numpy array)
    '''
    Ge, G1, T1 = X
    return Ge + G1 - G1 / (1 + T1 * (1 - exp(-1j * freq * pi / freq[-1]) / r))


def qmax_fluid(X, r, freq):
    Ge, G1, E1, Ef = X
    q = qmax([Ge, G1, E1 / G1], r, freq)
    z = (1 - exp(-1j * freq * pi / freq[-1]) / r)
    return q * z * Ef / (z * Ef + q)


def qvoigtfluid(X, r, freq):
    G, solid_visc, fluid_visc = X
    z = (1 - exp(-1j * freq * pi / freq[-1]) / r)
    return z * fluid_visc * (G + solid_visc * z)/(G + (solid_visc + fluid_visc) * z)


def obj_q(X, q_real, r, freq, func, lower_bounds, upper_bounds, re_weight, im_weight):
    '''
    calculates the mean square error between q_real and the model fit defined by func(X, r, freq) which is then
    penalized linearly if any of the parameters in X are out of the bounds defined in lower_bounds and upper_bounds
    :param X:  model parameters (array of floats) depends on the function (func) chosen to define the relaxance
    :param q_real: the real relaxance that is being fit by a model (numpy array)
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param freq: frequency array in Hz (listlike) [-Nyquist freq, Nyquist freq]
    :param func: fitting function to use to define the model for the relaxance (must take following arguments:
    X, r, freq in this order
    :param lower_bounds: lower bounds for parameters in X (array of floats) same length as X. if X is defined as
    X=[x1, x2, x3] then lower_bounds is defined as [xl1, xl2, xl3] where xl1 is the lower bound of x1 for example
    :param upper_bounds: upper bounds for parameters in X (array of floats) same length as X. if X is defiend as
    X=[x1, x2, x3] then upper_bounds is defined as [xu1, xu2, xu3] where xu1 is the upper bound of x1 for example
    :return: the mean square error between q_real and the model fit defined by func(X, r, freq), penalizing linearly
    for any parameters being out of bounds
    '''
    q_pred = func(X, r, freq)  # model fit relaxance
    cost = mean(((real(q_real) - real(q_pred)) ** 2 * re_weight +
                 (imag(q_real) - imag(q_pred)) ** 2 * im_weight))  # mean square error
    mult = 1 + (sqrt(sum((X - upper_bounds) ** 2 * ((X - upper_bounds) > 0))) +
                sqrt(sum((X - lower_bounds) ** 2 * ((X - lower_bounds) < 0)))) * 1e9  # boundary distance penalty
    return mult * cost  # boundary penalized mean square error


def fitq(q_real, r, freq, func, x0, lower_bounds, upper_bounds, iterlim=1e4, re_weight=1, im_weight=1):
    '''
    fit a model for the relaxance of a material defined by func, to known relaxance signal (q_real) in the Z-domain
    defined on the circle of radius r for a range of frequency values (freq) within set boundaries
    :param q_real: the real relaxance that is being fit by a model (numpy array)
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param freq: frequency array in Hz (listlike) [-Nyquist freq, Nyquist freq]
    :param func: fitting function to use to define the model for the relaxance (must take following arguments:
    X, r, freq in this order
    :param x0: initial guess (array) for the parameters (X) for func
    :param lower_bounds: lower bounds for parameters in X (array of floats) same length as x0. if X is defined as
    X=[x1, x2, x3] then lower_bounds is defined as [xl1, xl2, xl3] where xl1 is the lower bound of x1 for example
    :param upper_bounds: upper bounds for parameters in X (array of floats) same length as x0. if X is defiend as
    X=[x1, x2, x3] then upper_bounds is defined as [xu1, xu2, xu3] where xu1 is the upper bound of x1 for example
    :param iterlim: limit of iterations in the optimiser (int)
    :return: optimized parameters (X) of the model (func) (numpy array)
    '''
    optimizer_options = {'maxiter': iterlim, 'maxfev': iterlim, 'xatol': 1e-60, 'fatol': 1e-60}
    res = minimize(obj_q, x0=x0, args=(q_real, r, freq, func, lower_bounds, upper_bounds, re_weight, im_weight),
                   method='Nelder-Mead', options=optimizer_options)
    return res.x


import matplotlib.pyplot as plt
def plot_re_im(q, w, fit=None, label_q='Data', label_fit='Fit', save=None):
    plt.plot(w, real(q), label=label_q)
    if fit is not None:
        plt.plot(w, real(fit), label=label_fit)
        plt.legend()
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('Q\' (Pa)')
    plt.title('Storage Modulus')
    plt.grid()
    if save is not None:
        plt.savefig('re_'+save)
    plt.show()
    plt.plot(w, imag(q), label=label_q)
    if fit is not None:
        plt.plot(w, imag(fit), label=label_fit)
        plt.legend()
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('Q\" (Pa)')
    plt.title('Loss Modulus')
    plt.grid()
    if save is not None:
        plt.savefig('im_'+save)
    plt.show()
