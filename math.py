from numpy import cumsum, convolve, ones, linspace, pi, random, tile, round, sqrt
from scipy import signal


def deg_to_rad(angle):
    return angle * pi / 180

def rad_to_deg(angle):
    return angle * 180 / pi


def movingaverage(arr, n=10):
    ret = cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def sma(arr, n=10):
    return convolve(arr, ones(n), 'valid') / n


def sma_shift(arr, pct=0.05):
    length = int(pct * arr.size)
    arr_new = convolve(arr, ones(length), 'valid') / length
    return arr_new - arr_new[0]


def altspace(start, step, count, **kwargs):
    '''
    creates an evenly spaced numpy array starting at start, with a specified step size, and a given number of steps
    :param start: float starting position of array
    :param step: float step size between consecutive elements in the array
    :param count: int total number of elements in the array
    :param kwargs: any extra arguments that may be passed in the numpy array creation
    :return:
    '''
    return linspace(start, start + (step * count), count, endpoint=False, **kwargs)


def gaussian_white_noise(amplitude, shape):
    '''
    creates a gaussian white noise signal for adding experimental noise to a signal
    :param amplitude: float amplitude (2 * standard deviation) of the noise signal
    :param num_samples: tuple of ints size of the signal
    :return: (num_samples, num_signals) numpy array with the noise signal
    '''
    return random.normal(loc=0, scale=amplitude / 2, size=shape)

def row2mat(row, n): #@TODO move to helperfunctions
    '''
    stacks a row vector (numpy (m, )) n times to create a matrix (numpy (m, n)) NOTE: CAN SLOW DOWN COMPUTATION IF DONE MANY TIMES
    :param row: numpy array row vector
    :param n: int number of replications to perform
    :return: numpy matrix (m, n) replicated row vector
    '''
    # do once at the beginning of any calculation to improve performance
    return tile(row, (n, 1)).T

def norm_cross_corr(f1, f2, method = 'same'):
    '''
    computes the cross correlationg between two signals f1 and f2 and then normalizes the result
    by dividing by the square root of the product of the max of each auto correlation sqrt(f1*f1 f2*f2)
    :param f1: numpy array (n,)
    :param f2: numpy array (m,) n not necessarily == m
    :return: numpy array (n,)
    '''
    return signal.correlate(f1, f2, mode=method) / sqrt(signal.correlate(f1, f1, mode=method)[int(f1.size / 2)] * signal.correlate(f2, f2, mode=method)[int(f2.size / 2)])

def downsampu2v(u, v):
    '''
    downsamples an array, u, to the size of another, smaller array, v
    :param u: numpy array (n,)
    :param v: numpy array (m,) where m<=n
    :return: numpy array, u* (m,)
    '''
    return u[round(linspace(0, u.size - 1, v.size)).astype(int)]