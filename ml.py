from scipy.optimize import curve_fit
from numpy import sqrt, sum, argmin, array, arange, argwhere, nan, isnan, mean, random, unique
from mlinsights.mlmodel import KMeansL1L2
from numpy import min, abs
from numpy.fft import fftshift, fft


def dist(Y, y):
    return sqrt(sum((Y - y) ** 2))


def assign_groups(x_data, y_data, slope_guess, power):
    groups = []
    for x, y in zip(x_data, y_data):
        groups.append(argmin([dist(y, m * abs(x) ** power) for m in slope_guess]))
    return array(groups).astype(int)


def get_group_slopes(groups, slope_real, k):
    group_ids = arange(k)
    slope_guess_new = []

    for group in group_ids:
        group_index = argwhere(groups == group)
        if group_index.size == 0:
            slope_guess_new.append(nan)
        else:
            slope_guess_new.append(mean(slope_real[group_index]))
    slope_guess_new
    nan_index = isnan(slope_guess_new)
    sub = mean(array(slope_guess_new)[argwhere(~nan_index)])
    for i in argwhere(nan_index):
        slope_guess_new[i[0]] = sub

    return array(slope_guess_new)


def k_slopes(x_data, y_data, k, fit_power=3 / 2, iterlim=10):
    def objective(x, m):
        return m * abs(x) ** fit_power

    slope_real = array([curve_fit(objective, x, y)[0] for x, y in zip(x_data, y_data)]).flatten()
    slope_guess = random.choice(slope_real, k)
    for i in range(iterlim):
        groups = assign_groups(x_data, y_data, slope_guess, fit_power)
        slope_guess = get_group_slopes(groups, slope_real, k)
    # for every group, calculate the error between the TEST CURVE and the REAL CURVES
    group_errors = []
    for group in unique(groups):
        group_id = argwhere(groups == group).flatten()
        group_errors.append(mean([dist(y, slope_guess[group] * abs(x) ** (3 / 2))
                                  for x, y in zip(x_data[group_id], y_data[group_id])]))
    return {'labels': groups, 'errors': array(group_errors)}


def get_elbow_OLD(x_data, y_data, k_max=10, iterlim=10):
    return array([mean(k_slopes(x_data, y_data, i, iterlim=iterlim)['errors']) for i in range(1, k_max)])


def k_curves_fft(curves, k, errors=False, downsample_pct=0):
    n = int((1 - downsample_pct) * min(c.size for c in curves))
    ffts = array([abs(fftshift(fft(c, n=n))) for c in curves])
    model = KMeansL1L2(n_clusters=k, norm='l1', init='k-means++', random_state=42)
    model.fit(ffts)
    if errors:
        return model.inertia_
    else:
        return model.labels_


def get_elbow(curves, k_range=arange(1, 11, 1), downsample_pct=0):
    distortions = array([k_curves_fft(curves, k, errors=True, downsample_pct=downsample_pct) for k in k_range])
    return k_range, distortions
