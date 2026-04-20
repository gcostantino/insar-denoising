import numpy as np
from scipy.special import erf


def gaussian_cdf(x, mu, sig):
    """Gaussian Cumulative Distribution function (CDF)"""
    return 0.5 * (1. + erf((x - mu) / (sig * np.sqrt(2.))))
