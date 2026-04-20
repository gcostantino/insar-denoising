import numpy as np


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return phi, rho


def pol2cart(theta, r):
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return x1, x2
