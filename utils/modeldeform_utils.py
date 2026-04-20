import numpy as np

from utils.coords_utils import cart2pol, pol2cart


def mogi_defo(x, y, xcen=0, ycen=0, d=3e3, dV=1e6, nu=0.25):
    """Calculates surface deformation based on point source
    References: Mogi 1958, Segall 2010 p.203
    Args:
    ------------------
    x: x-coordinate grid (m)
    y: y-coordinate grid (m)
    Kwargs:
    -----------------
    xcen: y-offset of point source epicenter (m)
    ycen: y-offset of point source epicenter (m)
    d: depth to point (m)
    dV: change in volume (m^3)
    nu: poisson's ratio for medium
    Returns:
    -------
    (ux, uy, uz)
    """
    # Center coordinate grid on point source
    x = x - xcen
    y = y - ycen

    # Convert to surface cylindrical coordinates
    th, rho = cart2pol(x, y)
    R = np.hypot(d, rho)

    # Mogi displacement calculation
    C = ((1 - nu) / np.pi) * dV
    ur = C * rho / R ** 3
    uz = C * d / R ** 3

    ux, uy = pol2cart(th, ur)

    return np.array([ux, uy, uz])
