import numpy as np
from matplotlib.colors import hsv_to_rgb

def mod_pi(x):
    y = x / 2 / np.pi
    return 2 * np.pi * (y - np.round(y))

def remove_outliers(points, fraction=0.02):
    n = np.maximum(1, int(points.size * fraction))
    sorted = np.sort(points)
    return sorted[:-n]

def tr_form_from_eigvec(eigvec):
    cross = np.conjugate(eigvec[0]) * eigvec[1]
    sqrt_det = np.imag(cross)
    base = np.array([
         [np.abs(eigvec[0])**2, np.real(cross)],
         [np.real(cross), np.abs(eigvec[1])**2]], dtype=eigvec.dtype)
    return base / sqrt_det

def complex_to_rgb(z):
    l = 2 * np.arctan(np.abs(z)) / np.pi
    s = 2 * np.minimum(l, 1-l)
    v = np.minimum(2 * l, 1)
    h = np.angle(z) / 2 / np.pi + 0.5
    hsv = np.stack([h,s,v], axis=-1)
    return hsv_to_rgb(hsv)