import numpy as np

def mod_pi(x):
    y = x / 2 / np.pi
    return 2 * np.pi * (y - np.round(y))

def remove_outliers(points, fraction=0.02):
    n = np.maximum(1, int(points.size * fraction))
    sorted = np.sort(points)
    return sorted[:-n]