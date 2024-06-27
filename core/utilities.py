import numpy as np

def mod_pi(x):
    y = x / 2 / np.pi
    return 2 * np.pi * (y - np.round(y))