import numpy as np
from math import ceil
from matplotlib.colors import hsv_to_rgb

def remove_outliers(points, fraction=0.02):
    n = int(points.size * fraction)
    sorted = np.sort(points, axis=None)
    return sorted[n:sorted.size-n]

# def tr_form_from_eigvec(eigvec):
#     cross = np.conjugate(eigvec[0]) * eigvec[1]
#     sqrt_det = np.imag(cross)
#     base = np.array([
#          [np.abs(eigvec[0])**2, np.real(cross)],
#          [np.real(cross), np.abs(eigvec[1])**2]], dtype=eigvec.dtype)
#     return base / sqrt_det

def tr_form_from_ratio(zr, zi):
    return np.array([[1, zr], [zr, zr*zr + zi*zi]]) / zi

def tr_form_from_eigvec(vec):
    z = vec[1] / vec[0]
    return tr_form_from_ratio(np.real(z), np.imag(z))

def complex_to_rgb(z):
    l = 2 * np.arctan(np.abs(z)) / np.pi
    s = 2 * np.minimum(l, 1-l)
    v = np.minimum(2 * l, 1)
    h = np.angle(z) / 2 / np.pi + 0.5
    hsv = np.stack([h,s,v], axis=-1)
    return hsv_to_rgb(hsv)

def auto_subplots(plt, n, size=5):
    cols = ceil(np.sqrt(n))
    rows = ceil(n / cols)
    return plt.subplots(rows, cols, figsize=(cols*size,rows*size))

class Register(type):
    registry = None

    def id_from_name(name):
        return name
    
    def __new__(cls, name, bases, attrs, id=None, register=True):
        newcls = super().__new__(cls, name, bases, attrs)

        if id is None:
            id = cls.id_from_name(name)

        if register:
            cls.registry[id] = newcls

        return newcls

class cached_property(object):
    _missing = object()

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value