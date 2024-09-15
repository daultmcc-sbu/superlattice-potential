import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import CenteredNorm

from .banddata import BandData
from .utilities import Register, subplots_size, tr_form_from_eigvec, complex_to_rgb
from .plotting import Subplot2D, plot_bandstructure, make_subplots_2d

def make_plot_band_geometry(model, band, zoom, bn, sn, subplots):
    fig, axs = plt.subplots(*subplots_size(len(subplots) + 1))
    plot_bandstructure(model, sn, axs.flat[0], highlight=band)

    size = zoom * np.max(np.abs(model.lattice.trans @ np.array([1,1])))
    x = np.linspace(-size, size, bn)
    y = np.linspace(-size, size, bn)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    in_bz = model.lattice.in_first_bz(xv,yv)

    bd = BandData(model, xv, yv, band, in_bz)
    subplots = [plot(bd) for plot in subplots]
    make_subplots_2d(xv, yv, subplots, fig, axs.flat[1:], r"$k_x$", r"$k_y$")

    return fig, bd

bz_subplots = {}

class RegisterBzSubplots(Register):
    registry = bz_subplots

    def id_from_name(name):
        return name.removesuffix('BzSubplot').lower()
    
class BzSubplot(Subplot2D, metaclass=RegisterBzSubplots, register=False):
    def __init__(self, bd):
        pass

class BerryBzSubplot(BzSubplot):
    title = "Berry curvature"
    colormesh_opts = {'norm': CenteredNorm(), 'cmap': 'RdBu_r'}

    def __init__(self, bd):
        self.data = bd.berry

class QmDetBzSubplot(BzSubplot):
    title = "Quantum metric determinant"

    def __init__(self, bd):
        self.data = np.linalg.det(bd.qm)

class QgtEigvalBzSubplot(BzSubplot):
    title = "Minimum QGT eigval"

    def __init__(self, bd):
        self.minpos = (bd.xv.flat[bd.qgt_min_eigval_index], bd.yv.flat[bd.qgt_min_eigval_index])
        self.size = bd.xv[0,-1]
        self.data = bd.qgt_eigval

    def ax_post(self, ax):
        ax.add_patch(patches.Circle(self.minpos, radius=self.size/25, color='r'))

class TrViolIsoBzSubplot(BzSubplot):
    title = "Trace cond viol (isometric)"

    def __init__(self, bd):
        self.data = np.abs(np.tensordot(bd.qm, np.identity(2))) - np.abs(bd.berry)

class TrViolMinBzSubplot(BzSubplot):
    title = "Trace cond viol (min)"
    colormesh_opts = {'vmin': 0, 'vmax': 10}

    def __init__(self, bd):
        form = tr_form_from_eigvec(bd.qgt_bzmin_eigvec)
        self.data = np.abs(np.tensordot(bd.qm, form)) - np.abs(bd.berry)

class TrViolAvgBzSubplot(BzSubplot):
    title = "Trace cond viol (avg)"
    colormesh_opts = {'vmin': 0, 'vmax': 10}

    def __init__(self, bd):
        form = tr_form_from_eigvec(bd.avg_qgt_eigvec)
        self.data = np.abs(np.tensordot(bd.qm, form)) - np.abs(bd.berry)
    
class CstructBzSubplot(BzSubplot):
    title = "Complex struct (min)"
    colorbar = False

    def __init__(self, bd):
        w = bd.qgt_eigvec
        self.data = complex_to_rgb(w[...,1] / w[...,0])