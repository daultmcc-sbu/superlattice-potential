import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import CenteredNorm, LogNorm

from .banddata import BandData
from .utilities import Register, auto_subplots, tr_form_from_eigvec, complex_to_rgb, tr_form_from_ratio
from .plotting import Subplot2D, plot_bandstructure


##############
#### MAIN ####
##############

def make_plot_single(model, band, zoom, bn, sn, subplots):
    fig, axs = auto_subplots(plt, len(subplots) + 1, size=4.5)
    plot_bandstructure(model, sn, axs.flat[0], highlight=band)

    size = zoom * np.max(np.abs(model.lattice.trans @ np.array([1,1])))
    x = np.linspace(-size, size, bn)
    y = np.linspace(-size, size, bn)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    in_bz = model.lattice.in_first_bz(xv,yv)

    bd = BandData(model, xv, yv, band, in_bz)
    subplots = [plot(bd) for plot in subplots]
    for plot, ax in zip(subplots, axs.flat[1:]):
        plot.draw(fig, ax)

    return fig, bd




#######################
#### SUBPLOT SETUP ####
#######################

single_subplots = {}

class RegisterSingleSubplots(Register):
    registry = single_subplots

    def id_from_name(name):
        return name.removesuffix('SingleSubplot').lower()
    
class SingleSubplot(Subplot2D, metaclass=RegisterSingleSubplots, register=False):
    xlabel = r"$k_x$"
    ylabel = r"$k_y$"

    def __init__(self, bd):
        self.xv = bd.xv
        self.yv = bd.yv
        self.data = self.compute(bd)




##################
#### SUBPLOTS ####
##################

class BerrySingleSubplot(SingleSubplot):
    title = "Berry curvature"
    colormesh_opts = {'norm': CenteredNorm(), 'cmap': 'RdBu_r'}

    def compute(self, bd):
        return bd.berry

class QmDetSingleSubplot(SingleSubplot):
    title = "Quantum metric determinant"

    def compute(self, bd):
        return np.linalg.det(bd.qm)

class QgtEigvalSingleSubplot(SingleSubplot):
    title = "Minimum QGT eigval"

    def __init__(self, bd):
        super().__init__(self, bd)
        self.minpos = (bd.xv.flat[bd.qgt_min_eigval_index], bd.yv.flat[bd.qgt_min_eigval_index])
        self.size = bd.xv[0,-1]

    def compute(self, bd):
        return bd.qgt_eigval

    def ax_post(self, ax):
        ax.add_patch(patches.Circle(self.minpos, radius=self.size/25, color='r'))

class TrViolIsoSingleSubplot(SingleSubplot):
    title = "Trace cond viol (isometric)"

    def compute(self, bd):
        return bd.tr_viol_iso

class TrViolMinSingleSubplot(SingleSubplot):
    title = "Trace cond viol (min)"
    colormesh_opts = {'vmin': 0, 'vmax': 10}

    def compute(self, bd):
        form = tr_form_from_eigvec(bd.qgt_bzmin_eigvec)
        return bd.tr_viol(form)

class TrViolAvgSingleSubplot(SingleSubplot):
    title = "Trace cond viol (avg)"
    colormesh_opts = {'vmin': 0, 'vmax': 10}

    def compute(self, bd):
        form = tr_form_from_eigvec(bd.avg_qgt_eigvec)
        return bd.tr_viol(form)
    
class TrViolOptSingleSubplot(SingleSubplot):
    title = "Trace cond viol (opt)"
    colormesh_opts = {'cmap': 'plasma'}

    def compute(self, bd):
        form = tr_form_from_ratio(*bd.optimal_cstruct)
        return bd.tr_viol(form)
    
class CstructSingleSubplot(SingleSubplot):
    title = "Complex struct (min)"
    colorbar = False

    def compute(self, bd):
        w = bd.qgt_eigvec
        return complex_to_rgb(w[...,1] / w[...,0])
    
class TrViolByCstructSingleSubplot(SingleSubplot):
    title = "Tr viol per cstruct"
    xlabel = r"$\Re [\omega_2/\omega_1]$"
    ylabel = r"$\Im [\omega_2/\omega_1]$"
    colormesh_opts = {'cmap': 'plasma', 'norm': LogNorm()}

    def __init__(self, bd):
        N = 50
        x = np.linspace(-3, 3, N)
        y = np.linspace(-3, 3, N)
        self.xv, self.yv = np.meshgrid(x, y, indexing='ij')
        self.data = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                form = tr_form_from_ratio(x[i], y[j])
                viol = np.abs(np.tensordot(bd.qm, form) - bd.berry)
                self.data[i,j] = viol[bd.in_bz].sum() * bd.sample_area
        self.minpos = bd.optimal_cstruct
        self.size = self.xv[0,-1]

    def ax_post(self, ax):
        ax.add_patch(patches.Circle(self.minpos, radius=self.size/25, color='r'))