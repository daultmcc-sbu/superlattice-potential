import numpy as np
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize
import matplotlib.pyplot as plt

from .utilities import Register, tr_form_from_eigvec, complex_to_rgb, auto_subplots, remove_outliers, tr_form_from_ratio
from .plotting import Subplot2D
from .banddata import BandData


##############
#### MAIN ####
##############

def make_plot_scan(modelf, band,
        amin, amax, an, alabel, bmin, bmax, bn, blabel, subplots,
        spacing):
    m0 = modelf(0,0)

    a = np.linspace(amin, amax, an)
    b = np.linspace(bmin, bmax, bn)
    av, bv = np.meshgrid(a, b, indexing='ij')

    xf, yf = m0.lattice.bz_grid(spacing)
    bzsize = xf.size
    xf = np.append(xf, m0.lattice.hs_points[:,0])
    yf = np.append(yf, m0.lattice.hs_points[:,1])
    in_bz = np.arange(xf.size) < bzsize

    subplots = [plot() for plot in subplots]
    datas = [np.zeros(av.shape + plot.shape) for plot in subplots]

    for i in range(an):
        for j in range(bn):
            bd = BandData(modelf(a[i], b[j]), xf, yf, band, in_bz)

            for plot, data in zip(subplots, datas):
                data[i,j] = plot.compute(bd)

    fig, axs = auto_subplots(plt, len(subplots))
    for plot, data, ax in zip(subplots, datas, axs.flat):
        norm = plot.norm(data)
        mesh = ax.pcolormesh(av, bv, data, norm=norm, **plot.colormesh_opts)
        if plot.colorbar:
            cb = fig.colorbar(mesh, ax=ax)
        if plot.title is not None:
            ax.set_title(plot.title)
        ax.set_xlabel(alabel)
        ax.set_ylabel(blabel)

    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.90)

    return fig




#######################
#### SUBPLOT SETUP ####
#######################

int_observables = {}

class RegisterIntObservables(Register):
    registry = int_observables

    def id_from_name(name):
        return name.removesuffix('IntObservable').lower()
    
class IntObservable(metaclass=RegisterIntObservables, register=False):
    shape = ()

    colormesh_opts = {}
    colorbar = True
    title = None

    def compute(self, bd):
        pass

    def norm(self, data):
        return Normalize()




##################
#### SUBPLOTS ####
##################

class GapIntObservable(IntObservable):
    title = "Band gap"

    def compute(self, bd):
        return bd.gap
    
class WidthIntObservable(IntObservable):
    title = "Band width"

    def compute(self, bd):
        return bd.width
    
class ChernIntObservable(IntObservable):
    title = "Chern number"
    colormesh_opts = {'cmap': 'RdBu_r'}

    def compute(self, bd):
        return bd.chern
    
    def norm(self, data):
        return BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256, extend='both')

class BerryFlucIntObservable(IntObservable):
    title = "Berry fluc"
    colormesh_opts = {'cmap': 'plasma'}

    def compute(self, bd):
        return bd.berry_fluc
    
    def norm(self, data):
        body = remove_outliers(data)
        return LogNorm(vmin=body.min(), vmax=body.max())

class BerryFlucN1IntObservable(IntObservable):
    title = "Berry fluc (n1)"
    colormesh_opts = {'cmap': 'plasma'}

    def compute(self, bd):
        return bd.berry_fluc_n1

    def norm(self, data):
        body = remove_outliers(data)
        return LogNorm(vmin=body.min(), vmax=body.max())

class TrViolIsoIntObservable(IntObservable):
    title = "Trace cond viol (isometric)"
    colormesh_opts = {'cmap': 'plasma'}

    def compute(self, bd):
        return bd.int(bd.tr_viol_iso)
    
    def norm(self, data):
        return Normalize(0, 20)

class TrViolMinIntObservable(IntObservable):
    title = "Trace cond viol (min)"
    colormesh_opts = {'cmap': 'plasma'}

    def compute(self, bd):
        form = tr_form_from_eigvec(bd.qgt_bzmin_eigvec)
        return bd.int(bd.tr_viol(form))
    
    def norm(self, data):
        return Normalize(0, 20)

class TrViolAvgIntObservable(IntObservable):
    title = "Trace cond viol (avg)"
    colormesh_opts = {'cmap': 'plasma'}

    def compute(self, bd):
        form = tr_form_from_eigvec(bd.avg_qgt_eigvec)
        return bd.int(bd.tr_viol(form))
    
    def norm(self, data):
        return Normalize(0, 20)
    
class TrViolOptIntObservable(IntObservable):
    title = "Trace cond viol (opt)"
    colormesh_opts = {'cmap': 'plasma'}

    def compute(self, bd):
        form = tr_form_from_ratio(*bd.optimal_cstruct)
        return bd.int(bd.tr_viol(form))
    
    def norm(self, data):
        return Normalize(0, 20)
    
class CstructMinIntObservable(IntObservable):
    shape = (3,)
    title = "Complex struct (min)"
    colorbar = False

    def compute(self, bd):
        w = bd.qgt_bzmin_eigvec
        return complex_to_rgb(w[1] / w[0])
    
class CstructAvgIntObservable(IntObservable):
    shape = (3,)
    title = "Complex struct (avg)"
    colorbar = False
    
    def compute(self, bd):
        w = bd.avg_qgt_eigvec
        return complex_to_rgb(w[1] / w[0])
    
class CstructOptIntObservable(IntObservable):
    shape = (3,)
    title = "Complex struct (opt)"
    colorbar = False
    
    def compute(self, bd):
        z = bd.optimal_cstruct
        return complex_to_rgb(z[0] + 1j * z[1])