import numpy as np
from matplotlib.colors import BoundaryNorm, LogNorm
import matplotlib.pyplot as plt

from .utilities import Register, tr_form_from_eigvec, complex_to_rgb, auto_subplots, remove_outliers
from .plotting import Subplot2D
from .banddata import BandData

def make_plot_sweep_parameters_2d(modelf, band,
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

    subplots = [plot(an, bn) for plot in subplots]

    for i in range(an):
        for j in range(bn):
            bd = BandData(modelf(a[i], b[j]), xf, yf, band, in_bz)

            for plot in subplots:
                plot.update(i, j, bd)

    fig, axs = auto_subplots(plt, len(subplots))
    for plot, ax in zip(subplots, axs.flat):
        plot.finalize()
        plot.draw(av, bv, fig, ax)
        ax.set_xlabel(alabel)
        ax.set_ylabel(blabel)

    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.95)

    return fig

scan_subplots = {}

class RegisterScanSubplots(Register):
    registry = scan_subplots

    def id_from_name(name):
        return name.removesuffix('ScanSubplot').lower()

class ScanSubplot(Subplot2D, metaclass=RegisterScanSubplots, register=False):
    def __init__(self, an, bn):
        self.data = np.zeros((an, bn))

    def update(self, i, j, bd):
        self.data[i,j] = self.compute(bd)

    def finalize(self):
        pass

class GapScanSubplot(ScanSubplot):
    title = "Band gap"

    def compute(self, bd):
        return np.minimum(bd.above.min() - bd.at.max(), bd.at.min() - bd.below.max())
    
class WidthScanSubplot(ScanSubplot):
    title = "Band width"

    def compute(self, bd):
        return bd.at.max() - bd.at.min()
    
class ChernScanSubplot(ScanSubplot):
    title = "Chern number"
    colormesh_opts = {
        'cmap': 'RdBu_r', 
        'norm': BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256, extend='both')}

    def compute(self, bd):
        return bd.chern

class BerryStdevScanSubplot(ScanSubplot):
    title = "Berry curvature stdev"
    colormesh_opts = {'cmap': 'plasma'}

    def compute(self, bd):
        return bd.berry.std()
    
    def finalize(self):
        body = remove_outliers(self.data)
        self.colormesh_opts['norm'] = LogNorm(vmin=body.min(), vmax=body.max())

class TrViolIsoScanSubplot(ScanSubplot):
    title = "Trace cond viol (isometric)"
    colormesh_opts = {'vmin': 0, 'vmax': 10, 'cmap': 'plasma'}

    def compute(self, bd):
        viol = np.abs(np.tensordot(bd.qm, np.identity(2))) - np.abs(bd.berry)
        return np.sum(viol) * bd.sample_area

class TrViolMinScanSubplot(ScanSubplot):
    title = "Trace cond viol (min)"
    colormesh_opts = {'vmin': 0, 'vmax': 10, 'cmap': 'plasma'}

    def compute(self, bd):
        form = tr_form_from_eigvec(bd.qgt_bzmin_eigvec)
        viol = np.abs(np.tensordot(bd.qm, form)) - np.abs(bd.berry)
        return np.sum(viol) * bd.sample_area

class TrViolAvgScanSubplot(ScanSubplot):
    title = "Trace cond viol (avg)"
    colormesh_opts = {'vmin': 0, 'vmax': 10, 'cmap': 'plasma'}

    def compute(self, bd):
        form = tr_form_from_eigvec(bd.avg_qgt_eigvec)
        viol = np.abs(np.tensordot(bd.qm, form)) - np.abs(bd.berry)
        return np.sum(viol) * bd.sample_area
    
class CstructMinScanSubplot(ScanSubplot):
    title = "Complex struct (min)"
    colorbar = False

    def __init__(self, xn, yn):
        self.data = np.zeros((xn, yn, 3))

    def compute(self, bd):
        w = bd.qgt_bzmin_eigvec
        return complex_to_rgb(w[1] / w[0])
    
class CstructAvgScanSubplot(ScanSubplot):
    title = "Complex struct (avg)"
    colorbar = False

    def __init__(self, xn, yn):
        self.data = np.zeros((xn, yn, 3))

    def compute(self, bd):
        w = bd.avg_qgt_eigvec
        return complex_to_rgb(w[1] / w[0])