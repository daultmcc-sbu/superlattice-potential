import numpy as np
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt

from .utilities import Register, tr_form_from_eigvec, complex_to_rgb, subplots_size
from .plotting import Subplot2D, make_subplots_2d
from .banddata import BandData

def make_plot_sweep_parameters_2d(modelf, band,
        amin, amax, an, alabel, bmin, bmax, bn, blabel, subplots,
        spacing):
    m0 = modelf(0,0)

    a = np.linspace(amin, amax, an)
    b = np.linspace(bmin, bmax, bn)
    av, bv = np.meshgrid(a, b, indexing='ij')

    xf, yf = m0.lattice.bz_grid(spacing)

    subplots = [plot(an, bn) for plot in subplots]

    for i in range(an):
        for j in range(bn):
            bd = BandData(modelf(a[i], b[j]), xf, yf, band)

            for plot in subplots:
                plot.update(i, j, bd)

    fig, axs = plt.subplots(*subplots_size(len(subplots)))
    make_subplots_2d(av, bv, subplots, fig, axs.flat, alabel, blabel)

    return fig

scan_subplots = {}

class RegisterScanSubplots(Register):
    registry = scan_subplots

    def id_from_name(name):
        return name.removesuffix('ScanSubplot').lower()

class ScanSubplot(Subplot2D, metaclass=RegisterScanSubplots, register=False):
    def __init__(self, xn, yn):
        self.data = np.zeros((xn, yn))

    def update(self, i, j, bd):
        self.data[i,j] = self.compute(bd)

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

    def compute(self, bd):
        return np.stdev(bd.berry)

class TrViolIsoScanSubplot(ScanSubplot):
    title = "Trace cond viol (isometric)"

    def compute(self, bd):
        viol = np.abs(np.tensordot(bd.qm, np.identity(2))) - np.abs(bd.berry)
        return np.sum(viol) * bd.sample_area

class TrViolMinScanSubplot(ScanSubplot):
    title = "Trace cond viol (min)"
    colormesh_opts = {'vmin': 0, 'vmax': 10}

    def compute(self, bd):
        form = tr_form_from_eigvec(bd.qgt_bzmin_eigvec)
        viol = np.abs(np.tensordot(bd.qm, form)) - np.abs(bd.berry)
        return np.sum(viol) * bd.sample_area

class TrViolAvgScanSubplot(ScanSubplot):
    title = "Trace cond viol (avg)"
    colormesh_opts = {'vmin': 0, 'vmax': 10}

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