import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import CenteredNorm, LogNorm

from .banddata import BandData
from .utilities import Register, auto_subplots, tr_form_from_eigvec, complex_to_rgb, tr_form_from_ratio


##############
#### MAIN ####
##############

def make_plot_single(bd, sn, subplots):
    fig, axs = auto_subplots(plt, len(subplots) + 1, size=4.5)
    plot_bandstructure(bd.model, sn, axs.flat[0], highlight=bd.band)

    subplots = [plot(bd) for plot in subplots]
    for plot, ax in zip(subplots, axs.flat[1:]):
        plot.draw(fig, ax)

    return fig





########################
#### BAND STRUCTURE ####
########################

class Path:
    def __init__(self, vertices, closed=True):
        vertices = list(vertices)

        if closed:
            vertices.append(vertices[0])
        
        self.vertices = np.array(vertices)
        self.diffs = np.diff(self.vertices, axis=0)
        self.dists = np.linalg.norm(self.diffs, axis=1)
        self.length = np.sum(self.dists)

    # currently does not check bounds
    def along(self, t):
        t *= self.length
        for i in range(0, len(self.dists)):
            if t > self.dists[i]:
                t -= self.dists[i]
            else:
                return self.vertices[i] + self.diffs[i] * t / self.dists[i]

    def points(self, n):
        ts = np.linspace(0, 1, n)
        points = np.array([self.along(t) for t in ts])
        vertex_ts = np.insert(np.cumsum(self.dists) / self.length, 0, 0)
        return ts, points, vertex_ts

def plot_bandstructure(model, n, ax, highlight=None):
    path = Path(model.lattice.hs_points)
    ts, ks, hs_ts = path.points(n)
    Es = model.spectrum(ks[:,0], ks[:,1])
    ax.plot(ts, Es, c='k')

    if highlight:
        ax.plot(ts, Es[:,highlight], c='r')
        focus = highlight
    else:
        focus = np.argmax(Es[0] > 0)

    focus_range = 1.2 * Es[:, focus-5:focus+6]
    yscale = np.abs(focus_range).max()
    ax.set_ylim(-yscale, yscale)
    
    labels = model.lattice.hs_labels
    labels.append(labels[0])
    ax.set_xticks(ticks=hs_ts, labels=labels)
    for t in hs_ts:
        ax.axvline(x=t, ls=':', c='k')

    ax.set_title("Band structure")
    ax.set_ylabel("Energy")
    
    



#######################
#### SUBPLOT SETUP ####
#######################

single_subplots = {}

class RegisterSingleSubplots(Register):
    registry = single_subplots

    def id_from_name(name):
        return name.removesuffix('SingleSubplot').lower()
    
class SingleSubplot(metaclass=RegisterSingleSubplots, register=False):
    colormesh_opts = {}
    colorbar = True
    title = None
    xlabel = r"$k_x$"
    ylabel = r"$k_y$"

    def __init__(self, bd):
        self.xv = bd.xv
        self.yv = bd.yv
        self.data = self.compute(bd)

    def cb_post(self, cb):
        pass

    def ax_post(self, ax):
        pass

    def draw(self, fig, ax):
        mesh = ax.pcolormesh(self.xv, self.yv, self.data, **self.colormesh_opts)
        if self.colorbar:
            cb = fig.colorbar(mesh, ax=ax)
            self.cb_post(cb)
        if self.title is not None:
            ax.set_title(self.title)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        self.ax_post(ax)






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