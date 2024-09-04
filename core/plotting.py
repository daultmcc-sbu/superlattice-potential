import numpy as np
from numpy import ma
from matplotlib import pyplot as plt, patches
from matplotlib.colors import LogNorm, BoundaryNorm, CenteredNorm
import itertools

from .fast import *
from .utilities import *


###################
#### UTILITIES ####
###################

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

def plots_2d(xv, yv, data, fig, axs, axlabels, xlabel, ylabel, opts=itertools.cycle([{}]), blur=False):
    cbs = []
    for ax, z, name, kwargs in zip(axs, data, axlabels, opts):
        # z[is_outlier(z)] = np.nan
        mesh = ax.pcolormesh(xv, yv, z, 
                             shading='gouraud' if blur else 'auto', **kwargs)
        cb = fig.colorbar(mesh, ax=ax)
        cbs.append(cb)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
    return cbs



####################
#### PROCEDURES ####
####################
        
def make_plot_band_geometry(model, band, zoom, bn, sn, eigvec_loc=None, blur=True):
    fig, axs = plt.subplots(2, 3, figsize=(18,10))
    plot_bandstructure(model, sn, axs[0,0], highlight=band)

    size = zoom * np.max(np.abs(model.lattice.trans @ np.array([1,1])))
    x = np.linspace(-size, size, bn)
    y = np.linspace(-size, size, bn)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    Hs = make_hamiltonians_single(model, xv, yv)
    _, _, _, berry, g11, g22, g12 = fast_energies_and_qgt_raw(Hs, band)
    qgt = qgt_from_raw(berry, g11, g22, g12)
    qm = np.real(qgt)

    qm_det = np.linalg.det(qm)
    qgt_eig = np.linalg.eigvalsh(qgt)[...,0]

    minind = ma.array(qgt_eig, mask=~model.lattice.in_first_bz(xv,yv)).argmin()
    minx, miny = xv.flat[minind], yv.flat[minind]
    eigvec_minw = np.linalg.eigh(qgt.reshape((-1,2,2))[qgt_eig.argmin()])[1][:,0]
    tr_form_minw = tr_form_from_eigvec(eigvec_minw)
    tr_viol_minw = np.abs(np.tensordot(qm, tr_form_minw)) - np.abs(berry)
    
    eigvec_avgw = np.linalg.eigh(np.average(qgt.reshape(-1,2,2), axis=0))[1][:,0]
    tr_form_avgw = tr_form_from_eigvec(eigvec_avgw)
    tr_viol_avgw = np.abs(np.tensordot(qm, tr_form_avgw)) - np.abs(berry)

    plots_2d(xv, yv, 
            [berry, qm_det, qgt_eig, tr_viol_avgw, tr_viol_minw],
            fig, axs.flat[1:],
            ["Berry curvature", "QM det", "Min QGT eigval", "GTC violation (avg vec)", "GTC violation (min vec)"],
            "$k_x$", "$k_y$",
            opts=[{'norm': CenteredNorm(), 'cmap': 'RdBu_r'}, {}, {}, {}, {}],
            blur=blur)
    
    axs[1,0].add_patch(patches.Circle((minx, miny), radius=size/25, color='r'))
    
    plt.subplots_adjust(bottom=0.1, left=0.08, right=0.95, top=0.9)
    
    return fig

def make_plot_sweep_parameters_2d(modelf, band,
          amin, amax, an, alabel, bmin, bmax, bn, blabel,
          spacing, eigvec_loc=False):
    m0 = modelf(0,0)

    a = np.linspace(amin, amax, an)
    b = np.linspace(bmin, bmax, bn)
    av, bv = np.meshgrid(a, b, indexing='ij')

    xf, yf = m0.lattice.bz_grid(spacing)
    print(xf.size)

    above = np.zeros((an, bn))
    below = np.zeros((an, bn))
    width = np.zeros((an, bn))
    chern = np.zeros((an, bn))
    tr_viol_minw = np.zeros((an, bn))
    tr_viol_avgw = np.zeros((an, bn))
    tr_viol_iso = np.zeros((an, bn))
    avg_qgt_eigval = np.zeros((an, bn))
    berry_curv_stdev = np.zeros((an, bn))

    cstruct_minw = np.zeros((an, bn), dtype=m0.dtype)
    cstruct_avgw = np.zeros((an, bn), dtype=m0.dtype)

    for i in range(an):
        for j in range(bn):
            Hs = make_hamiltonians_single(modelf(a[i], b[j]), xf, yf)
            below_, at, above_, berry, g11, g22, g12 = fast_energies_and_qgt_raw(Hs, band)
            above[i,j] = np.min(above_) - np.max(at)
            below[i,j] = np.min(at) - np.max(below_)
            width[i,j] = np.max(at) - np.min(at)
            chern[i,j] = np.sum(berry)

            qgt = qgt_from_raw(berry, g11, g22, g12)
            qm = np.real(qgt)

            qgt_eigvals = np.linalg.eigvalsh(qgt)[...,0]
            avg_qgt_eigval[i,j] = np.sum(qgt_eigvals)
            
            min_ind = np.argmin(qgt_eigvals)
            eigvec_minw = np.linalg.eigh(qgt[min_ind])[1][:,0]
            cstruct_minw[i,j] = eigvec_minw[1] / eigvec_minw[0]
            tr_form_minw = tr_form_from_eigvec(eigvec_minw)
            tr_viol_minw[i,j] = np.sum(np.abs(np.tensordot(qm, tr_form_minw)) - np.abs(berry))

            eigvec_avgw = np.linalg.eigh(np.average(qgt, axis=0))[1][:,0]
            cstruct_avgw[i,j] = eigvec_avgw[1] / eigvec_avgw[0]
            tr_form_avgw = tr_form_from_eigvec(eigvec_avgw)
            tr_viol_avgw[i,j] = np.sum(np.abs(np.tensordot(qm, tr_form_avgw)) - np.abs(berry))

            tr_viol_iso[i,j] = np.sum(np.abs(np.tensordot(qm, np.identity(2))) - np.abs(berry))

            berry_curv_stdev[i,j] = np.std(berry)

    chern = chern * spacing * spacing / 2 / np.pi
    tr_viol_minw = tr_viol_minw * spacing * spacing
    tr_viol_avgw = tr_viol_avgw * spacing * spacing
    tr_viol_iso = tr_viol_iso * spacing * spacing
    avg_qgt_eigval = avg_qgt_eigval * spacing * spacing
    above = np.maximum(0, above)
    below = np.maximum(0, below)
    gap = np.minimum(above, below)
    flatness = gap / width

    gapless = gap < 0.3
    def mask(arr):
        return ma.masked_where(gapless | ~np.isfinite(arr), arr)
    
    # chern = ma.array(chern, mask=gapless)
    # tr_viol_avgw = mask(tr_viol_avgw)
    # tr_viol_minw = mask(tr_viol_minw)
    # tr_viol_iso = mask(tr_viol_iso)
    # avg_qgt_eigval= mask(avg_qgt_eigval)
    # berry_curv_stdev = mask(berry_curv_stdev)

    comb = [tr_viol_minw, tr_viol_avgw, tr_viol_iso]
    comb = ma.concatenate([remove_outliers(arr[arr > 0]) for arr in comb])
    # ideal_norm = LogNorm(comb.min(), comb.max())
    # ideal_opts = {'norm': ideal_norm, 'cmap': 'plasma'}
    ideal_opts = {'vmin': 0, 'vmax': 10, 'cmap': 'plasma'}

    # chern_norm = BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256, extend='both')
    chern_norm = CenteredNorm(halfrange=3)
    fig, axs = plt.subplots(3, 4, figsize=(18,15))
    cbs = plots_2d(av, bv, 
            [gap, width, flatness, chern, 
             berry_curv_stdev, tr_viol_iso, tr_viol_avgw, tr_viol_minw], 
            fig, axs.flat,
            ["Band gap", "Bandwidth", "Flatness", "Chern number", 
             "Berry curv stdev", "GTC violation (iso)", "GTC violation (avg)", "GTC violation (min)"],
            alabel, blabel,
            opts=[{}, {}, {}, {'norm': chern_norm, 'cmap': 'RdBu_r'},
                  {'norm': LogNorm(), 'cmap': 'plasma'}, ideal_opts, ideal_opts, ideal_opts])
    cbs[3].set_ticks([-2,-1,0,1,2])

    axs[2,2].pcolormesh(av, bv, complex_to_rgb(cstruct_avgw))
    axs[2,3].pcolormesh(av, bv, complex_to_rgb(cstruct_minw))
    
    plt.subplots_adjust(bottom=0.1, left=0.08, right=0.95, top=0.9)

    return fig