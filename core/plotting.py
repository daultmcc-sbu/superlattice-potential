import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, CenteredNorm
import itertools

from .fast import *


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
    # Es = np.array([model.spectrum(k) for k in ks])
    Es = fast_energies(model.hamiltonian(ks[:,0], ks[:,1]))
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

def plot_bz(model, f, n, fig, ax, zoom=0.5, blur=True):
    next_gamma = np.abs(model.lattice.trans @ np.array([1,1]))
    x = np.linspace(-zoom * next_gamma[0], zoom * next_gamma[0], n)
    y = np.linspace(-zoom * next_gamma[1], zoom * next_gamma[1], n)
    xv, yv = np.meshgrid(x,y, indexing='ij')
    z = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            z[i,j] = f(model, np.array([x[i], y[j]]))
    mesh = ax.pcolormesh(xv, yv, z, shading='gouraud' if blur else 'auto')
    fig.colorbar(mesh, ax=ax)

def plot_grid_data(grid, z, fig, ax):
    mesh = ax.pcolormesh(grid.points[...,0], grid.points[...,1], z)
    fig.colorbar(mesh, ax=ax)

# def is_outlier(points, thresh=0.5):
    # if len(points.shape) == 1:
    #     points = points[:,None]
    # median = np.median(points, axis=0)
    # diff = np.sum((points - median)**2, axis=-1)
    # diff = np.sqrt(diff)
    # med_abs_deviation = np.median(diff)

    # modified_z_score = 0.6745 * diff / med_abs_deviation

    # return modified_z_score > thresh
def remove_outliers(points, fraction=0.02):
    n = np.maximum(1, int(points.size * fraction))
    sorted = np.sort(points)
    return sorted[:-n]

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

    size = np.max(np.abs(model.lattice.trans @ np.array([1,1])))
    x = np.linspace(-zoom * size, zoom * size, bn)
    y = np.linspace(-zoom * size, zoom * size, bn)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    Hs = make_hamiltonians_single(model, xv, yv)
    _, _, _, berry, g11, g22, g12 = fast_energies_and_qgt_raw(Hs, band)
    qgt = qgt_from_raw(berry, g11, g22, g12)
    qm = np.real(qgt)

    qm_det = np.linalg.det(qm)
    qgt_eig = np.linalg.eigvalsh(qgt)[...,0]

    if eigvec_loc is not None:
        eigvec = np.linalg.eigh(model.qgt(eigvec_loc, band))[1][:,0]
    else:
        eigvec = np.linalg.eigh(qgt.reshape((-1,2,2))[qgt_eig.argmin()])[1][:,0]
        
    tr_form = np.array([
         [np.abs(eigvec[0])**2, np.real(np.conjugate(eigvec[0]) * eigvec[1])],
         [np.real(np.conjugate(eigvec[0]) * eigvec[1]), np.abs(eigvec[1])**2]])
    
    tr_viol = np.abs(np.tensordot(qm, tr_form)) - berry
    null_viol = np.linalg.norm(qgt @ eigvec, axis=-1)

    plots_2d(xv, yv, 
            [berry, qm_det, qgt_eig, null_viol, tr_viol],
            fig, axs.flat[1:],
            ["Berry curvature", "QM det", "Min QGT eigval", "Eigvec violation", "2-form violation"],
            "$k_x$", "$k_y$", 
            opts=[{'norm': CenteredNorm(), 'cmap': 'RdBu_r'}, {}, {}, {}, {}],
            blur=blur)
    
    plt.subplots_adjust(bottom=0.1, left=0.08, right=0.95, top=0.9)
    
    return fig

def make_plot_sweep_parameters_2d_grid(modelf, band,
          xmin, xmax, xn, xlabel, ymin, ymax, yn, ylabel,
          grid_size):
    x = np.linspace(xmin, xmax, xn)
    y = np.linspace(ymin, ymax, yn)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    above = np.zeros((xn, yn))
    below = np.zeros((xn, yn))
    width = np.zeros((xn, yn))
    chern = np.zeros((xn, yn))

    for i in range(xn):
        for j in range(yn):
            d = modelf(x[i], y[j]).solve(grid_size)
            above[i,j] = d.gap_above(band)
            below[i,j] = d.gap_above(band - 1)
            width[i,j] = d.bandwidth(band)
            chern[i,j] = d.chern_number(band)

    above = np.maximum(0, above)
    below = np.maximum(0, below)
    min_gap = np.minimum(above, below)

    fig, axs = plt.subplots(2, 3)
    plots_2d(xv, yv, 
            [above, below, min_gap, width], 
            fig, axs.flat,
            ["Gap above", "Gap below", "Min gap", "Band width"],
            xlabel, xlabel)
    
    norm = BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256, extend='both')
    mesh = axs[1,1].pcolormesh(xv, yv, chern, norm=norm, cmap='RdBu_r')
    fig.colorbar(mesh, ax=axs[1,1])
    axs[1,1].set_title("Chern number")
    axs[1,1].set_xlabel(xlabel)
    axs[1,1].set_ylabel(ylabel)
    
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
    avg_tr_viol_qm = np.zeros((an, bn))
    avg_tr_viol_berry = np.zeros((an, bn))
    avg_norm_viol = np.zeros((an, bn))
    qgt_min_eigval = np.zeros((an,bn))
    qgt_avg_eigval = np.zeros((an, bn))

    for i in range(an):
        for j in range(bn):
            Hs = make_hamiltonians_single(modelf(a[i], b[j]), xf, yf)
            below_, at, above_, berry, g11, g22, g12 = fast_energies_and_qgt_raw(Hs, band)
            above[i,j] = np.min(above_) - np.max(at)
            below[i,j] = np.min(at) - np.max(below_)
            width[i,j] = np.max(at) - np.min(at)
            chern[i,j] = np.sum(berry)

            qgt = qgt_from_raw(berry, g11, g22, g12)
            qgt_eigvals = np.linalg.eigvalsh(qgt)[...,0]
            min_ind = np.argmin(qgt_eigvals)
            qgt_min_eigval[i,j] = qgt_eigvals[min_ind]
            qgt_avg_eigval[i,j] = np.mean(qgt_eigvals)
            eigvec = np.linalg.eigh(qgt[min_ind])[1][:,0]
            tr_form = np.array([
                [np.abs(eigvec[0])**2, np.real(np.conjugate(eigvec[0]) * eigvec[1])],
                [np.real(np.conjugate(eigvec[0]) * eigvec[1]), np.abs(eigvec[1])**2]])
            qm = np.real(qgt)
            tr = np.abs(np.tensordot(qm, tr_form))
            norm_viol = np.linalg.norm(qgt @ eigvec, axis=-1)
            avg_tr_viol_qm[i,j] = np.mean(tr - np.sqrt(np.linalg.det(qm)))
            avg_tr_viol_berry[i,j] = np.mean(tr - 0.5 * np.abs(berry))
            avg_norm_viol[i,j] = np.mean(norm_viol)

    chern = chern * spacing * spacing / 2 / np.pi
    above = np.maximum(0, above)
    below = np.maximum(0, below)
    gap = np.minimum(above, below)
    flatness = gap / width

    comb = [qgt_avg_eigval, qgt_min_eigval, avg_norm_viol, avg_tr_viol_qm, avg_tr_viol_berry]
    comb = np.concatenate([remove_outliers(arr[np.isfinite(arr) & (arr > 0)]) for arr in comb])
    ideal_norm = LogNorm(comb.min(), comb.max())
    ideal_opts = {'norm': ideal_norm, 'cmap': 'plasma'}

    chern_norm = BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256, extend='both')
    fig, axs = plt.subplots(2, 4)
    plots_2d(av, bv, 
            [gap, flatness, chern, qgt_avg_eigval, 
             avg_norm_viol, avg_tr_viol_qm, avg_tr_viol_berry, qgt_min_eigval], 
            fig, axs.flat,
            ["Band gap", "Flatness", "Chern number", "Avg QGT eigval", 
             "Nullity violation", "GTC violation (QM)", "GTC violation (Berry)", "Min QGT eigval"],
            alabel, blabel,
            opts=[{}, {}, {'norm': chern_norm, 'cmap': 'RdBu_r'}] + 5 * [ideal_opts])

    return fig