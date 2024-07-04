import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from .fast import *


###################
#### UTILITIES ####
###################

class Path:
    def __init__(self, vertices, closed=True):
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
    path = Path(list(model.lattice.hs_points.values()))
    ts, ks, hs_ts = path.points(n)
    Es = np.array([model.spectrum(k) for k in ks])
    ax.plot(ts, Es, c='k')

    if highlight:
        ax.plot(ts, Es[:,highlight], c='r')

    ax.set_ylim(-0.035, 0.035)
    labels = list(model.lattice.hs_points.keys())
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

def plots_2d(xv, yv, data, fig, axs, axlabels, xlabel, ylabel, blur=False):
    for ax, z, name in zip(axs, data, axlabels):
        mesh = ax.pcolormesh(xv, yv, z, shading='gouraud' if blur else 'auto')
        fig.colorbar(mesh, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name)




####################
#### PROCEDURES ####
####################
        
def make_plot_band_geometry(model, band, zoom, k_n, band_n, eigvec_loc=None, blur=True, fast=True):
    fig, axs = plt.subplots(2, 3, figsize=(18,10))
    plot_bandstructure(model, band_n, axs[0,0], highlight=band)

    size = np.max(np.abs(model.lattice.trans @ np.array([1,1])))
    x = np.linspace(-zoom * size, zoom * size, k_n)
    y = np.linspace(-zoom * size, zoom * size, k_n)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    if fast:
        qgt = fast_qgt_bz(model, x, y, band)
    else:
        qgt = np.zeros((k_n, k_n, 2, 2), dtype=model.dtype)

    berry = np.zeros((k_n, k_n))
    qm_det = np.zeros((k_n, k_n))
    qgt_eig = np.zeros((k_n, k_n))

    min_qgt_eig = np.inf

    for i in range(k_n):
        for j in range(k_n):
            if not fast:
                qgt[i,j] = model.qgt(np.array([x[i], y[j]]), band)
                
            eigs, vecs = np.linalg.eigh(qgt[i,j])
            eig = eigs[0]
            vec = vecs[:,0]

            berry[i,j] = 2 * np.imag(qgt[i,j,0,1])
            qm_det[i,j] = np.linalg.det(np.real(qgt[i,j]))
            qgt_eig[i,j] = eig

            if eig < min_qgt_eig and model.lattice.in_first_bz(np.array([x[i], y[j]])):
                 min_qgt_eig = eig
                 min_qgt_eigvec = vec

    tr_viol = np.zeros((k_n, k_n))
    null_viol = np.zeros((k_n, k_n))

    if eigvec_loc is not None:
        eigvec = np.linalg.eigh(model.qgt(eigvec_loc, band))[1][:,0]
    else:
        eigvec = min_qgt_eigvec

    tr_form = np.array([
         [np.abs(eigvec[0])**2, np.real(np.conjugate(eigvec[0]) * eigvec[1])],
         [np.real(np.conjugate(eigvec[0]) * eigvec[1]), np.abs(eigvec[1])**2]])

    for i in range(k_n):
        for j in range(k_n):
            # tr_viol[i,j] = np.abs(np.tensordot(np.real(qgt[i,j]), tr_form) - berry[i,j])
            tr_viol[i,j] = np.sqrt(qm_det[i,j]) - np.abs(np.tensordot(np.real(qgt[i,j]), tr_form))
            null_viol[i,j] = np.linalg.norm(qgt[i,j] @ eigvec)

    plots_2d(xv, yv, 
            [berry, qm_det, qgt_eig, null_viol, tr_viol],
            fig, axs.flat[1:],
            ["Berry curvature", "QM det", "Min QGT eigval", "Eigvec violation", "2-form violation"],
            "$k_x$", "$k_y$", blur=blur)
    
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
    
    norm = colors.BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256, extend='both')
    mesh = axs[1,1].pcolormesh(xv, yv, chern, norm=norm, cmap='RdBu_r')
    fig.colorbar(mesh, ax=axs[1,1])
    axs[1,1].set_title("Chern number")
    axs[1,1].set_xlabel(xlabel)
    axs[1,1].set_ylabel(ylabel)
    
    return fig