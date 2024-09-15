import numpy as np
from numpy import ma
from matplotlib import pyplot as plt, patches
from matplotlib.colors import LogNorm, BoundaryNorm, CenteredNorm
import itertools

from .banddata import BandData
from .utilities import *


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

class Subplot2D:
    colormesh_opts = {}
    colorbar = True
    title = ""

    def final_data(self):
        return self.data
    
    def cb_post(self, cb):
        pass

    def ax_post(self, ax):
        pass

def make_subplots_2d(xv, yv, subplots, fig, axs, xlabel, ylabel):
    for ax, subplot in zip(axs, subplots):
        mesh = ax.pcolormesh(xv, yv, subplot.final_data(), **subplot.colormesh_opts)
        if subplot.colorbar:
            cb = fig.colorbar(mesh, ax=ax)
            subplot.cb_post(cb)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subplot.title)
        subplot.ax_post(ax)