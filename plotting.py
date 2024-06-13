import numpy as np
from matplotlib import pyplot as plt

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
    xv, yv = np.meshgrid(x,y)
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