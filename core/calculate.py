import numpy as np

from .banddata import BandData

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

def single_bands(model, n):
    path = Path(model.lattice.hs_points)
    ts, ks, hs_ts = path.points(n)
    Es = model.spectrum(ks[:,0], ks[:,1])
    return ts, hs_ts, Es

def single_bz(model, band, zoom, bn):
    size = zoom * np.max(np.abs(model.lattice.trans @ np.array([1,1])))
    x = np.linspace(-size, size, bn)
    y = np.linspace(-size, size, bn)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    in_bz = model.lattice.in_first_bz(xv,yv)
    return BandData(model, xv, yv, band, in_bz)

def scan(modelf, band, bounds, observs, spacing):
    m0 = modelf(*np.zeros(len(bounds)))

    marks = [np.linspace(lower, upper, n) for (lower, upper, n) in bounds]
    grids = np.meshgrid(*marks, indexing='ij')

    xf, yf = m0.lattice.bz_grid(spacing)
    bzsize = xf.size
    xf = np.append(xf, m0.lattice.hs_points[:,0])
    yf = np.append(yf, m0.lattice.hs_points[:,1])
    in_bz = np.arange(xf.size) < bzsize

    observs = [observ(grids[0].shape) for observ in observs]

    it = np.nditer(grids, flags=['multi_index'])
    for params in it:
        bd = BandData(modelf(*np.atleast_1d(params)), xf, yf, band, in_bz)
        for observ in observs:
            observ.update(it.multi_index, bd)

    return grids, observs