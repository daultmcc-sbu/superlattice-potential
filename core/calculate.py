import numpy as np

from .banddata import BandData

def single(model, band, zoom, bn):
    size = zoom * np.max(np.abs(model.lattice.trans @ np.array([1,1])))
    x = np.linspace(-size, size, bn)
    y = np.linspace(-size, size, bn)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    in_bz = model.lattice.in_first_bz(xv,yv)

    return BandData(model, xv, yv, band, in_bz)

def scan_2d(modelf, band, amin, amax, an, bmin, bmax, bn, observs, spacing):
    m0 = modelf(0,0)

    a = np.linspace(amin, amax, an)
    b = np.linspace(bmin, bmax, bn)
    av, bv = np.meshgrid(a, b, indexing='ij')

    xf, yf = m0.lattice.bz_grid(spacing)
    bzsize = xf.size
    xf = np.append(xf, m0.lattice.hs_points[:,0])
    yf = np.append(yf, m0.lattice.hs_points[:,1])
    in_bz = np.arange(xf.size) < bzsize

    observs = [observ(av.shape) for observ in observs]

    for i in range(an):
        for j in range(bn):
            bd = BandData(modelf(a[i], b[j]), xf, yf, band, in_bz)

            for observ in observs:
                observ.update((i,j), bd)

    return av, bv, observs