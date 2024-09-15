import numpy as np
import numba as nb
import numpy.ma as ma

from .utilities import *

@nb.guvectorize(
    ['(complex64[:,:,:], int32, float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32[:])',
     '(complex128[:,:,:], int64, float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])'],
    '(m,n,n),()->(),(),(),(),(),(),()',
    target='parallel', cache=True
)
def fast_energies_and_qgt_raw(Hs, band, below, at, above, berry, g11, g22, g12):
    ceigvals, ceigvecs = np.linalg.eigh(Hs[0,:,:])
    below[0] = ceigvals[band-1]
    at[0] = ceigvals[band]
    above[0] = ceigvals[band+1]

    vec00 = ceigvecs[:,band]
    vec10 = np.linalg.eigh(Hs[1,:,:])[1][:,band]
    vec11 = np.linalg.eigh(Hs[2,:,:])[1][:,band]
    vec01 = np.linalg.eigh(Hs[3,:,:])[1][:,band]

    link0010 = np.vdot(vec00, vec10)
    link1011 = np.vdot(vec10, vec11)
    link1101 = np.vdot(vec11, vec01)
    link0100 = np.vdot(vec01, vec00)
    link0011 = np.vdot(vec00, vec11)

    berry[0] = 1e8 * np.angle(link0010 * link1011 * link1101 * link0100)

    g11_ = 2e8 * (1 - np.abs(link0010))
    g22_ = 2e8 * (1 - np.abs(link0100))
    g11[0] = g11_
    g22[0] = g22_
    g12[0] = 1e8 * (1 - np.abs(link0011)) - 0.5 * (g11_ + g22_)

def make_hamiltonians_single(model, xv, yv):
    xo = np.array([0, 1e-4, 1e-4, 0])
    yo = np.array([0, 0, 1e-4, 1e-4])
    return model.hamiltonian(xv[...,np.newaxis] + xo, yv[...,np.newaxis] + yo)

def make_hamiltonians_sweep(modelf, paramvs, xv, yv):
    m0 = modelf(*[pv.flat[0] for pv in paramvs])
    Hs = np.zeros(paramvs[0].shape + xv.shape + (4, m0.bands, m0.bands), dtype=m0.dtype)
    it = np.nditer(paramvs, flags=['multi_index'])
    for params in it:
        model = modelf(*params)
        Hs[it.multi_index] = make_hamiltonians_single(model, xv, yv)
    return Hs

def qgt_from_raw(berry, g11, g22, g12):
    dt = (berry.dtype.type(1) * np.complex64(1j)).dtype

    qgt = np.zeros(berry.shape + (2, 2), dtype=dt)
    qgt[...,0,0] = g11
    qgt[...,1,1] = g22
    qgt[...,0,1] = g12 + 0.5j * berry
    qgt[...,1,0] = g12 - 0.5j * berry

    return qgt

class BandData:
    def __init__(self, model, xv, yv, band, in_bz=None):
        Hs = make_hamiltonians_single(model, xv, yv)
        self.model = model
        self.xv = xv
        self.yv = yv
        if in_bz is None:
            self.in_bz = np.full(xv.shape, True)
        else:
            self.in_bz = in_bz
        self.sample_area = model.lattice.bz_area / self.in_bz.sum()
        self.below, self.at, self.above, self.berry, self.g11, self.g22, self.g12 = fast_energies_and_qgt_raw(Hs, band)

    @cached_property
    def qgt(self):
        return qgt_from_raw(self.berry, self.g11, self.g22, self.g12)

    @cached_property
    def qm(self):
        return np.real(self.qgt)

    @cached_property
    def chern(self):
        return self.berry[self.in_bz].sum() * self.sample_area / 2 / np.pi

    @cached_property
    def qgt_eigval(self):
        return np.linalg.eigvalsh(self.qgt)[...,0]
    
    @cached_property
    def qgt_eigvec(self):
        vals, vecs = np.linalg.eigh(self.qgt)
        self.qgt_eigval = vals[...,0]
        return vecs[...,0]

    @cached_property
    def avg_qgt(self):
        return np.average(self.qgt[self.in_bz], axis=0)

    @cached_property
    def avg_qgt_eigvec(self):
        return np.linalg.eigh(self.avg_qgt)[1][:,0]
    
    @cached_property
    def qgt_min_eigval_index(self):
        return ma.array(self.qgt_eigval, mask=~self.in_bz).argmin()
    
    @cached_property
    def qgt_bzmin_eigvec(self):
        return np.linalg.eigh(self.qgt.reshape((-1,2,2))[self.qgt_min_eigval_index])[1][:,0]