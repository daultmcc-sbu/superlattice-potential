import numpy as np
import numba as nb

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
    Hs = np.zeros(xv.shape + (4, model.bands, model.bands), dtype=model.dtype)
    it = np.nditer([xv, yv], flags=['multi_index'])
    xo = np.array([0, 1e-4, 1e-4, 0])
    yo = np.array([0, 0, 1e-4, 1e-4])
    for x, y in it:
        Hs[it.multi_index] = model.hamiltonian(x + xo, y + yo)
    return Hs

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