import numpy as np
import numba as nb

@nb.guvectorize(
    ['(complex64[:,:,:], int32, int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])',
     '(complex128[:,:,:], int64, int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:])'],
    '(m,n,n),()->(),(),(),(),(),(),()',
    target='parallel', cache=True
)
def fast_energies_and_geom(Hs, band, below, at, above, berry, g11, g22, g12):
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

def fast_qgt_bz(model, x, y, band):
    Hs = np.zeros((x.size, y.size, 4, model.bands, model.bands), dtype=model.dtype)
    for i in range(x.size):
        for j in range(y.size):
            Hs[i,j,0] = model.hamiltonian(np.array([x[i], y[j]]))
            Hs[i,j,1] = model.hamiltonian(np.array([x[i] + 1e-4, y[j]]))
            Hs[i,j,2] = model.hamiltonian(np.array([x[i] + 1e-4, y[j] + 1e-4]))
            Hs[i,j,3] = model.hamiltonian(np.array([x[i], y[j] + 1e-4]))

    _, _, _, berry, g11, g22, g12 = fast_energies_and_geom(Hs, band)

    qgt = np.zeros((x.size, y.size, 2, 2), dtype=model.dtype)
    qgt[...,0,0] = g11
    qgt[...,1,1] = g22
    qgt[...,0,1] = g12 + 0.5j * berry
    qgt[...,1,0] = g12 - 0.5j * berry

    return qgt