import numpy as np
from math import factorial, sqrt
from scipy.special import genlaguerre

from core import Model

# units: nm = meV = T = c = 1
ENERGYFAC = 0.1158 # hbar e / m
E = 300
M = 5.11e8
HBAR = 1.97e5

class Harmonics:
    def __init__(self, lat, inds, amplitudes, gs=None):
        self.lat = lat
        self.amplitudes = np.array(amplitudes)
        self.inds = inds
        if gs is None:
            gs = np.array([lat.point_at(ind) for ind in self.inds])
        self.gs = gs
        self.gx = gs[:,0]
        self.gy = gs[:,1]

    def modsq(self):
        new = {}
        dtype = self.inds[0].dtype
        for ind1, amp1 in zip(self.inds, self.amplitudes):
            for ind2, amp2 in zip(self.inds, self.amplitudes):
                newind = (ind2 - ind1).tobytes()
                new[newind] = new.get(newind, 0) + np.conj(amp1) * amp2
        return Harmonics(self.lat, [np.frombuffer(i, dtype=dtype) for i in new.keys()], list(new.values()))

    def __call__(self, x, y):
        gr = np.tensordot(self.gx, x, axes=0) + np.tensordot(self.gy, y, axes=0)
        return np.tensordot(self.amplitudes, np.exp(1j * gr), axes=1)

    def __rmul__(self, other):
        return Harmonics(self.lat, self.inds, self.amplitudes * other, self.gs)

class SymmHarmonics(Harmonics):
    def __init__(self, lat, symmamplitudes):
        self.symmharmonics = symmamplitudes
        inds = []
        amplitudes = []
        for i, amplitude in enumerate(symmamplitudes):
            ringinds = lat.indices_at_radius(i)
            inds.extend(ringinds)
            amplitudes.extend([amplitude] * len(ringinds))
        super().__init__(lat, inds, amplitudes)

def potential_from_field(harmonics):
    newamplitudes = []
    for gx, gy, amplitude in zip(harmonics.gx, harmonics.gy, harmonics.amplitudes):
        newamplitudes.append(0.5 * (1/(gx or np.inf) + 1j/(gy or np.inf)) * amplitude)
    return Harmonics(harmonics.lat, harmonics.inds, newamplitudes, harmonics.gs)

def m1pow(n):
    return 1 - 2 * (n & 0x1)

def laguerre2(m, n, x, y):
    x = x/np.sqrt(2)
    y = y/np.sqrt(2)
    if n > m:
        s = m1pow(n)
        l = genlaguerre(m, n-m)(x*x + y*y)
        p = np.power(x - 1j*y, n-m)
        f = factorial(m)
    else:
        s = m1pow(m)
        l = genlaguerre(n, m-n)(x*x + y*y)
        p = np.power(x + 1j*y, m-n)
        f = factorial(n)
    return s * l * p * f

def modlaguerre2(m, n, x, y):
    e = np.exp(-x*x/4 - y*y/4)
    return e / sqrt(factorial(m) * factorial(n)) * laguerre2(m, n, x, y)

def harmonics_matel(harmonics, m, n, kx, ky, l0):
    kx = np.array(kx) * l0
    ky = np.array(ky) * l0
    mel = 0
    for ind, gx, gy, amplitude in zip(harmonics.inds, harmonics.gx * l0, harmonics.gy * l0, harmonics.amplitudes):
        eta = -1 if np.all(ind & 0x1) else 1
        l = modlaguerre2(m, n, gx, gy)
        e = np.exp(1j * (gx*ky - gy*kx))
        mel += m1pow(m) * eta * l * e * amplitude
    return mel

def harmonics_mat(harmonics, kx, ky, numll, l0):
    kx = np.array(kx)
    ky = np.array(ky)
    mat = np.zeros(kx.shape + (numll, numll), dtype=complex)
    for m in range(numll):
        for n in range(numll):
            mat[..., m, n] = harmonics_matel(harmonics, m, n, kx, ky, l0)
    return mat

def generalized_lls(b, kx, ky, numll, l0):
    return np.linalg.qr(harmonics_mat(b, kx, ky, numll, l0))[0]

def lowering_op(numll):
    mat = np.zeros((numll, numll), dtype=complex)
    for n in range(1,numll):
        mat[n-1, n] = np.sqrt(n)
    return mat

class LandauLevelModel(Model):
    trivial_basis = False

    def __init__(self, b1, u, numll, gfactor=2):
        self.numll = numll
        self.lattice = b1.lat
        self.u = u
        self.ea1 = E * potential_from_field(b1)
        self.ea1modsq = self.ea1.modsq()
        l0sq = 2 * np.pi / np.linalg.det(self.lattice.trans)
        self.l0 = np.sqrt(l0sq)
        self.low = lowering_op(numll)
        mu = E * HBAR * gfactor / 4 / M
        self.h0 = HBAR * HBAR / M / l0sq * (self.low.T @ self.low + (2 - gfactor) / 4 * np.identity(numll))
        self.z1 = E * HBAR * (2 - gfactor) / 4 / M * b1
        self.pifactor = np.sqrt(2 / l0sq) * HBAR

    def hamiltonian(self, kx, ky):
        kx = np.array(kx)
        ky = np.array(ky)
        ham = np.zeros(kx.shape + (self.numll, self.numll), dtype=complex)
        ham += self.h0
        ham += harmonics_mat(self.ea1modsq, kx, ky, self.numll, self.l0) / 2 / M
        ham += harmonics_mat(self.z1, kx, ky, self.numll, self.l0)
        ham += harmonics_mat(self.u, kx, ky, self.numll, self.l0)
        amat = harmonics_mat(self.ea1, kx, ky, self.numll, self.l0) * self.pifactor / 2 / M
        ham += self.low.T @ amat + amat.conj().swapaxes(-1, -2) @ self.low
        return ham

    def form_factor(self, kx, ky, qx, qy):
        kx = np.array(kx) * self.l0
        ky = np.array(ky) * self.l0
        qx = np.array(qx) * self.l0
        qy = np.array(qy) * self.l0
        mat = np.zeros(kx.shape + (self.numll, self.numll), dtype=complex)
        for m in np.arange(self.numll):
            for n in np.arange(self.numll):
                mat[..., m, n] = m1pow(n) * np.exp(0.5j * (kx * qy - ky * qx)) * modlaguerre2(m, n, qx, qy)
        return mat
