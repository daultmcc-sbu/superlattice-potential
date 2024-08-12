import numpy as np

from core import Model

# arXiv:1205.6953
# UNITS: nm, meV, hbar = 1
LATTICE_CONST = 0.246 # atomic spacing
GAMMA0 = 3160 # intralayer hopping (~ fermi velocity)
GAMMA1 = 380 # interlayer hopping
GAMMA3 = 380 # trigonal warping
GAMMA4 = 140 # e-h asymmetry

class BernalBLG4BandModel(Model):
    def __init__(self, il_potential=0, lattice_const=LATTICE_CONST, 
                 gamma0=GAMMA0, gamma1=GAMMA1, gamma3=GAMMA3, gamma4=GAMMA4,
                 dtype=np.complex128):
        self.bands = 4
        self.dtype = dtype

        u = il_potential
        a = np.sqrt(3) / 2 * lattice_const
        v = a * gamma0
        v3 = a * gamma3
        v4 = a * gamma4
        t = gamma1
        self.fixed_ = np.array([
            [-u, 0, 0, 0],
            [0, -u, t, 0],
            [0, t, u, 0],
            [0, 0, 0, u]
        ], dtype=dtype)
        self.x_mul_ = np.array([
            [0, v, -v4, v3],
            [v, 0, 0, -v4],
            [-v4, 0, 0, v],
            [v3, -v4, v, 0]
        ], dtype=dtype)
        self.y_mul_ = 1j * np.array([
            [0, -v, v4, v3],
            [v, 0, 0, v4],
            [-v4, 0, 0, -v],
            [-v3, -v4, v, 0]
        ], dtype=dtype)

    def hamiltonian(self, x, y):
        return np.tensordot(x, self.x_mul_, axes=0) \
            + np.tensordot(y, self.y_mul_, axes=0) \
            + self.fixed_
    
class BernalBLG2BandModel(Model):
    def __init__(self, il_potential=0, lattice_const=LATTICE_CONST, 
                 gamma0=GAMMA0, gamma1=GAMMA1, gamma3=GAMMA3, gamma4=GAMMA4,
                 dtype=np.complex128):
        self.bands = 2
        self.dtype = dtype

        u = il_potential
        t = gamma1
        a = np.sqrt(3) / 2 * lattice_const
        v = a * gamma0
        v3 = a * gamma3
        v4 = a * gamma4
        im0 = v*v / t + v3 * lattice_const / 4 / np.sqrt(3)
        im4 = 2 * v / t * (u * v / t + v4)
        self.fixed_ = np.array([
            [-u, 0],
            [0, u]
        ], dtype=dtype)
        self.x_mul_ = np.array([
            [0, v3],
            [v3, 0]
        ], dtype=dtype)
        self.y_mul_ = 1j *  np.array([
            [0, v3],
            [-v3, 0]
        ], dtype=dtype)
        self.x2_mul_ = np.array([
            [im4, -im0],
            [-im0, -im4]
        ], dtype=dtype)
        self.y2_mul_ = np.array([
            [im4, im0],
            [im0, -im4]
        ])
        self.xy_mul_ = 2j * np.array([
            [0, im0],
            [-im0, 0]
        ], dtype=dtype)

    def hamiltonian(self, x, y):
        return np.tensordot(x, self.x_mul_, axes=0) \
            + np.tensordot(y, self.y_mul_, axes=0) \
            + np.tensordot(x*x, self.x2_mul_, axes=0) \
            + np.tensordot(y*y, self.y2_mul_, axes=0) \
            + np.tensordot(x*y, self.xy_mul_, axes=0) \
            + self.fixed_