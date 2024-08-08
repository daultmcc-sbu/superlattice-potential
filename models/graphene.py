import numpy as np

from core import Model

# physical values
# UNITS: nm, eV, s
# LATTICE_SPACING = 0.246
VELOCITY = 0.6582 # hbar * v_F
IL_HOPPING = 0.4 # gamma1
TRIGONAL = 0 # 0.081 # √3 * a * gamma3 (=0.38) / 2

class BernalBLG4BandModel(Model):
    def __init__(self, velocity=VELOCITY, il_hopping=IL_HOPPING, il_potential=0, trigonal=TRIGONAL, dtype=np.complex128):
        self.bands = 4
        self.dtype = dtype

        u = il_potential
        t = il_hopping
        v = velocity
        g = trigonal
        self.fixed_ = np.array([
            [-u, 0, 0, 0],
            [0, -u, t, 0],
            [0, t, u, 0],
            [0, 0, 0, u]
        ], dtype=dtype)
        self.x_mul_ = np.array([
            [0, v, 0, g],
            [v, 0, 0, 0],
            [0, 0, 0, v],
            [g, 0, v, 0]
        ], dtype=dtype)
        self.y_mul_ = 1j * np.array([
            [0, -v, 0, g],
            [v, 0, 0, 0],
            [0, 0, 0, -v],
            [-g, 0, v, 0]
        ], dtype=dtype)

    def hamiltonian(self, x, y):
        return np.tensordot(x, self.x_mul_, axes=0) \
            + np.tensordot(y, self.y_mul_, axes=0) \
            + self.fixed_
    
class BernalBLG2BandModel(Model):
    def __init__(self, velocity=VELOCITY, il_hopping=IL_HOPPING, il_potential=0, trigonal=TRIGONAL, dtype=np.complex128):
        self.bands = 2
        self.dtype = dtype

        u = il_potential
        t = il_hopping
        v = velocity
        im = v*v/t # hbar^2 / 2 m_eff
        g = 3 * trigonal
        self.fixed_ = u * np.array([
            [-1, 0],
            [0, 1]
        ], dtype=dtype)
        self.x_mul_ = g * np.array([
            [0, 1],
            [1, 0]
        ], dtype=dtype)
        self.y_mul_ = 1j * g * np.array([
            [0, 1],
            [-1, 0]
        ], dtype=dtype)
        self.x2_mul_ = im * np.array([
            [2*u/t, -1],
            [-1, -2*u/t]
        ], dtype=dtype)
        self.y2_mul_ = im * np.array([
            [2*u/t, 1],
            [1, -2*u/t]
        ])
        self.xy_mul_ = 2j * im * np.array([
            [0, 1],
            [-1, 0]
        ], dtype=dtype)

    def hamiltonian(self, x, y):
        return np.tensordot(x, self.x_mul_, axes=0) \
            + np.tensordot(y, self.y_mul_, axes=0) \
            + np.tensordot(x*x, self.x2_mul_, axes=0) \
            + np.tensordot(y*y, self.y2_mul_, axes=0) \
            + np.tensordot(x*y, self.xy_mul_, axes=0) \
            + self.fixed_