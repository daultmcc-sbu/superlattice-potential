import numpy as np

from core import Model

# physical values
VELOCITY = 6.582
IL_HOPPING = 0.4
TRIGONAL = 0 # 0.3 * 1.42

class BernalBLG4BandModel(Model):
    def __init__(self, velocity=VELOCITY, il_hopping=IL_HOPPING, il_potential=0, trigonal=TRIGONAL, dtype=np.complex128):
        self.velocity = velocity
        self.il_hopping = il_hopping
        self.il_potential = il_potential
        self.trigonal = trigonal
        self.bands = 4
        self.dtype = dtype
        self.lattice = None

        u = il_potential
        t = il_hopping
        v = velocity
        g = 3 * trigonal
        self.fixed_ = np.array([
            [u, 0, 0, t],
            [0, u, 0, 0],
            [0, 0, -u, 0],
            [t, 0, 0, -u]
        ], dtype=dtype)
        self.x_mul_ = np.array([
            [0, v, 0, 0],
            [v, 0, g, 0],
            [0, g, 0, v],
            [0, 0, v, 0]
        ], dtype=dtype)
        self.y_mul_ = 1j * np.array([
            [0, -v, 0, 0],
            [v, 0, -g, 0],
            [0, g, 0, -v],
            [0, 0, v, 0]
        ], dtype=dtype)

    def hamiltonian(self, x, y):
        return np.tensordot(x, self.x_mul_, axes=0) \
            + np.tensordot(y, self.y_mul_, axes=0) \
            + self.fixed_
    
class BernalBLG2BandModel(Model):
    def __init__(self, velocity=VELOCITY, il_hopping=IL_HOPPING, il_potential=0, trigonal=TRIGONAL, dtype=np.complex128):
        self.velocity = velocity
        self.il_hopping = il_hopping
        self.il_potential = il_potential
        self.trigonal = trigonal
        self.bands = 2
        self.dtype = dtype
        self.lattice = None

        u = il_potential
        t = il_hopping
        v = velocity
        g = 3 * trigonal
        self.fixed_ = np.array([
            [u, 0],
            [0, -u]
        ], dtype=dtype)
        self.x_mul_ = np.array([
            [0, g],
            [g, 0]
        ], dtype=dtype)
        self.y_mul_ = np.array([
            [0, -g],
            [g, 0]
        ], dtype=dtype)
        self.x2my2_mul_ = np.array([
            [0, v*v/t],
            [v*v/t, 0]
        ], dtype=dtype)
        self.xy_mul_ = np.array([
            [0, 2j*v*v/t],
            [-2j*v*v/t, 0]
        ], dtype=dtype)

    def hamiltonian(self, x, y):
        return np.tensordot(x, self.x_mul_, axes=0) \
            + np.tensordot(y, self.y_mul_, axes=0) \
            + np.tensordot(x*x - y*y, self.x2my2_mul_, axes=0) \
            + np.tensordot(x*y, self.xy_mul_, axes=0) \
            + self.fixed_