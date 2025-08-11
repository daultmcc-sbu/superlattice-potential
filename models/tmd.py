import numpy as np

from core import Model

# arxiv:1506.08860
# UNITS: nm, meV
WSE_LATTICE_CONST = 0.33
WSE_F0 = 1550
WSE_F1 = 1190
WSE_F7 = -61.2

class TMDBilayerModel(Model):
    def __init__(self, il_potential=0, lattice_const=WSE_LATTICE_CONST,
                 f0 = WSE_F0, f1 = WSE_F1, f7 = WSE_F7,
                 dtype=np.complex128):
        self.bands = 4
        self.dtype = dtype

        u = il_potential
        self.fixed_ = np.array([
            [-u + f0, 0, 0, 0],
            [0, -u, 0, f7],
            [0, 0, u + f0, 0],
            [0, f7, 0, u]
        ], dtype=dtype)
        self.x_mul_ = f1 * lattice_const * np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=dtype)
        self.y_mul_ = 1j * f1 * lattice_const * np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0]
        ], dtype=dtype)

    def hamiltonian(self, x, y):
        return np.tensordot(x, self.x_mul_, axes=0) \
            + np.tensordot(y, self.y_mul_, axes=0) \
            + self.fixed_
