import numpy as np

from core import Model

# physical values
VELOCITY = 6.582
IL_HOPPING = 0.4

class BernalBLGModel(Model):
    def __init__(self, velocity=VELOCITY, il_hopping=IL_HOPPING, il_potential=0, dtype=np.complex128):
        self.velocity = velocity
        self.il_hopping = il_hopping
        self.il_potential = il_potential
        self.bands = 4
        self.dtype = dtype

    def hamiltonian(self, k):
        [kx, ky] = k
        v = self.velocity
        u0 = self.il_potential
        t = self.il_hopping
        return np.array([
            [u0, v * (kx - 1j * ky), 0, t],
            [v * (kx + 1j * ky), u0, 0, 0],
            [0, 0, -u0, v * (kx - 1j * ky)],
            [t, 0, v * (kx + 1j * ky), -u0]
        ], dtype=self.dtype)