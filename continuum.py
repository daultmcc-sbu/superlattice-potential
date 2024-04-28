import numpy as np
from model import *

class GatedBernalBGModel(Model):
    def __init__(self, velocity, il_hopping, il_potential):
        self.velocity = velocity
        self.il_hopping = il_hopping
        self.il_potential = il_potential

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
        ], dtype=complex)