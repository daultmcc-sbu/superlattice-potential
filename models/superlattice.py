import numpy as np

from core import Model

class SuperlatticeModel(Model):
    def __init__(self, continuum, sl_potential, lattice, radius):
        self.continuum = continuum
        self.sl_potential = sl_potential
        self.n_bands = sl_potential.shape[0]
        self.lattice = lattice
        self.indices = lattice.indices(radius)
        self.points = [lattice.point_at(ind) for ind in self.indices]
        
        self.sl_potential_hamiltonian = np.block([[
            self.sl_potential if lattice.is_adjacent(ind1,ind2) else np.zeros((self.n_bands, self.n_bands))
                for ind1 in self.indices] for ind2 in self.indices])

    def hamiltonian(self, k):
        n = self.n_bands
        diag = np.zeros((n * len(self.indices), n * len(self.indices)), dtype=complex)
        for i, point in enumerate(self.points):
            diag[n*i:n*i+n, n*i:n*i+n] = self.continuum.hamiltonian(k + point)

        return self.sl_potential_hamiltonian + diag