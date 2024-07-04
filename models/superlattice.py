import numpy as np

from core import Model

class SuperlatticeModel(Model):
    def __init__(self, continuum, sl_potential, lattice, radius):
        self.continuum = continuum
        self.lattice = lattice
        self.indices = lattice.indices(radius)
        self.points = [lattice.point_at(ind) for ind in self.indices]
        self.bands = continuum.bands * len(self.points)
        self.dtype = continuum.dtype
        self.sl_potential = sl_potential.astype(self.dtype)
        if sl_potential.shape != (continuum.bands, continuum.bands):
            raise ValueError("sl_potential has incorrect shape")
        
        self.sl_potential_hamiltonian = np.block([[
            self.sl_potential if lattice.is_adjacent(ind1,ind2) else np.zeros((continuum.bands, continuum.bands), dtype=self.dtype)
                for ind1 in self.indices] for ind2 in self.indices])

    def hamiltonian(self, k):
        n = self.continuum.bands
        diag = np.zeros((n * len(self.indices), n * len(self.indices)), dtype=self.dtype)
        for i, point in enumerate(self.points):
            diag[n*i:n*i+n, n*i:n*i+n] = self.continuum.hamiltonian(k + point)

        return self.sl_potential_hamiltonian + diag