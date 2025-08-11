import numpy as np

from core import Model

class SuperlatticeModel(Model):
    def __init__(self, continuum, sl_potential, lattice, radius):
        self.continuum = continuum
        self.lattice = lattice
        self.indices = lattice.indices_in_radius(radius)
        self.points = np.array([lattice.point_at(ind) for ind in self.indices])
        self.bands = continuum.bands * len(self.indices)
        self.dtype = continuum.dtype
        self.sl_potential = sl_potential.astype(self.dtype)
        if sl_potential.shape != (continuum.bands, continuum.bands):
            raise ValueError("sl_potential has incorrect shape")

        zeros = np.zeros((continuum.bands, continuum.bands), dtype=self.dtype)
        self.sl_potential_hamiltonian = np.block([[
            self.sl_potential if lattice.is_adjacent(ind1,ind2) else zeros
                for ind1 in self.indices] for ind2 in self.indices])

    def hamiltonian(self, x, y):
        x = np.array(x)
        y = np.array(y)
        xe = x[...,np.newaxis] + self.points[:,0]
        ye = y[...,np.newaxis] + self.points[:,1]
        Hs = self.continuum.hamiltonian(xe, ye)
        n = self.continuum.bands
        diag = np.zeros(x.shape + (self.bands, self.bands), dtype=self.dtype)
        for i in range(len(self.indices)):
            diag[...,n*i:n*i+n,n*i:n*i+n] = Hs[...,i,:,:]

        return diag + self.sl_potential_hamiltonian
