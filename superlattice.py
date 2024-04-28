import numpy as np

from continuum import *
from lattice import *
from band_data import *
from model import *

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
    
    def solve(self, divisions):
        # marks = np.arange(0, divisions + 1) / divisions
        marks = np.linspace(0, 1, divisions + 1)
        a1s = np.outer(marks, self.lattice.a1)
        a1s_expanded = np.repeat(a1s[:,np.newaxis], divisions + 1, axis=1)
        a2s = np.outer(marks, self.lattice.a2)
        a2s_expanded = np.repeat(a2s[np.newaxis,:], divisions + 1, axis=0)
        points = a1s_expanded + a2s_expanded

        hamiltonians = np.array([[self.hamiltonian(k) for k in row] for row in points])
        eigvals, eigvecs = np.linalg.eigh(hamiltonians)

        return BandData(self.lattice.trans / divisions, points, eigvals, eigvecs, divisions)

# gated Bernal BG model with triangular lattice potential
class BasicModel(SuperlatticeModel):
    def __init__(self, potential, alpha, beta, scale, radius):
        VELOCITY = 6.582
        IL_HOPPING = 0.4
        continuum = GatedBernalBGModel(VELOCITY, IL_HOPPING, beta * potential)
        lattice = TriangularLattice(scale)
        sl_potential = np.diag([potential, potential, alpha * potential, alpha * potential])
        super().__init__(continuum, sl_potential, lattice, radius)