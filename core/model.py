import numpy as np
from .utilities import mod_pi
from .grid_data import GridData

class Model:
    def spectrum(self, k):
        return np.linalg.eigvalsh(self.hamiltonian(k))
    
    def eig(self, k):
        return np.linalg.eigh(self.hamiltonian(k))
    
    def qgt(self, k, band, trans=np.array([[1e-7,0],[0,1e-7]])):
        a = trans[:,0]
        b = trans[:,1]
        solver = np.linalg.inv(np.array([
            [a[0]**2, a[1]**2, 2 * a[0] * a[1]],
            [b[0]**2, b[1]**2, 2 * b[0] * b[1]],
            [(a+b)[0]**2, (a+b)[1]**2, 2 * (a+b)[0] * (a+b)[1]]
        ]))

        states = [ [ self.eig(k + trans @ np.array([i,j]))[1][:,band] 
                    for j in range(2) ] for i in range(2) ]
        
        link = lambda i,j,k,l: np.vdot(states[i][j], states[k][l])

        phase = np.imag(np.log(link(0,0,1,0) * link(1,0,1,1) * link(1,1,0,1) * link(0,1,0,0)))
        berry_curv = mod_pi(phase) / np.linalg.det(trans)

        [g11, g22, g12] = solver @ (2 - 2 * np.abs(np.array([link(0,0,1,0), link(0,0,0,1), link(0,0,1,1)])))

        return np.array([
            [g11, g12 + 0.5j * berry_curv],
            [g12 - 0.5j * berry_curv, g22]
            ])
    
    def lowest_pos_band(self):
        return np.argmax(self.spectrum([0,0]) > 0)
    
    def gaps_at(self, k, band):
        spec = self.spectrum(k)
        return (spec[band + 1] - spec[band], spec[band] - spec[band-1])
    
    def gaps(self, ks, band):
        above = np.inf
        below = np.inf
        for k in ks:
            a, b = self.gaps_at(k, band)
            above = min(above, a)
            below = min(below, b)
        return (above, below)
    
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

        return GridData(self.lattice.trans / divisions, points, eigvals, eigvecs, divisions)