import numpy as np

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

def mod_pi(x):
    y = x / 2 / np.pi
    return 2 * np.pi * (y - np.round(y))