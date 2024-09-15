import numpy as np

class Model:
    def spectrum(self, x, y):
        return np.linalg.eigvalsh(self.hamiltonian(x, y))
    
    def eig(self, x, y):
        return np.linalg.eigh(self.hamiltonian(x, y))
    
    def qgt(self, x, y, band, trans=np.array([[1e-5,0],[0,1e-5]])):
        a = trans[:,0]
        b = trans[:,1]
        solver = np.linalg.inv(np.array([
            [a[0]**2, a[1]**2, 2 * a[0] * a[1]],
            [b[0]**2, b[1]**2, 2 * b[0] * b[1]],
            [(a+b)[0]**2, (a+b)[1]**2, 2 * (a+b)[0] * (a+b)[1]]
        ]))

        diffs = trans @ np.array([[[i,j] for j in range(2)] for i in range(2)])
        xf = x + diffs[...,0]
        yf = y + diffs[...,1]
        states = self.eig(xf, yf)
        
        link = lambda i,j,k,l: np.vdot(states[i,j], states[k,l])

        phase = np.angle(link(0,0,1,0) * link(1,0,1,1) * link(1,1,0,1) * link(0,1,0,0))
        berry_curv = phase / np.linalg.det(trans)

        [g11, g22, g12] = solver @ (2 - 2 * np.abs(np.array([link(0,0,1,0), link(0,0,0,1), link(0,0,1,1)])))

        return np.array([
            [g11, g12 + 0.5j * berry_curv],
            [g12 - 0.5j * berry_curv, g22]
            ], dtype=self.dtype)

    def lowest_pos_band(self):
        return np.argmax(self.spectrum(0,0) > 0)

    def gaps(self, x, y, band):
        spec = self.spectrum(x, y)
        return (spec[...,band + 1] - spec[...,band], spec[...,band] - spec[...,band-1])