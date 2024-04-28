import numpy as np

class Model:
    def spectrum(self, k):
        return np.linalg.eigvalsh(self.hamiltonian(k))
    
    def eig(self, k):
        return np.linalg.eigh(self.hamiltonian(k))
    
    def berry_curvature(self, k, band, trans=np.array([[1e-7,0],[0,1e-7]])):
        state1 = self.eig(k)[1][:,band]
        state2 = self.eig(k + trans @ np.array([1,0]))[1][:,band]
        state3 = self.eig(k + trans @ np.array([1,1]))[1][:,band]
        state4 = self.eig(k + trans @ np.array([0,1]))[1][:,band]

        link1 = np.vdot(state1, state2)
        link2 = np.vdot(state2, state3)
        link3 = np.vdot(state3, state4)
        link4 = np.vdot(state4, state1)

        phase = np.imag(np.log(link1 * link2 * link3 * link4))
        return mod_pi(phase) / np.linalg.det(trans)
    
    def quantum_metric(self, k, band, trans=np.array([[1e-7,0],[0,1e-7]])):
        a = trans[:,0]
        b = trans[:,1]
        solver = np.linalg.inv(np.array([
            [a[0]**2, a[1]**2, 2 * a[0] * a[1]],
            [b[0]**2, b[1]**2, 2 * b[0] * b[1]],
            [(a+b)[0]**2, (a+b)[1]**2, 2 * (a+b)[0] * (a+b)[1]]
        ]))

        state00 = self.eig(k)[1][:,band]
        state10 = self.eig(k + trans @ np.array([1,0]))[1][:,band]
        state11 = self.eig(k + trans @ np.array([1,1]))[1][:,band]
        state01 = self.eig(k + trans @ np.array([0,1]))[1][:,band]

        link1 = np.vdot(state00, state10)
        link2 = np.vdot(state00, state01)
        link12 = np.vdot(state00, state11)

        [g11, g22, g12] = solver @ (2 - 2 * np.abs(np.array([link1, link2, link12])))
        return np.array([[g11, g12], [g12, g22]])
    
    def quantum_geometric_tensor(self, k, band, trans=np.array([[1e-7,0],[0,1e-7]])):
        g = self.quantum_metric(k, band, trans)
        omega = self.berry_curvature(k, band, trans) * np.array([[0, 0.5j],[-0.5j,0]])
        return g + omega

def mod_pi(x):
    y = x / 2 / np.pi
    return 2 * np.pi * (y - np.round(y))