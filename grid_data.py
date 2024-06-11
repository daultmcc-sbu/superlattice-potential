import numpy as np

class GridData:
    def __init__(self, trans, points, eigvals, eigvecs, divisions):
        self.trans = trans
        self.points = points
        self.divisions = divisions
        self.eigvals = eigvals
        self.eigvecs = eigvecs

    def chern_number(self, band):
        return np.sum(self.berry_curvature(band)) * np.linalg.det(self.trans) / 2 / np.pi
    
    def berry_curvature(self, band):
        eigvecs = self.eigvecs[..., band]

        arr = np.zeros((self.divisions, self.divisions))
        for i in range(0, self.divisions):
            for j in range(0, self.divisions):
                link1 = np.vdot(eigvecs[i,j], eigvecs[i+1,j])
                link2 = np.vdot(eigvecs[i+1,j], eigvecs[i+1,j+1])
                link3 = np.vdot(eigvecs[i+1,j+1], eigvecs[i,j+1])
                link4 = np.vdot(eigvecs[i,j+1], eigvecs[i,j])
                arr[i,j] = np.imag(np.log(link1 * link2 * link3 * link4))

        return mod_pi(arr) / np.linalg.det(self.trans)
    
    def quantum_metric(self, band):
        eigvecs = self.eigvecs[..., band]

        a = self.trans[:,0]
        b = self.trans[:,1]
        solver = np.linalg.inv(np.array([
            [a[0]**2, a[1]**2, 2 * a[0] * a[1]],
            [b[0]**2, b[1]**2, 2 * b[0] * b[1]],
            [(a+b)[0]**2, (a+b)[1]**2, 2 * (a+b)[0] * (a+b)[1]]
        ]))

        arr = np.zeros((self.divisions, self.divisions, 2, 2))
        for i in range(0, self.divisions):
            for j in range(0, self.divisions):
                link1 = np.vdot(eigvecs[i,j], eigvecs[i+1,j])
                link2 = np.vdot(eigvecs[i,j], eigvecs[i,j+1])
                link12 = np.vdot(eigvecs[i,j], eigvecs[i+1,j+1])
                [g11, g22, g12] = solver @ (2 - 2 * np.abs(np.array([link1, link2, link12])))
                arr[i,j,...] = np.array([[g11, g12], [g12, g22]])

        return arr
    
    def quantum_geometric_tensor(self, band):
        g = self.quantum_metric(band)
        omega = np.tensordot(self.berry_curvature(band), np.array([[0, 0.5j],[-0.5j,0]]), axes=0)
        return g + omega
    
    def lowest_pos_band(self):
        return np.argmax(self.eigvals[0,0] > 0)
    
    def bandwidth(self, band):
        return np.max(self.eigvals[...,band]) - np.min(self.eigvals[...,band])
    
    def gap_above(self, band):
        return np.min(self.eigvals[...,band+1]) - np.max(self.eigvals[...,band])
                
    
def extend_grid(grid):
    extended_0 = np.concatenate((grid, grid[np.newaxis,0]), axis=0)
    extended_01 = np.concatenate((extended_0, extended_0[:,0,np.newaxis]), axis=1)
    return extended_01

def mod_pi(x):
    y = x / 2 / np.pi
    return 2 * np.pi * (y - np.round(y))