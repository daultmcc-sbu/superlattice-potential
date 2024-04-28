import numpy as np

class Lattice:
    def __init__(self, a1, a2, adjacents, hs_points=None):
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.trans = np.array([a1,a2]).T
        self.adjacents = np.array(adjacents)
        self.hs_points = hs_points

    def point_at(self, ind):
        # return ind[0] * self.a1 + ind[1] * self.a2
        return self.trans @ ind
    
    def is_adjacent(self, ind1, ind2):
        diff = ind2 - ind1
        return np.any(np.all(diff == self.adjacents, axis=1))
    
class TriangularLattice(Lattice):
    def __init__(self, a):
        a1 = [a, 0]
        a2 = [a/2, np.sqrt(3)/2 * a]
        adjacents = [[-1,1],[0,1],[1,0],[1,-1],[0,-1],[-1,0]]
        hs_points = {
            r"$\Gamma$": np.array([0,0]), 
            "K": np.array([a / 2., a / (2 * np.sqrt(3))]),
            "M": np.array([a / 2., 0.])
        }
        super().__init__(a1, a2, adjacents, hs_points)

    @staticmethod
    def indices(radius):
        partial_lattice = [np.array([j-k, k]) for j in range(radius+1) for k in range(j)]
        rot = np.array([[0, -1], [1, 1]])
        full_lattice = [np.array([0,0])]
        for t in range(6):
            for pt in partial_lattice:
                full_lattice.append(np.linalg.matrix_power(rot, t) @ pt)
        return full_lattice
    
class SquareLattice(Lattice):
    def __init__(self, a):
        adjacents = [[1,0],[0,1],[-1,0],[0,-1]]
        hs_points = {
            r"$\Gamma$": np.array([0,0]),
            "X": np.array([a,0]),
            "M": np.array([a,a])
        }
        super().__init__([a,0], [0,a], adjacents, hs_points)

    @staticmethod
    def indices(radius):
        partial_lattice = [np.array([j-k, k]) for j in range(radius+1) for k in range(j)]
        rot = np.array([[0, -1], [1, 0]])
        full_lattice = [np.array([0, 0])]
        for t in range(4):
            for pt in partial_lattice:
                full_lattice.append(np.linalg.matrix_power(rot, t) @ pt)
        return full_lattice
    
# class SquareLattice(RectangularLattice):
#     def __init__(self, a):
#         super().__init__(a, a)