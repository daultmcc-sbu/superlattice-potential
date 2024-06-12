import numpy as np
from matplotlib import pyplot as plt

from continuum import *
from superlattice import *
from lattice import *
from plotting import *

####################
#### PARAMETERS ####
####################

# graphene
VELOCITY = 6.582
IL_HOPPING = 0.4

# superlattice
SCALE = 2.*np.pi/500.
ALPHA = 0.3

# simulation quality
RADIUS = 4
GRID_SIZE = 15

# plotting
N, M = 4, 4
SL_MIN, SL_MAX = 0.017, 0.025
DISP_MIN, DISP_MAX = -0.010, 0.001



###################
#### UTILITIES ####
###################

def model(sl_pot, disp_pot):
    continuum = GatedBernalBGModel(VELOCITY, IL_HOPPING, disp_pot)
    lattice = TriangularLattice(SCALE)
    sl_potential = np.diag([sl_pot, sl_pot, ALPHA * sl_pot, ALPHA * sl_pot])
    return SuperlatticeModel(continuum, sl_potential, lattice, RADIUS)

band = model(0.005, -0.010).lowest_pos_band()

def inspect(sl_pot, disp_pot):
    m = model(sl_pot, disp_pot)
    fig, ax = plt.subplots()
    plot_bandstructure(m, 50, ax)
    d = m.solve(GRID_SIZE)
    chern = d.chern_number(band)
    fig.suptitle(f"SL: {sl_pot}, Disp: {disp_pot}, Chern: {chern}")

def plot_2d(rows, cols, xv, yv, data, names):
    fig, axs = plt.subplots(rows, cols)
    for ax, z, name in zip(axs.flat, data, names):
        mesh = ax.pcolormesh(xv, yv, z)
        fig.colorbar(mesh, ax=ax)
        ax.set_xlabel("$V_{SL}$")
        ax.set_ylabel("$V_0$")
        ax.set_title(name)




############
### MAIN ###
############

if __name__ == '__main__':
    sl = np.linspace(SL_MIN, SL_MAX, N)
    disp = np.linspace(DISP_MIN, DISP_MAX, M)
    slv, dispv = np.meshgrid(sl, disp, indexing='ij')
    above = np.zeros((N,M))
    below = np.zeros((N,M))
    width = np.zeros((N,M))
    chern = np.zeros((N,M))
    # qm_eig = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            d = model(sl[i], disp[j]).solve(GRID_SIZE)
            # band = d.lowest_pos_band()
            above[i,j] = d.gap_above(band)
            below[i,j] = d.gap_above(band - 1)
            width[i,j] = d.bandwidth(band)
            chern[i,j] = d.chern_number(band)
            # qm_eig[i,j] = np.min(np.abs(np.linalg.eigvalsh(d.quantum_metric(band))))

    min_gap = np.minimum(above, below)

    plot_2d(2, 3, slv, dispv, 
            [above, below, min_gap, width, chern], 
            ["Gap above", "Gap below", "Min gap", "Band width", "Chern number"])

    plt.show()