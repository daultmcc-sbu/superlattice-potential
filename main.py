import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors

from continuum import *
from superlattice import *
from lattice import *
from plotting import *

#############################
#### MODEL SPECIFICATION ####
#############################

# fixed parameters
VELOCITY = 6.582
IL_HOPPING = 0.4
ALPHA = 0.3

def model(sl_pot, disp_pot, scale, radius):
        continuum = GatedBernalBGModel(VELOCITY, IL_HOPPING, disp_pot)
        lattice = TriangularLattice(scale)
        sl_potential = np.diag([sl_pot, sl_pot, ALPHA * sl_pot, ALPHA * sl_pot])
        return SuperlatticeModel(continuum, sl_potential, lattice, radius)



################
### ROUTINES ###
################

def sweep_pots_grid(
          sl_min, sl_max, sl_n, disp_min, disp_max, disp_n, 
          scale, band_offset, radius, grid_size):
    sl = np.linspace(sl_min, sl_max, sl_n)
    disp = np.linspace(disp_min, disp_max, disp_n)
    slv, dispv = np.meshgrid(sl, disp, indexing='ij')
    above = np.zeros((sl_n, disp_n))
    below = np.zeros((sl_n, disp_n))
    width = np.zeros((sl_n, disp_n))
    chern = np.zeros((sl_n, disp_n))

    # awful hack
    band = model(0.005, -0.010, 2.*np.pi/500, radius).lowest_pos_band() + band_offset

    for i in range(sl_n):
        for j in range(disp_n):
            d = model(sl[i], disp[j], scale, radius).solve(grid_size)
            above[i,j] = d.gap_above(band)
            below[i,j] = d.gap_above(band - 1)
            width[i,j] = d.bandwidth(band)
            chern[i,j] = d.chern_number(band)

    above = np.maximum(0, above)
    below = np.maximum(0, below)
    min_gap = np.minimum(above, below)

    fig, axs = plt.subplots(2, 3)
    plots_2d(slv, dispv, 
            [above, below, min_gap, width], 
            fig, axs.flat,
            ["Gap above", "Gap below", "Min gap", "Band width"],
            "$V_{SL}$", "$V_0$")
    
    norm = colors.BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256, extend='both')
    mesh = axs[1,1].pcolormesh(slv, dispv, chern, norm=norm, cmap='RdBu_r')
    fig.colorbar(mesh, ax=axs[1,1])
    axs[1,1].set_title("Chern number")
    axs[1,1].set_xlabel("$V_{SL}$")
    axs[1,1].set_ylabel("$V_0$")
    
    plt.show()



def bands_and_bz(
          sl_pot, disp_pot, scale, band_offset, zoom, 
          radius, k_n, band_n):
    m = model(disp_pot, sl_pot, scale, radius)
    band = model(0.005, -0.010, 2.*np.pi/500, radius).lowest_pos_band() + band_offset

    fig, axs = plt.subplots(2,2)
    plot_bandstructure(m, band_n, axs[0,0], highlight=band)

    size = np.max(np.abs(m.lattice.trans @ np.array([1,1])))
    x = np.linspace(-zoom * size, zoom * size, k_n)
    y = np.linspace(-zoom * size, zoom * size, k_n)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    berry = np.zeros((k_n, k_n))
    qm_det = np.zeros((k_n, k_n))
    qgt_eig = np.zeros((k_n, k_n))

    for i in range(k_n):
        for j in range(k_n):
            qgt = m.qgt(np.array([x[i], y[j]]), band)
            berry[i,j] = 2 * np.imag(qgt[0,1])
            qm_det[i,j] = np.linalg.det(np.real(qgt))
            qgt_eig[i,j] = np.linalg.eigvalsh(qgt)[0]

    plots_2d(xv, yv, 
            [berry, qm_det, qgt_eig],
            fig, axs.flat[1:],
            ["Berry curvature", "FSM det", "QGT min eigval"],
            "$k_x$", "$k_y$", blur=True)
    
    plt.show()