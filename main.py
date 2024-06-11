import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

import cProfile

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
SL_POT = 0.005
DISP_POT = -0.010
ALPHA = 0.3

# simulation quality
RADIUS = 3

# plotting
POINTS = 100
DIVISIONS = 10



###############
#### MODEL ####
###############

continuum = GatedBernalBGModel(VELOCITY, IL_HOPPING, DISP_POT)
lattice = TriangularLattice(SCALE)
sl_potential = np.diag([SL_POT, SL_POT, ALPHA * SL_POT, ALPHA * SL_POT])
model = SuperlatticeModel(continuum, sl_potential, lattice, RADIUS)



##################
#### PLOTTING ####
##################

fig, axs = plt.subplots(2, 2)

plot_bandstructure(model, POINTS, axs[0,0])

data = model.solve(DIVISIONS)
top_band = data.lowest_pos_band()
print(data.chern_number(top_band - 1), data.chern_number(top_band))
plot_gd_berry_curvature(data, top_band, fig, axs[1,0])
plot_gd_quantum_metric_det(data, top_band, fig, axs[1,1])
plot_gd(data, np.linalg.eigvalsh(data.quantum_geometric_tensor(top_band))[...,0], fig, axs[0,1])
# plot_bz(data, np.linalg.eigvalsh(data.quantum_metric(top_band))[...,0], fig, axs[0,1])

plt.show()