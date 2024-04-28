import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

import cProfile

from continuum import *
from superlattice import *
from lattice import *
from plotting import *

SCALE = 2.*np.pi/500.

POTENTIAL = 0.005
ALPHA = 0.3
BETA = 0.5

RADIUS = 5

POINTS = 100
DIVISIONS = 30

model = BasicModel(POTENTIAL, ALPHA, BETA, SCALE, RADIUS)

fig, axs = plt.subplots(2, 2)

plot_bandstructure(model, POINTS, axs[0,0])

data = model.solve(DIVISIONS)
top_band = data.lowest_pos_band()
print(data.chern_number(top_band - 1), data.chern_number(top_band))
plot_berry_curvature(data, top_band, fig, axs[1,0])
plot_quantum_metric_det(data, top_band, fig, axs[1,1])
plot_bz(data, np.linalg.eigvalsh(data.quantum_geometric_tensor(top_band))[...,0], fig, axs[0,1])
# plot_bz(data, np.linalg.eigvalsh(data.quantum_metric(top_band))[...,0], fig, axs[0,1])

plt.show()