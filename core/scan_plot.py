import numpy as np
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize
import matplotlib.pyplot as plt

from .utilities import Register, tr_form_from_eigvec, complex_to_rgb, auto_subplots, remove_outliers, tr_form_from_ratio
from .banddata import BandData


##############
#### MAIN ####
##############

def make_plot_scan_2d(av, alabel, bv, blabel, observs):
    fig, axs = auto_subplots(plt, len(observs))
    for plot, ax in zip(observs, axs.flat):
        mesh = ax.pcolormesh(av, bv, plot.plotting_data(), norm=plot.norm(), **plot.colormesh_opts)
        if plot.colorbar:
            cb = fig.colorbar(mesh, ax=ax)
        if plot.title is not None:
            ax.set_title(plot.title)
        ax.set_xlabel(alabel)
        ax.set_ylabel(blabel)

    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.90)

    return fig




#######################
#### SUBPLOT SETUP ####
#######################

