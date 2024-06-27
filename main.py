import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors

import argparse

from core import *
from models.graphene import BernalBLGModel
from models.superlattice import SuperlatticeModel


#############################
#### MODEL SPECIFICATION ####
#############################

# fixed parameters
VELOCITY = 6.582
IL_HOPPING = 0.4
ALPHA = 0.3

def make_model(sl_pot, disp_pot, scale, radius):
        continuum = BernalBLGModel(VELOCITY, IL_HOPPING, disp_pot)
        lattice = TriangularLattice(2 * np.pi / scale)
        sl_potential = np.diag([sl_pot, sl_pot, ALPHA * sl_pot, ALPHA * sl_pot])
        return SuperlatticeModel(continuum, sl_potential, lattice, radius)




###################
#### UTILITIES ####
###################

def band_from_offset(radius, offset):
    return make_model(0.005, -0.010, 500, radius).lowest_pos_band() + offset




##############
#### MAIN ####
##############

def bz_sc(args):
    model = make_model(args.sl_pot, args.disp_pot, args.scale, args.radius)
    band = band_from_offset(args.radius, args.band_offset)
    fig = make_plot_band_geometry(model, band, args.zoom, args.bz_res, args.struct_res)
    fig.suptitle(f"$V_{{SL}} = {args.sl_pot}$, $V_0 = {args.disp_pot}$")
    return fig

def scan_sc(args):
    def modelf(sl_pot, disp_pot):
        return make_model(sl_pot, disp_pot, args.scale, args.radius)
    band = band_from_offset(args.radius, args.band_offset)
    fig = make_plot_sweep_parameters_2d_grid(modelf, band, 
                        args.sl_min, args.sl_max, args.sl_n, "$V_{SL}$",
                        args.disp_min, args.disp_max, args.disp_n, "$V_0$",
                        grid_size=args.bz_res)
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-file', default=None)
    subparsers = parser.add_subparsers(required=True)

    parser_bz = subparsers.add_parser("bz")
    parser_bz.add_argument('sl_pot', type=float)
    parser_bz.add_argument('disp_pot', type=float)
    parser_bz.add_argument('-z', '--zoom', default=0.6)
    parser_bz.add_argument('-r', '--radius', default=3)
    parser_bz.add_argument('-s', '--scale', default=500.0)
    parser_bz.add_argument('-bn', '--bz-res', default=25)
    parser_bz.add_argument('-kn', '--struct-res', default=50)
    parser_bz.add_argument('-b', '--band-offset', default=0)
    parser_bz.set_defaults(func=bz_sc)

    parser_scan = subparsers.add_parser("scan")
    parser_scan.add_argument('sl_min', type=float)
    parser_scan.add_argument('sl_max', type=float)
    parser_scan.add_argument('sl_n', type=int)
    parser_scan.add_argument('disp_min', type=float)
    parser_scan.add_argument('disp_max', type=float)
    parser_scan.add_argument('disp_n', type=int)
    parser_scan.add_argument('-r', '--radius', default=3)
    parser_scan.add_argument('-s', '--scale', default=500.0)
    parser_scan.add_argument('-bn', '--bz-res', default=4)
    parser_scan.add_argument('-b', '--band-offset', default=0)
    parser_scan.set_defaults(func=scan_sc)

    args = parser.parse_args()
    fig = args.func(args)

    if args.output_file is None:
        plt.show()
    else:
        fig.savefig(args.output_file)