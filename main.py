import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors

import argparse

from core import *
from models.graphene import BernalBLG2BandModel, BernalBLG4BandModel, VELOCITY, TRIGONAL
from models.superlattice import SuperlatticeModel


###############
#### MODEL ####
###############

ALPHA = 0.3

def make_model(sl_pot, disp_pot, scale, radius, four_band=False):
        scale = 2 * np.pi / scale
        velocity = VELOCITY * scale
        trigonal = TRIGONAL * scale

        if four_band:
            continuum = BernalBLG4BandModel(il_potential=disp_pot, velocity=velocity, trigonal=trigonal, dtype=np.complex128)
            sl_potential = np.diag(np.array([ALPHA * sl_pot, ALPHA * sl_pot, sl_pot, sl_pot], dtype=np.complex128)) 
        else:    
            continuum = BernalBLG2BandModel(il_potential=disp_pot, velocity=velocity, trigonal=trigonal, dtype=np.complex128)
            sl_potential = np.diag(np.array([ALPHA * sl_pot, sl_pot], dtype=np.complex128))

        lattice = TriangularLattice(1)
        return SuperlatticeModel(continuum, sl_potential, lattice, radius)




###################
#### UTILITIES ####
###################

def band_from_offset(args):
    return make_model(0.005, -0.010, args.scale, args.radius, args.four_band).lowest_pos_band() + args.band_offset




##############
#### MAIN ####
##############

def bz_sc(args):
    model = make_model(args.sl_pot * 1e-3, args.disp_pot * 1e-3, args.scale, args.radius, args.four_band)
    band = band_from_offset(args)
    fig = make_plot_band_geometry(model, band, args.zoom, args.bz_res, args.struct_res)
    fig.suptitle(f"$V_{{SL}} = {args.sl_pot}$, $V_0 = {args.disp_pot}$")
    return fig

def scan_sc(args):
    def modelf(sl_pot, disp_pot):
        return make_model(sl_pot, disp_pot, args.scale, args.radius, args.four_band)
    band = band_from_offset(args)
    fig = make_plot_sweep_parameters_2d(modelf, band,
                        args.sl_min * 1e-3, args.sl_max * 1e-3, args.sl_n, "$V_{SL}$",
                        args.disp_min * 1e-3, args.disp_max * 1e-3, args.disp_n, "$V_0$",
                        spacing = 1 / args.bz_quality)
                        # spacing = 2*np.pi / args.scale / args.bz_quality)
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-file', default=None)
    parser.add_argument('-4', '--four-band', action='store_true')
    parser.add_argument('-r', '--radius', type=int, default=3)
    parser.add_argument('-s', '--scale', type=float, default=50.0)
    subparsers = parser.add_subparsers(required=True)

    parser_bz = subparsers.add_parser("bz")
    parser_bz.add_argument('sl_pot', type=float)
    parser_bz.add_argument('disp_pot', type=float)
    parser_bz.add_argument('-z', '--zoom', type=float, default=0.6)
    parser_bz.add_argument('-bn', '--bz-res', type=int, default=50)
    parser_bz.add_argument('-sn', '--struct-res', type=int, default=100)
    parser_bz.add_argument('-b', '--band-offset', type=int, default=0)
    parser_bz.set_defaults(func=bz_sc)

    parser_scan = subparsers.add_parser("scan")
    parser_scan.add_argument('sl_min', type=float)
    parser_scan.add_argument('sl_max', type=float)
    parser_scan.add_argument('sl_n', type=int)
    parser_scan.add_argument('disp_min', type=float)
    parser_scan.add_argument('disp_max', type=float)
    parser_scan.add_argument('disp_n', type=int)
    parser_scan.add_argument('-bq', '--bz-quality', type=int, default=10)
    parser_scan.add_argument('-b', '--band-offset', type=int, default=0)
    parser_scan.set_defaults(func=scan_sc)

    args = parser.parse_args()
    fig = args.func(args)

    if args.output_file is None:
        plt.show()
    else:
        fig.savefig(args.output_file)