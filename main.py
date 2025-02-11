import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors

import argparse
import pickle

from core import *
from models.graphene import BernalBLG2BandModel, BernalBLG4BandModel, LATTICE_CONST, GAMMA0, GAMMA1, GAMMA3, GAMMA4
from models.superlattice import SuperlatticeModel


###############
#### MODEL ####
###############

ALPHA = 0.3

def make_model(sl_pot, disp_pot, scale, radius, alpha, square, two_band=False, gamma0=GAMMA0, gamma1=GAMMA1, gamma3=GAMMA3, gamma4=GAMMA4):
        if square:
            lattice = SquareLattice(1)
            scale = 2 * np.pi / scale 
        else:
            lattice = TriangularLattice(1)
            scale = 4 * np.pi / np.sqrt(3) / scale

        # alpha = np.sqrt(alpha)
        if not two_band:
            continuum = BernalBLG4BandModel(disp_pot, LATTICE_CONST * scale, gamma0, gamma1, gamma3, gamma4)
            # sl_potential = np.diag(np.array([alpha * sl_pot, alpha * sl_pot, sl_pot / alpha, sl_pot / alpha], dtype=np.complex128)) 
            sl_potential = np.diag(np.array([alpha * sl_pot, alpha * sl_pot, sl_pot, sl_pot], dtype=np.complex128))
        else:    
            continuum = BernalBLG2BandModel(disp_pot, LATTICE_CONST * scale, gamma0, gamma1, gamma3, gamma4)
            # sl_potential = np.diag(np.array([alpha * sl_pot, sl_pot / alpha], dtype=np.complex128))
            sl_potential = np.diag(np.array([alpha * sl_pot, sl_pot], dtype=np.complex128))

        return SuperlatticeModel(continuum, sl_potential, lattice, radius)




###################
#### UTILITIES ####
###################

# def band_from_offset(args):
#     return make_model(0.005, -0.010, args.scale, args.radius, args.alpha, args.square, args.four_band).lowest_pos_band() + args.band_offset

def band_from_offset(args):
    return int(make_model(0.005, -0.010, args.scale, args.radius, args.alpha, args.square, args.two_band).hamiltonian(0, 0).shape[0] / 2) + args.band_offset

def strlist(s):
    return s.split(',')




##############
#### MAIN ####
##############

def bz_sc(args):
    model = make_model(args.sl_pot, args.disp_pot, args.scale, args.radius, args.alpha, args.square,
                       args.two_band, args.gamma0, args.gamma1, args.gamma3, args.gamma4)
    band = band_from_offset(args)
    subplots = [single_subplots[id] for id in args.subplots]
    bd = single_bz(model, band, args.zoom, args.bz_res)
    ts, hs_ts, Es = single_bands(model, args.struct_res)
    fig = make_plot_single(bd, ts, hs_ts, Es, subplots)
    fig.suptitle(f"$V_{{SL}} = {args.sl_pot}$, $V_0 = {args.disp_pot}, C = {round(bd.chern)}$")
    for observable in args.observables:
        observable = int_observables[observable]
        print(f"{observable.title}: {observable.compute(bd)}")
    data = {'bd': bd, 'ts': ts, 'hs_ts': hs_ts, 'Es': Es}
    return fig, data

def scan_sc(args):
    def modelf(sl_pot, disp_pot):
        return make_model(sl_pot, disp_pot, args.scale, args.radius, args.alpha, args.square,
                          args.two_band, args.gamma0, args.gamma1, args.gamma3, args.gamma4)
    
    band = band_from_offset(args)
    observables = [int_observables[id] for id in args.observables]
    [av, bv], observs = scan(modelf, band, 
                            [(args.sl_min, args.sl_max, args.sl_n), 
                             (args.disp_min, args.disp_max, args.disp_n)], 
                            observables, spacing = 1 / args.bz_quality)
    fig = make_plot_scan_2d(av, "$V_{SL}$", bv, "$V_0$", observs)
    if args.title is not None:
        fig.suptitle(args.title)
    data = {'av': av, 'bv': bv, 'observs': observs}
    return fig, data

def scang_sc(args):
    gs = {'gamma0': args.gamma0, 'gamma1': args.gamma1, 'gamma3': args.gamma3, 'gamma4': args.gamma4}
    aid = 'gamma' + args.gammas[0]
    bid = 'gamma' + args.gammas[1]
    def modelf(a, b):
        gs[aid] = a
        gs[bid] = b
        return make_model(args.sl_pot, args.disp_pot, args.scale, args.radius, args.alpha, args.square, args.two_band, **gs)
    
    band = band_from_offset(args)
    observables = [int_observables[id] for id in args.observables]
    [av, bv], observs = scan(modelf, band,
                            [(args.a_min, args.a_max, args.a_n),
                             (args.b_min, args.b_max, args.b_n)],
                            observables, spacing = 1 / args.bz_quality)
    fig = make_plot_scan_2d(av, f"$\\gamma_{args.gammas[0]}$", bv, f"$\\gamma_{args.gammas[1]}$", observs)
    data = {'av': av, 'bv': bv, 'observs': observs}
    return fig, data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--plot-file', default=None)
    parser.add_argument('-D', '--data-file', default=None)
    parser.add_argument('-2', '--two-band', action='store_true')
    parser.add_argument('-r', '--radius', type=int, default=3)
    parser.add_argument('-s', '--scale', type=float, default=30.0)
    parser.add_argument('-a', '--alpha', type=float, default=ALPHA)
    parser.add_argument('-sq', '--square', action='store_true')
    parser.add_argument('-g0', '--gamma0', type=float, default=GAMMA0)
    parser.add_argument('-g1', '--gamma1', type=float, default=GAMMA1)
    parser.add_argument('-g3', '--gamma3', type=float, default=GAMMA3)
    parser.add_argument('-g4', '--gamma4', type=float, default=GAMMA4)
    subparsers = parser.add_subparsers(required=True)

    parser_single = subparsers.add_parser("single")
    parser_single.add_argument('sl_pot', type=float)
    parser_single.add_argument('disp_pot', type=float)
    parser_single.add_argument('-z', '--zoom', type=float, default=0.6)
    parser_single.add_argument('-bn', '--bz-res', type=int, default=50)
    parser_single.add_argument('-sn', '--struct-res', type=int, default=100)
    parser_single.add_argument('-b', '--band-offset', type=int, default=0)
    parser_single.add_argument('-p', '--subplots', type=strlist, default="berry,trvioliso,trviolbycstruct")
    parser_single.add_argument('-o', '--observables', type=strlist, default="chern,gap,width,trvioliso,berryfluc,berryflucn1")
    parser_single.set_defaults(func=bz_sc)

    parser_scan = subparsers.add_parser("scan")
    parser_scan.add_argument('sl_min', type=float)
    parser_scan.add_argument('sl_max', type=float)
    parser_scan.add_argument('sl_n', type=int)
    parser_scan.add_argument('disp_min', type=float)
    parser_scan.add_argument('disp_max', type=float)
    parser_scan.add_argument('disp_n', type=int)
    parser_scan.add_argument('-bq', '--bz-quality', type=int, default=10)
    parser_scan.add_argument('-b', '--band-offset', type=int, default=0)
    parser_scan.add_argument('-o', '--observables', type=strlist, default="width,gap,chern,berryfluc,trvioliso")
    parser_scan.add_argument('-t', '--title')
    parser_scan.set_defaults(func=scan_sc)

    parser_scang = subparsers.add_parser("scang")
    parser_scang.add_argument('sl_pot', type=float)
    parser_scang.add_argument('disp_pot', type=float)
    parser_scang.add_argument('gammas', type=strlist)
    parser_scang.add_argument('a_min', type=float)
    parser_scang.add_argument('a_max', type=float)
    parser_scang.add_argument('a_n', type=int)
    parser_scang.add_argument('b_min', type=float)
    parser_scang.add_argument('b_max', type=float)
    parser_scang.add_argument('b_n', type=int)
    parser_scang.add_argument('-bq', '--bz-quality', type=int, default=10)
    parser_scang.add_argument('-o', '--observables', type=strlist, default="width,gap,chern,trvioliso")
    parser_scang.add_argument('-b', '--band-offset', type=int, default=0)
    parser_scang.set_defaults(func=scang_sc)

    args = parser.parse_args()
    fig, data = args.func(args)

    if args.data_file is not None:
        with open(args.data_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    if args.plot_file is None:
        plt.show()
    else:
        fig.savefig(args.plot_file)