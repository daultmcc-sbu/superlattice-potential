import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors
import scipy.optimize as opt

import argparse
import pickle
import copy

from core import *
from models.graphene import BernalBLG2BandModel, BernalBLG4BandModel, LATTICE_CONST, GAMMA0, GAMMA1, GAMMA3, GAMMA4
from models.superlattice import SuperlatticeModel


###############
#### MODEL ####
###############

ALPHA = 0.3

def make_model(sl_pot, disp_pot, length=30, radius=3, alpha=0.3, square=False, two_band=False, gamma0=GAMMA0, gamma1=GAMMA1, gamma3=GAMMA3, gamma4=GAMMA4, **kwargs):
        if square:
            lattice = SquareLattice(1)
            scale = 2 * np.pi / length
        else:
            lattice = TriangularLattice(1)
            scale = 4 * np.pi / np.sqrt(3) / length

        if not two_band:
            continuum = BernalBLG4BandModel(disp_pot, LATTICE_CONST * scale, gamma0, gamma1, gamma3, gamma4)
            sl_potential = np.diag(np.array([alpha * sl_pot, alpha * sl_pot, sl_pot, sl_pot], dtype=np.complex128))
        else:    
            continuum = BernalBLG2BandModel(disp_pot, LATTICE_CONST * scale, gamma0, gamma1, gamma3, gamma4)
            sl_potential = np.diag(np.array([alpha * sl_pot, sl_pot], dtype=np.complex128))

        return SuperlatticeModel(continuum, sl_potential, lattice, radius)




###################
#### UTILITIES ####
###################

def band_from_offset(args):
    args.sl_pot = 0
    args.disp_pot = 0
    return int(make_model(**vars(args)).hamiltonian(0, 0).shape[0] / 2) + args.band_offset

def strlist(s):
    return s.split(',')




##############
#### MAIN ####
##############

def single_sc(args):
    model = make_model(**vars(args))
    band = band_from_offset(args)
    subplots = [single_subplots[id] for id in args.subplots]
    bd = single_square(model, band, args.zoom, args.bz_res)
    ts, hs_ts, Es = single_bands(model, args.struct_res)
    fig = make_plot_single(bd, ts, hs_ts, Es, subplots)
    fig.suptitle(f"$V_{{SL}} = {args.sl_pot}$, $V_0 = {args.disp_pot}, C = {round(bd.chern)}$")
    for observable in args.observables:
        observable = int_observables[observable]
        print(f"{observable.title}: {observable.compute(bd)}")
    data = {'bd': bd, 'ts': ts, 'hs_ts': hs_ts, 'Es': Es}
    return fig, data

def scan_sc(args):
    margs = vars(copy.copy(args))
    def modelf(sl_pot, disp_pot):
        margs['sl_pot'] = sl_pot
        margs['disp_pot'] = disp_pot
        return make_model(**margs)
    
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

def scanl_sc(args):
    args.sl_pot = 0
    args.disp_pot = 0
    margs = vars(copy.copy(args))

    if args.optimize:
        sl_pots = []
        disp_pots = []

        def modelf(l):
            if l == 0:
                return make_model(**margs)

            sl_pot0 = args.sl_pot_adj / l
            disp_pot0 = args.disp_pot_adj / l
            margs['length'] = l

            def neg_gap(x):
                margs['sl_pot'] = x[0]
                margs['disp_pot'] = x[1]
                model = make_model(**margs)
                band = band_from_offset(args)
                bd = single_bz(model, band, 1 / args.bz_quality)
                if bd.chern > 1.5 or bd.chern < 0.5:
                    return np.infty
                return -bd.gap
                # return -bd.gap + 0.1 * bd.width + 0.3 * bd.int(bd.tr_viol_iso) + 0.2 * bd.berry_fluc
            
            if args.global_width is None:
                res = opt.minimize(neg_gap, np.array([sl_pot0, disp_pot0]), 
                                method='Nelder-Mead', options={'xatol': args.tolerance, 'fatol': args.tolerance})
            else:
                width = args.global_width / l
                res = opt.shgo(neg_gap, [(sl_pot0 - width, sl_pot0 + width), (disp_pot0 - width, disp_pot0 + width)],
                               options={'f_tol': args.tolerance})



            margs['sl_pot'] = res.x[0]
            sl_pots.append(res.x[0])
            margs['disp_pot'] = res.x[1]
            disp_pots.append(res.x[1])

            return make_model(**margs)
    else:
        def modelf(l):
            margs['sl_pot'] = args.sl_pot_adj / l
            margs['disp_pot'] = args.disp_pot_adj / l
            margs['length'] = l
            return make_model(**margs)
    
    band = band_from_offset(args)
    observables = [int_observables[id] for id in args.observables]
    [a], observs = scan(modelf, band, [(args.l_min, args.l_max, args.l_n)], 
                        observables, spacing = 1 / args.bz_quality)
    fig = make_plot_scan_1d(a, "$L$", observs)
    if args.title is not None:
        fig.suptitle(args.title)
    data = {'l': a, 'observs': observs}

    if args.optimize:
        fig2, axs2 = plt.subplots(1, 2)
        axs2[0].plot(a, sl_pots)
        axs2[1].plot(a, disp_pots)
        data['sl_pots'] = sl_pots
        data['disp_pots'] = disp_pots

    return fig, data

def scang_sc(args):
    aid = 'gamma' + args.gammas[0]
    bid = 'gamma' + args.gammas[1]
    margs = vars(copy.copy(args))
    def modelf(a, b):
        margs[aid] = a
        margs[bid] = b
        return make_model(**margs)
    
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
    parser.add_argument('-l', '--length', type=float, default=30.0)
    parser.add_argument('-a', '--alpha', type=float, default=ALPHA)
    parser.add_argument('-sq', '--square', action='store_true')
    parser.add_argument('-t', '--title')
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
    parser_single.set_defaults(func=single_sc)

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

    parser_scanl = subparsers.add_parser("scanl")
    parser_scanl.add_argument('sl_pot_adj', type=float)
    parser_scanl.add_argument('disp_pot_adj', type=float)
    parser_scanl.add_argument('l_min', type=float)
    parser_scanl.add_argument('l_max', type=float)
    parser_scanl.add_argument('l_n', type=int)
    parser_scanl.add_argument('-bq', '--bz-quality', type=int, default=10)
    parser_scanl.add_argument('-o', '--observables', type=strlist, default="gap,width,trvioliso,berryfluc")
    parser_scanl.add_argument('-O', '--optimize', action='store_true')
    parser_scanl.add_argument('-G', '--global-width', type=float)
    parser_scanl.add_argument('-b', '--band-offset', type=int, default=0)
    parser_scanl.add_argument('-t', '--tolerance', type=float, default=0.1)
    parser_scanl.set_defaults(func=scanl_sc)

    args = parser.parse_args()
    fig, data = args.func(args)

    if args.data_file is not None:
        with open(args.data_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    if args.plot_file is None:
        plt.show()
    else:
        fig.savefig(args.plot_file)