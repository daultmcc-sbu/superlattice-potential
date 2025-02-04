import numpy as np
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize

from .utilities import Register, complex_to_rgb, remove_outliers, tr_form_from_eigvec, tr_form_from_ratio



###############
#### SETUP ####
###############

int_observables = {}

class RegisterIntObservables(Register):
    registry = int_observables

    def id_from_name(name):
        return name.removesuffix('IntObservable').lower()
    
class IntObservable(metaclass=RegisterIntObservables, register=False):
    colormesh_opts = {}
    colorbar = True
    title = None

    def __init__(self, shape):
        self.data = np.zeros(shape)

    def update(self, i, bd):
        self.data[i] = self.compute(bd)

    @staticmethod
    def compute(bd):
        pass

    def plotting_data(self):
        return self.data

    def norm(self):
        return Normalize()
    
class ComplexIntObservable(IntObservable, register=False):
    colorbar = False

    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=complex)

    def plotting_data(self):
        return complex_to_rgb(self.data)





#####################
#### OBSERVABLES ####
#####################

class GapIntObservable(IntObservable):
    title = "Band gap"

    @staticmethod
    def compute(bd):
        return bd.gap
    
class WidthIntObservable(IntObservable):
    title = "Band width"

    @staticmethod
    def compute(bd):
        return bd.width
    
class ChernIntObservable(IntObservable):
    title = "Chern number"
    colormesh_opts = {'cmap': 'RdBu_r'}

    @staticmethod
    def compute(bd):
        return bd.chern
    
    def norm(self):
        return BoundaryNorm(boundaries=[-2.5,-1.5,-0.5,0.5,1.5,2.5], ncolors=256, extend='both')

class BerryFlucIntObservable(IntObservable):
    title = "Berry fluc"
    colormesh_opts = {'cmap': 'plasma'}

    @staticmethod
    def compute(bd):
        return bd.berry_fluc
    
    def norm(self):
        body = remove_outliers(self.data)
        return LogNorm(vmin=body.min(), vmax=body.max())

class BerryFlucN1IntObservable(IntObservable):
    title = "Berry fluc (n1)"
    colormesh_opts = {'cmap': 'plasma'}

    @staticmethod
    def compute(bd):
        return bd.berry_fluc_n1

    def norm(self):
        body = remove_outliers(self.data)
        return LogNorm(vmin=body.min(), vmax=body.max())

class TrViolIsoIntObservable(IntObservable):
    title = "Trace cond viol (isometric)"
    colormesh_opts = {'cmap': 'plasma'}

    @staticmethod
    def compute(bd):
        return bd.int(bd.tr_viol_iso)
    
    def norm(self):
        return Normalize(0, 20)

class TrViolMinIntObservable(IntObservable):
    title = "Trace cond viol (min)"
    colormesh_opts = {'cmap': 'plasma'}

    @staticmethod
    def compute(bd):
        form = tr_form_from_eigvec(bd.qgt_bzmin_eigvec)
        return bd.int(bd.tr_viol(form))
    
    def norm(self):
        return Normalize(0, 20)

class TrViolAvgIntObservable(IntObservable):
    title = "Trace cond viol (avg)"
    colormesh_opts = {'cmap': 'plasma'}

    @staticmethod
    def compute(bd):
        form = tr_form_from_eigvec(bd.avg_qgt_eigvec)
        return bd.int(bd.tr_viol(form))
    
    def norm(self):
        return Normalize(0, 20)
    
class TrViolOptIntObservable(IntObservable):
    title = "Trace cond viol (opt)"
    colormesh_opts = {'cmap': 'plasma'}

    @staticmethod
    def compute(bd):
        form = tr_form_from_ratio(*bd.optimal_cstruct)
        return bd.int(bd.tr_viol(form))
    
    def norm(self):
        return Normalize(0, 20)
    
class CstructMinIntObservable(ComplexIntObservable):
    title = "Complex struct (min)"

    @staticmethod
    def compute(bd):
        w = bd.qgt_bzmin_eigvec
        return w[1] / w[0]
    
class CstructAvgIntObservable(ComplexIntObservable):
    title = "Complex struct (avg)"
    
    @staticmethod
    def compute(bd):
        w = bd.avg_qgt_eigvec
        return w[1] / w[0]
    
class CstructOptIntObservable(ComplexIntObservable):
    title = "Complex struct (opt)"
    
    @staticmethod
    def compute(bd):
        z = bd.optimal_cstruct
        return z[0] + 1j * z[1]