__author__ = 'ashawkey'
__version__ = '0.1.0'

# inherit everything from np
from numpy import *

# patch torch operations
from . import ops

NEVER_OVERRIDE = ['view', 'size', 'repeat']

PREFIX_TORCH = 'torch_'
PREFIX_NUMPY = 'numpy_'

patched_ops = []
overridden_ops = []

def patch_ops(mode='compatible'):
    for name in ops.__all__:
        # do not patch in-place operators
        if name.endswith('_'):
            continue
        globals()[PREFIX_TORCH + name] = getattr(ops, name)
        patched_ops.append(PREFIX_TORCH + name)
        if not name in globals():
            globals()[name] = getattr(ops, name)
            patched_ops.append(name)
        else:
            if mode == 'override' and name not in NEVER_OVERRIDE:
                globals()[PREFIX_NUMPY + name] = globals()[name]
                globals()[name] = getattr(ops, name)
                overridden_ops.append(name)
                
def unpatch_ops():
    for name in patched_ops:
        del globals()[name]
    patched_ops.clear()
    for name in overridden_ops:
        globals()[name] = globals()[PREFIX_NUMPY + name]
        del globals()[PREFIX_NUMPY + name]
    overridden_ops.clear()


# patch ndarray
from . import arr
import forbiddenfruit

patched_arr = []
overridden_arr = []

def patch_arr(mode='compatible'):
    for name, op in arr.patches.items():
        forbiddenfruit.curse(ndarray, PREFIX_TORCH + name, op)
        patched_arr.append(PREFIX_TORCH + name)
        if not name in dir(ndarray):
            forbiddenfruit.curse(ndarray, name, op)
            patched_arr.append(name)
        else:
            if mode == 'override' and name not in NEVER_OVERRIDE:
                forbiddenfruit.curse(ndarray, PREFIX_NUMPY + name, getattr(ndarray, name))
                forbiddenfruit.curse(ndarray, name, op)
                overridden_arr.append(name)

def unpatch_arr():
    for name in patched_arr:
        forbiddenfruit.reverse(ndarray, name)
    patched_arr.clear()
    for name in overridden_arr:
        forbiddenfruit.curse(ndarray, name, getattr(ndarray, PREFIX_NUMPY + name))
        forbiddenfruit.reverse(ndarray, PREFIX_NUMPY + name)
    overridden_arr.clear()


# set mode
import warnings

def set_patch_mode(mode='compatible'):
    assert mode in ['compatible', 'override', 'none']
    unpatch_ops()
    unpatch_arr()
    if mode == 'none':
        return
    # if mode == 'override':
    #     warnings.warn('override mode may break up existing numpy code and lead to unexpected crashes!', UserWarning)
    patch_ops(mode)
    patch_arr(mode)

import inspect
def list_patches():
    print(f'============== np. ==============')
    for name in patched_ops:
        print(f'**{name}** `{str(inspect.signature(eval(name)))}`')
    for name in overridden_ops:
        print(f'**{name}** [overridden] `{str(inspect.signature(eval(name)))}`')
    print(f'========== np.ndarray. ==========')
    for name in patched_arr:
        op = getattr(ndarray, name)
        print(f'**{name}** `{str(inspect.signature(op)) if callable(op) else "attribute"}`')
    for name in overridden_arr:
        op = getattr(ndarray, name)
        print(f'**{name}** [overridden] `{str(inspect.signature(op)) if callable(op) else "attribute"}`')
    print(f'==================================')

set_patch_mode()