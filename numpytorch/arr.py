import numpy as np
from . import ops

patches = {
    'dim': np.ndarray.ndim,
    'numel': lambda self: self.size,
    'index_select': ops.index_select,
    'squeeze_': ops.squeeze_,
    'unsqueeze': ops.unsqueeze,
    'unsqueeze_': ops.unsqueeze_,
    'is_contiguous': ops.is_contiguous,
    'contiguous': ops.contiguous,
    'clone': ops.clone,
    'repeat': ops.repeat,
    'repeat_interleave': ops.repeat_interleave,
    'view': ops.view,
    'permute': ops.permute,
    'expand': ops.expand,
    'expand_as': ops.expand_as,
    'uniform_': ops.uniform_, 
    'normal_': ops.normal_,
    'zero_': ops.zero_,
    'clamp': ops.clamp,
    'clamp_': ops.clamp_,
    'clip_': ops.clamp_,
    'flatten': ops.flatten,
    'max': ops.max,
    'min': ops.min,
    'amax': np.amax,
    'amin': np.amin,
    # type casting (will copy)
    'half': lambda self: self.astype(np.float16),
    'float': lambda self: self.astype(np.float32),
    'double': lambda self: self.astype(np.float64),
    'short': lambda self: self.astype(np.int16),
    'int': lambda self: self.astype(np.int32),
    'long': lambda self: self.astype(np.int64),
    'bool': lambda self: self.astype(bool),
}

# chain all the ufuncs 
ufuncs = [
    # math
    'add', 'subtract', 'multiply', 'matmul', 'divide', 'true_divide', 'floor_divide', 'negative', 'positive', 'power', 'float_power', 
    'remainder', 'mod', 'fmod', 'divmod', 'absolute', 'abs', 'fabs', 'sign', 'rint', 'conj', 'exp', 'exp2', 'log', 'log2', 'log10',
    'sqrt', 'square', 'cbrt', 'reciprocal', 'gcd', 'lcm', 'expm1', 'log1p',
    # trig
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
    'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
    'degrees', 'radians', 'deg2rad', 'rad2deg',
    # bitwise
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert', 'left_shift', 'right_shift',
    # cmp
    'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal',
    # logical
    'logical_and', 'logical_or', 'logical_not', 'logical_xor',
    # else
    'maximum', 'minimum', 'fmax', 'fmin',
    'nextafter', 'spacing', 'modf', 'fmod', 'ldexp', 'frexp',
    'isfinite', 'isinf', 'isnan', 'isnat', 'fabs', 'signbit', 'copysign', 'floor', 'ceil', 'trunc',
]

ufunc_aliases = {
    'atan': 'arctan',
    'asin': 'arcsin',
    'acos': 'arccos',
    'atan2': 'arctan2',
    'ge': 'greater_equal',
    'gt': 'greater',
    'le': 'less_equal',
    'lt': 'less',
}

# a little tricky to bind ufuncs 
# forbiddenfruit only supports binding of pure functions, so we need to wrap the callable functors into explicit functions.
def construct_ufunc(fname, inplace=False):
    ufunc = getattr(np, fname)
    if ufunc.nin == 1:
        def inner(self):
            return ufunc(self, out=self if inplace else None)
    elif ufunc.nin == 2:
        def inner(self, other):
            return ufunc(self, other, out=self if inplace else None)
    else:
        raise NotImplementedError('ufunc with more than 3 inputs is not supported now.')
    return inner

for fname in ufuncs:
    patches[fname] = construct_ufunc(fname)
    patches[fname + '_'] = construct_ufunc(fname, inplace=True)

for fname, rname in ufunc_aliases.items():
    patches[fname] = construct_ufunc(rname)
    patches[fname + '_'] = construct_ufunc(rname, inplace=True)