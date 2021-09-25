import numpy as np
from typing import List, Union, Tuple, Sequence, Optional
from numbers import Number

from numpy.lib.arraysetops import isin

__all__ = [
    'cat', 'chunk', 'gather', 'index_select', 'masked_select', 'movedim', 'swapdims', 'narrow', 
    'nonzero', 'scatter', 'scatter_add', 'scatter_', 'scatter_add_', 'split', 'squeeze', 'stack',
    'unsqueeze', 'unsqueeze_', 'unbind', 'meshgrid', 'clone', 'is_contiguous', 'contiguous', 'repeat', 'repeat_interleave', 
    'permute', 'view', 'view_as', 'expand', 'expand_as',
    't', 'seed', 'manual_seed', 'rand', 'rand_like', 'randint', 'randint_like', 'randn', 'randn_like', 'randperm', 'uniform_', 'normal_', 'zero_',
    'clamp', 'max', 'min', 'flatten',
]

def cat(tensors: Sequence[np.ndarray], 
        dim: int=0, 
        out: Optional[np.ndarray]=None):
    return np.concatenate(tensors, axis=dim, out=out)

def chunk(input: np.ndarray, 
          chunks: int, 
          dim: int=0):
    # if not divisible, shrink the last chunk
    s = input.shape[dim] 
    if s % chunks != 0:
        x = s // chunks + 1
        indices = []
        while x < s: 
            indices.append(x)
            x += x
        return np.split(input, indices, axis=dim)
    else:
        return np.split(input, chunks, axis=dim)

def gather(input: np.ndarray,
           dim: int,
           index: np.ndarray,
           out: Optional[np.ndarray] = None):

    # handle the special case of input.shape[other_dim] > index.shape[other_dim]
    # np.take_along_axis tries to broadcast
    # torch.gather just use input[:index.shape[other_dim]]
    index_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    input_xsection_shape = input.shape[:dim] + input.shape[dim + 1:]
    if index_xsection_shape > input_xsection_shape: # tuple performs element-wise cmp
        raise IndexError(f"index shape should be not greater than input shape except at dim: got {index_xsection_shape} > {input_xsection_shape}")
    new_input_shape= list(index_xsection_shape)
    new_input_shape.insert(dim, input.shape[dim])
    slices = tuple([np.s_[:s] for s in new_input_shape])
    res = np.take_along_axis(input[slices], index, axis=dim)
    if out is not None:
        np.copyto(out, res)
    return res


# equals:
#      dim-1, dim, dim+1
# input[..., index, ...]
def index_select(input: np.ndarray,
                 dim: int,
                 index: np.ndarray,
                 out: Optional[np.ndarray] = None):
    return np.take(input, index, axis=dim, out=out)

def masked_select(input: np.ndarray,
                  mask: np.ndarray,
                  out: Optional[np.ndarray] = None):
    # this function will first try to broadcast mask to the same shape as input.
    if mask.shape != input.shape:
        mask = np.broadcast_to(mask, input.shape)
    return input[mask]

def movedim(input: np.ndarray,
            source: Union[int, Tuple[int]],
            destination: Union[int, Tuple[int]]):
    return np.moveaxis(input, source, destination)

def swapdims(input: np.ndarray,
             dim0: int,
             dim1: int):
    return np.swapaxes(input, dim0, dim1)

def narrow(input: np.ndarray,
           dim: int,
           start: int,
           length: int):
    return input.take(range(start, start+length), axis=dim)

def nonzero(input: np.ndarray,
            as_tuple: bool = False):
    res = np.nonzero(input)
    if as_tuple:
        return res
    else:
        return np.asarray(res).T

# a helper function to implement different reduce methods beyond put_along_axis
def _scatter_along_axis(arr, indices, values, axis, reduce=None):
    if axis < 0:
        axis += arr.ndim
    # a tuple
    indices = np.lib.shape_base._make_along_axis_idx(arr.shape, indices, axis)
    if reduce is None:
        arr[indices] = values
    elif reduce == 'add':
        np.add.at(arr, indices, values) # safe for duplicated indices!
    elif reduce == 'multiply':
        np.multiply.at(arr, indices, values)
    else:
        raise NotImplementedError

def scatter_(input: np.ndarray,
            dim: int,
            index: np.ndarray,
            src: np.ndarray,
            reduce: Optional[str] = None):
    if index.shape > src.shape: # tuple performs element-wise cmp
        raise IndexError(f"index shape should be not greater than source shape: got {index.shape} > {src.shape}")
    index_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    input_xsection_shape = input.shape[:dim] + input.shape[dim + 1:]
    if index_xsection_shape > input_xsection_shape: # tuple performs element-wise cmp
        raise IndexError(f"index shape should be not greater than input shape except at dim: got {index_xsection_shape} > {input_xsection_shape}")
    new_input_shape= list(index_xsection_shape)
    new_input_shape.insert(dim, input.shape[dim])
    new_src_shape= list(index_xsection_shape)
    new_src_shape.insert(dim, src.shape[dim])
    input_slices = tuple([np.s_[:s] for s in new_input_shape])
    src_slices = tuple([np.s_[:s] for s in new_src_shape])
    _scatter_along_axis(input[input_slices], index, src[src_slices], axis=dim, reduce=reduce)
    return input

def scatter(input: np.ndarray,
            dim: int,
            index: np.ndarray,
            src: np.ndarray,
            reduce: Optional[str] = None):
    output = input.copy()
    return scatter_(output, dim, index, src, reduce)

def scatter_add(input: np.ndarray,
                dim: int,
                index: np.ndarray,
                src: np.ndarray):
    return scatter(input, dim, index, src, reduce='add')
    

def scatter_add_(input: np.ndarray,
                dim: int,
                index: np.ndarray,
                src: np.ndarray):
    return scatter_(input, dim, index, src, reduce='add')

def split(input: np.ndarray,
          split_size_or_sections: Union[int, Sequence[int]],
          dim: int = 0):
    return np.split(input, split_size_or_sections, axis=dim)

def squeeze(input: np.ndarray,
            dim: Optional[int] = None):
    res = np.squeeze(input, axis=dim)
    return res

def stack(input: Sequence[np.ndarray],
          dim: int = 0,
          out: Optional[np.ndarray] = None):
    return np.stack(input, axis=dim, out=out)

def unsqueeze(input: np.ndarray,
              dim: int):
    return np.expand_dims(input, axis=dim)

def unsqueeze_(input: np.ndarray,
              dim: int):
    # implement an in-place unsqueeze (modify on the same view)
    new_shape = list(input.shape)
    if dim < 0:
        dim += input.ndim + 1 # note: insert(-1, x) will insert before the last element
    new_shape.insert(dim, 1)
    input.shape = tuple(new_shape)
    return input

def squeeze_(input: np.ndarray,
             dim: Optional[int] = None):
    # in-place shape modification
    new_shape = list(input.shape)
    if dim is None:
        new_shape = [x for x in new_shape if x != 1]
        if new_shape == []:
            new_shape = [1]
    else:
        if new_shape[dim] != 1:
            raise ValueError('cannot select an axis to squeeze out which has size not equal to one')
        new_shape.pop(dim)
    input.shape = tuple(new_shape)
    return input

def unbind(input: np.ndarray,
           dim: int = 0):
    if dim == 0:
        return list(input)           
    else:
        return list(input.swapaxes(0, dim))

def meshgrid(*xi: np.ndarray):
    return np.meshgrid(*xi, indexing='ij')

def clone(input: np.ndarray):
    return input.copy()

def is_contiguous(input: np.ndarray):
    return input.data.contiguous

def contiguous(input: np.ndarray):
    if is_contiguous(input):
        return input
    else:
        return np.ascontiguousarray(input)

# WARN: there is no torch.repeat(), only Tensor.repeat().
def repeat(input: np.ndarray,
                 *sizes: int):
    return np.tile(input, sizes)

def repeat_interleave(input: np.ndarray,
                            repeats: Union[int, np.ndarray],
                            dim: Optional[int] = None):
    return np.repeat(input, repeats, axis=dim)


# ref: https://stackoverflow.com/questions/19826262/transpose-array-and-actually-reorder-memory
# permute never copy.
def permute(input: np.ndarray,
            *axes: int):
    return np.transpose(input, axes=axes)

# ref: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
# view never copy.
def view(input: np.ndarray,
         *sizes: int):
    output = input.view() # must not override the original view.
    output.shape = sizes
    return output

def view_as(input: np.ndarray,
            other: np.ndarray):
    return view(input, other.shape)

# ref: https://stackoverflow.com/questions/65234748/what-is-the-numpy-equivalent-of-expand-in-pytorch
# ref: https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969/9
# expand should not copy the array, while repeat() copy.
# WARN: numpy broadcast is read-only, but torch expand still can be modified.
def expand(input: np.ndarray,
           *sizes: int):
    # check for -1 and replace with original size
    sizes = list(sizes)
    for i in range(len(sizes)):
        if sizes[i] == -1:
            sizes[i] = input.shape[i]
    return np.broadcast_to(input, sizes)

def expand_as(input: np.ndarray,
              other: np.ndarray):
    return expand(input, *other.shape)

# ref: https://pytorch.org/docs/stable/generated/torch.t.html#torch.t
# note: arr.T will transpose all dimensions, but t() only transpose 0 and 1.
def t(input: np.ndarray):
    if input.ndim < 2: 
        return input
    else:
        return input.swapaxes(0, 1)

def seed():
    np.random.seed()

def manual_seed(seed: int):
    np.random.seed(seed)

def rand(*size: int, 
         out: Optional[np.ndarray] = None):
    res = np.random.rand(*size)
    if out is not None:
        np.copyto(out, res)
    return res

def rand_like(input: np.ndarray):
    return rand(*input.shape)

def randint(low: int,
            high: Optional[int] = None,
            size: Optional[Sequence[int]] = None,
            out: Optional[np.ndarray] = None):
    res = np.random.randint(low=low, high=high, size=size)
    if out is not None:
        np.copyto(out, res)
    return res

def randint_like(input: np.ndarray,
                 low: int,
                 high: Optional[int] = None):
    return randint(low=low, high=high, size=input.shape)

def randn(*size: int, 
         out: Optional[np.ndarray] = None):
    res = np.random.randn(*size)
    if out is not None:
        np.copyto(out, res)
    return res

def randn_like(input: np.ndarray):
    return randn(*input.shape)

def randperm(n: int,
             out: Optional[np.ndarray] = None):
    res = np.random.permutation(n)
    if out is not None:
        np.copyto(out, res)
    return res

# TODO: this is silly, how to make it really in-place?
def uniform_(input: np.ndarray,
             fro: int = 0,
             to: int = 1):
    np.copyto(input, rand_like(input) * (to - fro) + fro)
    return input

def normal_(input: np.ndarray,
            mean: int = 0,
            std: int = 1):
    np.copyto(input, randn_like(input) * std + mean)
    return input

def zero_(input: np.ndarray):
    input.fill(0)
    return input

def clamp(input: np.ndarray,
          min: Union[Number, np.ndarray, None] = None,
          max: Union[Number, np.ndarray, None] = None):
    return np.clip(input, min, max)

def clamp_(input: np.ndarray,
          min: Union[Number, np.ndarray, None] = None,
          max: Union[Number, np.ndarray, None] = None):
    return np.clip(input, min, max, out=input)

# should I mimic the full behaviour? it's silly...
def max(input: np.ndarray,
        dim: Union[int, np.ndarray, None] = None,
        keepdim: bool = False,
        out: Optional[np.ndarray] = None):
    if isinstance(dim, np.ndarray):
        return np.maximum(input, dim, out=out)
    indices = np.expand_dims(np.argmax(input, axis=dim), axis=dim)
    values = np.take_along_axis(input, indices, axis=dim)
    if not keepdim:
        indices = indices.squeeze(dim)
        values = values.squeeze(dim)
    if out is not None:
        np.copyto(out, values)
    return (values, indices)

def min(input: np.ndarray,
        dim: Union[int, np.ndarray, None] = None,
        keepdim: bool = False,
        out: Optional[np.ndarray] = None):
    if isinstance(dim, np.ndarray):
        return np.minimum(input, dim, out=out)
    indices = np.expand_dims(np.argmin(input, axis=dim), axis=dim)
    values = np.take_along_axis(input, indices, axis=dim)
    if not keepdim:
        indices = indices.squeeze(dim)
        values = values.squeeze(dim)
    if out is not None:
        np.copyto(out, values)
    return (values, indices)    

def flatten(input: np.ndarray,
            start_dim: int = 0,
            end_dim: int = -1):
    if start_dim < 0:
        start_dim += input.ndim
    if end_dim < 0:
        end_dim += input.ndim
    new_shape = input.shape[:start_dim] + [np.prod(input.shape[start_dim:end_dim])] + input.shape[end_dim:]
    return np.reshape(new_shape)

""" TODO
def topk()
def to()
"""