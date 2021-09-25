# numpytorch



Monkey-patched `numpy` with `pytorch` syntax.

If you are also tired of `dim, axis, keepdim, keepdims, cat, concatenate` , or wasted enough time debugging `repeat(), meshgrid() `, this package provides a dirty solution:

```python
# Just replace the import

#import numpy as np
import numpytorch as np

# use the torch syntax:
x = np.randn(2, 3)
x = x.permute(1, 0).unsqueeze(-1)
x = x.add(1).abs().sin()

# while it won't break the original numpy syntax:
x = np.random.rand(2, 3)
x = np.expand_dims(x.transpose(1, 0), -1)
x = np.sin(np.abs(x + 1))
```



### Features

* fully compatible with pure `numpy` code.

* patched most `pytorch` functions and `Tensor` methods into `numpy` and `ndarray`.



### Install

Only `Cpython` is supported since we use [`forbiddenfruit`](https://github.com/clarete/forbiddenfruit)  to extend the built-in `np.ndarray`.

```bash
pip install numpytorch
```


### Documentations

Since there are conflicted names in `numpy` and `pytorch`, such as `np.stack() & torch.stack()`, `ndarray.view() & Tensor.view()`, two modes are provided to handle these conflicts: `compatible` or `override`.

In the default `compatible` mode, all of the names in `numpy` are kept unchanged:

* If the name is conflicted, we add `torch_{name}` to distinguish from the original `numpy` method. 

  ```python
  np.torch_stack()
  arr.torch_view()
  np.stack() # original numpy stack()
  arr.view() # original numpy.ndarray view()
  ```

* If the name is not conflicted, we add both `torch_{name}` and `{name}`.

  ```python
  np.torch_randn()
  arr.torch_permute()
  np.randn() # alias of torch_randn()
  arr.permute() # alias of torch_permute()
  ```

In the `override` mode, we instead keep the `torch` functions unchanged and rename `numpy` functions. `{name} & torch_{name}` are always added (except some special functions like `view(), size`), and the conflicted `numpy` versions are renamed to `numpy_{name}`. However, this is only experimental and may lead to unexpected bugs since it may break some `numpy` functions. Use at your own risk!

```python
# 'compatible' mode is invoked by default at import
import numpytorch as np

# invoke override mode 
np.set_patch_mode('override')

# invoke compatible mode
np.set_patch_mode('compatible')

# remove all patches
np.set_patch_mode('none')

# list current patches
np.list_patches()
```

All of the patched functions and methods are listed below. Unless specifically mentioned, they should behave similarly as the `torch` counterparts.

#### ============== np.* ==============
**torch_cat** `(tensors: Sequence[numpy.ndarray], dim: int = 0, out: Union[numpy.ndarray, NoneType] = None)`

**cat** `(tensors: Sequence[numpy.ndarray], dim: int = 0, out: Union[numpy.ndarray, NoneType] = None)`

**torch_chunk** `(input: numpy.ndarray, chunks: int, dim: int = 0)`

**chunk** `(input: numpy.ndarray, chunks: int, dim: int = 0)`

**torch_gather** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, out: Union[numpy.ndarray, NoneType] = None)`

**gather** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, out: Union[numpy.ndarray, NoneType] = None)`

**torch_index_select** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, out: Union[numpy.ndarray, NoneType] = None)`

**index_select** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, out: Union[numpy.ndarray, NoneType] = None)`

**torch_masked_select** `(input: numpy.ndarray, mask: numpy.ndarray, out: Union[numpy.ndarray, NoneType] = None)`

**masked_select** `(input: numpy.ndarray, mask: numpy.ndarray, out: Union[numpy.ndarray, NoneType] = None)`

**torch_movedim** `(input: numpy.ndarray, source: Union[int, Tuple[int]], destination: Union[int, Tuple[int]])`

**movedim** `(input: numpy.ndarray, source: Union[int, Tuple[int]], destination: Union[int, Tuple[int]])`

**torch_swapdims** `(input: numpy.ndarray, dim0: int, dim1: int)`

**swapdims** `(input: numpy.ndarray, dim0: int, dim1: int)`

**torch_narrow** `(input: numpy.ndarray, dim: int, start: int, length: int)`

**narrow** `(input: numpy.ndarray, dim: int, start: int, length: int)`

**torch_nonzero** `(input: numpy.ndarray, as_tuple: bool = False)`

**torch_scatter** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, src: numpy.ndarray, reduce: Union[str, NoneType] = None)`

**scatter** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, src: numpy.ndarray, reduce: Union[str, NoneType] = None)`

**torch_scatter_add** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, src: numpy.ndarray)`

**scatter_add** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, src: numpy.ndarray)`

**torch_split** `(input: numpy.ndarray, split_size_or_sections: Union[int, Sequence[int]], dim: int = 0)`

**torch_squeeze** `(input: numpy.ndarray, dim: Union[int, NoneType] = None)`

**torch_stack** `(input: Sequence[numpy.ndarray], dim: int = 0, out: Union[numpy.ndarray, NoneType] = None)`

**torch_unsqueeze** `(input: numpy.ndarray, dim: int)`

**unsqueeze** `(input: numpy.ndarray, dim: int)`

**torch_unbind** `(input: numpy.ndarray, dim: int = 0)`

**unbind** `(input: numpy.ndarray, dim: int = 0)`

**torch_meshgrid** `(*xi: numpy.ndarray)`

**torch_clone** `(input: numpy.ndarray)`

**clone** `(input: numpy.ndarray)`

**torch_is_contiguous** `(input: numpy.ndarray)`

**is_contiguous** `(input: numpy.ndarray)`

**torch_contiguous** `(input: numpy.ndarray)`

**contiguous** `(input: numpy.ndarray)`

**torch_repeat** `(input: numpy.ndarray, *sizes: int)`

**torch_repeat_interleave** `(input: numpy.ndarray, repeats: Union[int, numpy.ndarray], dim: Union[int, NoneType] = None)`

**repeat_interleave** `(input: numpy.ndarray, repeats: Union[int, numpy.ndarray], dim: Union[int, NoneType] = None)`

**torch_permute** `(input: numpy.ndarray, *axes: int)`

**permute** `(input: numpy.ndarray, *axes: int)`

**torch_view** `(input: numpy.ndarray, *sizes: int)`

**view** `(input: numpy.ndarray, *sizes: int)`

**torch_view_as** `(input: numpy.ndarray, other: numpy.ndarray)`

**view_as** `(input: numpy.ndarray, other: numpy.ndarray)`

**torch_expand** `(input: numpy.ndarray, *sizes: int)`

**expand** `(input: numpy.ndarray, *sizes: int)`

**torch_expand_as** `(input: numpy.ndarray, other: numpy.ndarray)`

**expand_as** `(input: numpy.ndarray, other: numpy.ndarray)`

**torch_t** `(input: numpy.ndarray)`

**t** `(input: numpy.ndarray)`

**torch_seed** `()`

**seed** `()`

**torch_manual_seed** `(seed: int)`

**manual_seed** `(seed: int)`

**torch_rand** `(*size: int, out: Union[numpy.ndarray, NoneType] = None)`

**rand** `(*size: int, out: Union[numpy.ndarray, NoneType] = None)`

**torch_rand_like** `(input: numpy.ndarray)`

**rand_like** `(input: numpy.ndarray)`

**torch_randint** `(low: int, high: Union[int, NoneType] = None, size: Union[Sequence[int], NoneType] = None, out: Union[numpy.ndarray, NoneType] = None)`

**randint** `(low: int, high: Union[int, NoneType] = None, size: Union[Sequence[int], NoneType] = None, out: Union[numpy.ndarray, NoneType] = None)`

**torch_randint_like** `(input: numpy.ndarray, low: int, high: Union[int, NoneType] = None)`

**randint_like** `(input: numpy.ndarray, low: int, high: Union[int, NoneType] = None)`

**torch_randn** `(*size: int, out: Union[numpy.ndarray, NoneType] = None)`

**randn** `(*size: int, out: Union[numpy.ndarray, NoneType] = None)`

**torch_randn_like** `(input: numpy.ndarray)`

**randn_like** `(input: numpy.ndarray)`

**torch_randperm** `(n: int, out: Union[numpy.ndarray, NoneType] = None)`

**randperm** `(n: int, out: Union[numpy.ndarray, NoneType] = None)`

**torch_clamp** `(input: numpy.ndarray, min: Union[numbers.Number, numpy.ndarray, NoneType] = None, max: Union[numbers.Number, numpy.ndarray, NoneType] = None)`

**clamp** `(input: numpy.ndarray, min: Union[numbers.Number, numpy.ndarray, NoneType] = None, max: Union[numbers.Number, numpy.ndarray, NoneType] = None)`

**torch_max** `(input: numpy.ndarray, dim: Union[int, numpy.ndarray, NoneType] = None, keepdim: bool = False, out: Union[numpy.ndarray, NoneType] = None)`

**max** `(input: numpy.ndarray, dim: Union[int, numpy.ndarray, NoneType] = None, keepdim: bool = False, out: Union[numpy.ndarray, NoneType] = None)`

**torch_min** `(input: numpy.ndarray, dim: Union[int, numpy.ndarray, NoneType] = None, keepdim: bool = False, out: Union[numpy.ndarray, NoneType] = None)`

**min** `(input: numpy.ndarray, dim: Union[int, numpy.ndarray, NoneType] = None, keepdim: bool = False, out: Union[numpy.ndarray, NoneType] = None)`

**torch_flatten** `(input: numpy.ndarray, start_dim: int = 0, end_dim: int = -1)`

**flatten** `(input: numpy.ndarray, start_dim: int = 0, end_dim: int = -1)`

#### ========== np.ndarray.* ==========
**torch_dim** `attribute`
**dim** `attribute`
**torch_numel** `(self)`

**numel** `(self)`

**torch_index_select** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, out: Union[numpy.ndarray, NoneType] = None)`

**index_select** `(input: numpy.ndarray, dim: int, index: numpy.ndarray, out: Union[numpy.ndarray, NoneType] = None)`

**torch_squeeze_** `(input: numpy.ndarray, dim: Union[int, NoneType] = None)`

**squeeze_** `(input: numpy.ndarray, dim: Union[int, NoneType] = None)`

**torch_unsqueeze** `(input: numpy.ndarray, dim: int)`

**unsqueeze** `(input: numpy.ndarray, dim: int)`

**torch_unsqueeze_** `(input: numpy.ndarray, dim: int)`

**unsqueeze_** `(input: numpy.ndarray, dim: int)`

**torch_is_contiguous** `(input: numpy.ndarray)`

**is_contiguous** `(input: numpy.ndarray)`

**torch_contiguous** `(input: numpy.ndarray)`

**contiguous** `(input: numpy.ndarray)`

**torch_clone** `(input: numpy.ndarray)`

**clone** `(input: numpy.ndarray)`

**torch_repeat** `(input: numpy.ndarray, *sizes: int)`

**torch_repeat_interleave** `(input: numpy.ndarray, repeats: Union[int, numpy.ndarray], dim: Union[int, NoneType] = None)`

**repeat_interleave** `(input: numpy.ndarray, repeats: Union[int, numpy.ndarray], dim: Union[int, NoneType] = None)`

**torch_view** `(input: numpy.ndarray, *sizes: int)`

**torch_permute** `(input: numpy.ndarray, *axes: int)`

**permute** `(input: numpy.ndarray, *axes: int)`

**torch_expand** `(input: numpy.ndarray, *sizes: int)`

**expand** `(input: numpy.ndarray, *sizes: int)`

**torch_expand_as** `(input: numpy.ndarray, other: numpy.ndarray)`

**expand_as** `(input: numpy.ndarray, other: numpy.ndarray)`

**torch_uniform_** `(input: numpy.ndarray, fro: int = 0, to: int = 1)`

**uniform_** `(input: numpy.ndarray, fro: int = 0, to: int = 1)`

**torch_normal_** `(input: numpy.ndarray, mean: int = 0, std: int = 1)`

**normal_** `(input: numpy.ndarray, mean: int = 0, std: int = 1)`

**torch_zero_** `(input: numpy.ndarray)`

**zero_** `(input: numpy.ndarray)`

**torch_clamp** `(input: numpy.ndarray, min: Union[numbers.Number, numpy.ndarray, NoneType] = None, max: Union[numbers.Number, numpy.ndarray, NoneType] = None)`

**clamp** `(input: numpy.ndarray, min: Union[numbers.Number, numpy.ndarray, NoneType] = None, max: Union[numbers.Number, numpy.ndarray, NoneType] = None)`

**torch_clamp_** `(input: numpy.ndarray, min: Union[numbers.Number, numpy.ndarray, NoneType] = None, max: Union[numbers.Number, numpy.ndarray, NoneType] = None)`

**clamp_** `(input: numpy.ndarray, min: Union[numbers.Number, numpy.ndarray, NoneType] = None, max: Union[numbers.Number, numpy.ndarray, NoneType] = None)`

**torch_clip_** `(input: numpy.ndarray, min: Union[numbers.Number, numpy.ndarray, NoneType] = None, max: Union[numbers.Number, numpy.ndarray, NoneType] = None)`

**clip_** `(input: numpy.ndarray, min: Union[numbers.Number, numpy.ndarray, NoneType] = None, max: Union[numbers.Number, numpy.ndarray, NoneType] = None)`

**torch_flatten** `(input: numpy.ndarray, start_dim: int = 0, end_dim: int = -1)`

**torch_max** `(input: numpy.ndarray, dim: Union[int, numpy.ndarray, NoneType] = None, keepdim: bool = False, out: Union[numpy.ndarray, NoneType] = None)`

**torch_min** `(input: numpy.ndarray, dim: Union[int, numpy.ndarray, NoneType] = None, keepdim: bool = False, out: Union[numpy.ndarray, NoneType] = None)`

**torch_amax** `(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)`

**amax** `(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)`

**torch_amin** `(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)`

**amin** `(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)`

**torch_half** `(self)`

**half** `(self)`

**torch_float** `(self)`

**float** `(self)`

**torch_double** `(self)`

**double** `(self)`

**torch_short** `(self)`

**short** `(self)`

**torch_int** `(self)`

**int** `(self)`

**torch_long** `(self)`

**long** `(self)`

**torch_bool** `(self)`

**bool** `(self)`

**torch_add** `(self, other)`

**add** `(self, other)`

**torch_add_** `(self, other)`

**add_** `(self, other)`

**torch_subtract** `(self, other)`

**subtract** `(self, other)`

**torch_subtract_** `(self, other)`

**subtract_** `(self, other)`

**torch_multiply** `(self, other)`

**multiply** `(self, other)`

**torch_multiply_** `(self, other)`

**multiply_** `(self, other)`

**torch_matmul** `(self, other)`

**matmul** `(self, other)`

**torch_matmul_** `(self, other)`

**matmul_** `(self, other)`

**torch_divide** `(self, other)`

**divide** `(self, other)`

**torch_divide_** `(self, other)`

**divide_** `(self, other)`

**torch_true_divide** `(self, other)`

**true_divide** `(self, other)`

**torch_true_divide_** `(self, other)`

**true_divide_** `(self, other)`

**torch_floor_divide** `(self, other)`

**floor_divide** `(self, other)`

**torch_floor_divide_** `(self, other)`

**floor_divide_** `(self, other)`

**torch_negative** `(self)`

**negative** `(self)`

**torch_negative_** `(self)`

**negative_** `(self)`

**torch_positive** `(self)`

**positive** `(self)`

**torch_positive_** `(self)`

**positive_** `(self)`

**torch_power** `(self, other)`

**power** `(self, other)`

**torch_power_** `(self, other)`

**power_** `(self, other)`

**torch_float_power** `(self, other)`

**float_power** `(self, other)`

**torch_float_power_** `(self, other)`

**float_power_** `(self, other)`

**torch_remainder** `(self, other)`

**remainder** `(self, other)`

**torch_remainder_** `(self, other)`

**remainder_** `(self, other)`

**torch_mod** `(self, other)`

**mod** `(self, other)`

**torch_mod_** `(self, other)`

**mod_** `(self, other)`

**torch_fmod** `(self, other)`

**fmod** `(self, other)`

**torch_fmod_** `(self, other)`

**fmod_** `(self, other)`

**torch_divmod** `(self, other)`

**divmod** `(self, other)`

**torch_divmod_** `(self, other)`

**divmod_** `(self, other)`

**torch_absolute** `(self)`

**absolute** `(self)`

**torch_absolute_** `(self)`

**absolute_** `(self)`

**torch_abs** `(self)`

**abs** `(self)`

**torch_abs_** `(self)`

**abs_** `(self)`

**torch_fabs** `(self)`

**fabs** `(self)`

**torch_fabs_** `(self)`

**fabs_** `(self)`

**torch_sign** `(self)`

**sign** `(self)`

**torch_sign_** `(self)`

**sign_** `(self)`

**torch_rint** `(self)`

**rint** `(self)`

**torch_rint_** `(self)`

**rint_** `(self)`

**torch_conj** `(self)`

**torch_conj_** `(self)`

**conj_** `(self)`

**torch_exp** `(self)`

**exp** `(self)`

**torch_exp_** `(self)`

**exp_** `(self)`

**torch_exp2** `(self)`

**exp2** `(self)`

**torch_exp2_** `(self)`

**exp2_** `(self)`

**torch_log** `(self)`

**log** `(self)`

**torch_log_** `(self)`

**log_** `(self)`

**torch_log2** `(self)`

**log2** `(self)`

**torch_log2_** `(self)`

**log2_** `(self)`

**torch_log10** `(self)`

**log10** `(self)`

**torch_log10_** `(self)`

**log10_** `(self)`

**torch_sqrt** `(self)`

**sqrt** `(self)`

**torch_sqrt_** `(self)`

**sqrt_** `(self)`

**torch_square** `(self)`

**square** `(self)`

**torch_square_** `(self)`

**square_** `(self)`

**torch_cbrt** `(self)`

**cbrt** `(self)`

**torch_cbrt_** `(self)`

**cbrt_** `(self)`

**torch_reciprocal** `(self)`

**reciprocal** `(self)`

**torch_reciprocal_** `(self)`

**reciprocal_** `(self)`

**torch_gcd** `(self, other)`

**gcd** `(self, other)`

**torch_gcd_** `(self, other)`

**gcd_** `(self, other)`

**torch_lcm** `(self, other)`

**lcm** `(self, other)`

**torch_lcm_** `(self, other)`

**lcm_** `(self, other)`

**torch_expm1** `(self)`

**expm1** `(self)`

**torch_expm1_** `(self)`

**expm1_** `(self)`

**torch_log1p** `(self)`

**log1p** `(self)`

**torch_log1p_** `(self)`

**log1p_** `(self)`

**torch_sin** `(self)`

**sin** `(self)`

**torch_sin_** `(self)`

**sin_** `(self)`

**torch_cos** `(self)`

**cos** `(self)`

**torch_cos_** `(self)`

**cos_** `(self)`

**torch_tan** `(self)`

**tan** `(self)`

**torch_tan_** `(self)`

**tan_** `(self)`

**torch_arcsin** `(self)`

**arcsin** `(self)`

**torch_arcsin_** `(self)`

**arcsin_** `(self)`

**torch_arccos** `(self)`

**arccos** `(self)`

**torch_arccos_** `(self)`

**arccos_** `(self)`

**torch_arctan** `(self)`

**arctan** `(self)`

**torch_arctan_** `(self)`

**arctan_** `(self)`

**torch_arctan2** `(self, other)`

**arctan2** `(self, other)`

**torch_arctan2_** `(self, other)`

**arctan2_** `(self, other)`

**torch_sinh** `(self)`

**sinh** `(self)`

**torch_sinh_** `(self)`

**sinh_** `(self)`

**torch_cosh** `(self)`

**cosh** `(self)`

**torch_cosh_** `(self)`

**cosh_** `(self)`

**torch_tanh** `(self)`

**tanh** `(self)`

**torch_tanh_** `(self)`

**tanh_** `(self)`

**torch_arcsinh** `(self)`

**arcsinh** `(self)`

**torch_arcsinh_** `(self)`

**arcsinh_** `(self)`

**torch_arccosh** `(self)`

**arccosh** `(self)`

**torch_arccosh_** `(self)`

**arccosh_** `(self)`

**torch_arctanh** `(self)`

**arctanh** `(self)`

**torch_arctanh_** `(self)`

**arctanh_** `(self)`

**torch_degrees** `(self)`

**degrees** `(self)`

**torch_degrees_** `(self)`

**degrees_** `(self)`

**torch_radians** `(self)`

**radians** `(self)`

**torch_radians_** `(self)`

**radians_** `(self)`

**torch_deg2rad** `(self)`

**deg2rad** `(self)`

**torch_deg2rad_** `(self)`

**deg2rad_** `(self)`

**torch_rad2deg** `(self)`

**rad2deg** `(self)`

**torch_rad2deg_** `(self)`

**rad2deg_** `(self)`

**torch_bitwise_and** `(self, other)`

**bitwise_and** `(self, other)`

**torch_bitwise_and_** `(self, other)`

**bitwise_and_** `(self, other)`

**torch_bitwise_or** `(self, other)`

**bitwise_or** `(self, other)`

**torch_bitwise_or_** `(self, other)`

**bitwise_or_** `(self, other)`

**torch_bitwise_xor** `(self, other)`

**bitwise_xor** `(self, other)`

**torch_bitwise_xor_** `(self, other)`

**bitwise_xor_** `(self, other)`

**torch_invert** `(self)`

**invert** `(self)`

**torch_invert_** `(self)`

**invert_** `(self)`

**torch_left_shift** `(self, other)`

**left_shift** `(self, other)`

**torch_left_shift_** `(self, other)`

**left_shift_** `(self, other)`

**torch_right_shift** `(self, other)`

**right_shift** `(self, other)`

**torch_right_shift_** `(self, other)`

**right_shift_** `(self, other)`

**torch_greater** `(self, other)`

**greater** `(self, other)`

**torch_greater_** `(self, other)`

**greater_** `(self, other)`

**torch_greater_equal** `(self, other)`

**greater_equal** `(self, other)`

**torch_greater_equal_** `(self, other)`

**greater_equal_** `(self, other)`

**torch_less** `(self, other)`

**less** `(self, other)`

**torch_less_** `(self, other)`

**less_** `(self, other)`

**torch_less_equal** `(self, other)`

**less_equal** `(self, other)`

**torch_less_equal_** `(self, other)`

**less_equal_** `(self, other)`

**torch_equal** `(self, other)`

**equal** `(self, other)`

**torch_equal_** `(self, other)`

**equal_** `(self, other)`

**torch_not_equal** `(self, other)`

**not_equal** `(self, other)`

**torch_not_equal_** `(self, other)`

**not_equal_** `(self, other)`

**torch_logical_and** `(self, other)`

**logical_and** `(self, other)`

**torch_logical_and_** `(self, other)`

**logical_and_** `(self, other)`

**torch_logical_or** `(self, other)`

**logical_or** `(self, other)`

**torch_logical_or_** `(self, other)`

**logical_or_** `(self, other)`

**torch_logical_not** `(self)`

**logical_not** `(self)`

**torch_logical_not_** `(self)`

**logical_not_** `(self)`

**torch_logical_xor** `(self, other)`

**logical_xor** `(self, other)`

**torch_logical_xor_** `(self, other)`

**logical_xor_** `(self, other)`

**torch_maximum** `(self, other)`

**maximum** `(self, other)`

**torch_maximum_** `(self, other)`

**maximum_** `(self, other)`

**torch_minimum** `(self, other)`

**minimum** `(self, other)`

**torch_minimum_** `(self, other)`

**minimum_** `(self, other)`

**torch_fmax** `(self, other)`

**fmax** `(self, other)`

**torch_fmax_** `(self, other)`

**fmax_** `(self, other)`

**torch_fmin** `(self, other)`

**fmin** `(self, other)`

**torch_fmin_** `(self, other)`

**fmin_** `(self, other)`

**torch_nextafter** `(self, other)`

**nextafter** `(self, other)`

**torch_nextafter_** `(self, other)`

**nextafter_** `(self, other)`

**torch_spacing** `(self)`

**spacing** `(self)`

**torch_spacing_** `(self)`

**spacing_** `(self)`

**torch_modf** `(self)`

**modf** `(self)`

**torch_modf_** `(self)`

**modf_** `(self)`

**torch_ldexp** `(self, other)`

**ldexp** `(self, other)`

**torch_ldexp_** `(self, other)`

**ldexp_** `(self, other)`

**torch_frexp** `(self)`

**frexp** `(self)`

**torch_frexp_** `(self)`

**frexp_** `(self)`

**torch_isfinite** `(self)`

**isfinite** `(self)`

**torch_isfinite_** `(self)`

**isfinite_** `(self)`

**torch_isinf** `(self)`

**isinf** `(self)`

**torch_isinf_** `(self)`

**isinf_** `(self)`

**torch_isnan** `(self)`

**isnan** `(self)`

**torch_isnan_** `(self)`

**isnan_** `(self)`

**torch_isnat** `(self)`

**isnat** `(self)`

**torch_isnat_** `(self)`

**isnat_** `(self)`

**torch_signbit** `(self)`

**signbit** `(self)`

**torch_signbit_** `(self)`

**signbit_** `(self)`

**torch_copysign** `(self, other)`

**copysign** `(self, other)`

**torch_copysign_** `(self, other)`

**copysign_** `(self, other)`

**torch_floor** `(self)`

**floor** `(self)`

**torch_floor_** `(self)`

**floor_** `(self)`

**torch_ceil** `(self)`

**ceil** `(self)`

**torch_ceil_** `(self)`

**ceil_** `(self)`

**torch_trunc** `(self)`

**trunc** `(self)`

**torch_trunc_** `(self)`

**trunc_** `(self)`

**torch_atan** `(self)`

**atan** `(self)`

**torch_atan_** `(self)`

**atan_** `(self)`

**torch_asin** `(self)`

**asin** `(self)`

**torch_asin_** `(self)`

**asin_** `(self)`

**torch_acos** `(self)`

**acos** `(self)`

**torch_acos_** `(self)`

**acos_** `(self)`

**torch_atan2** `(self, other)`

**atan2** `(self, other)`

**torch_atan2_** `(self, other)`

**atan2_** `(self, other)`

**torch_ge** `(self, other)`

**ge** `(self, other)`

**torch_ge_** `(self, other)`

**ge_** `(self, other)`

**torch_gt** `(self, other)`

**gt** `(self, other)`

**torch_gt_** `(self, other)`

**gt_** `(self, other)`

**torch_le** `(self, other)`

**le** `(self, other)`

**torch_le_** `(self, other)`

**le_** `(self, other)`

**torch_lt** `(self, other)`

**lt** `(self, other)`

**torch_lt_** `(self, other)`

**lt_** `(self, other)`