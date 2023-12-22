# Copyright 2023 Kévin Garanger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from TFENN.util.array_util import canonicalize_tuple, normalize_axes
from TFENN.util.geometry import angle_to_rot_mat_2d, quat_to_rot_mat_3d

from .symmetric_tensor_representation import (MandelNotation,
                                              SymmetricTensorNotation,
                                              SymmetricTensorNotationType,
                                              SymmetricTensorRepresentation)
from .tensor_symmetry_class import TensorSymmetryClassType


class TensorActivation(nn.module.Module):
    """Activation function that applies to the eigenvalues of tensor features
    :param activation: Activation function to apply to the eigenvalues.
    :type activation nn.module.Module
    :param feature_notation: Notation of the features.
    :type feature_notation: TensorNotation
    """

    activation: nn.Module
    feature_notation: SymmetricTensorNotation

    @nn.compact
    def __call__(self, inputs):
        v, u = jnp.linalg.eigh(self.feature_notation.to_full(inputs))
        v = self.activation(v)
        return self.feature_notation.to_reduced(
            jnp.einsum("...ij,...j,...kj->...ik", u, v, u)
        )


def tensor_init_wrap(tensor_init, tensor_rep, rng, shape, dtype):
    """Wrapper for tensor initialization that handles tensor representations.
    :param tensor_rep: the representation of the tensor.
    :type tensor_rep: SymmetricTensorRepresentation
    :param tensor_init: initializer function for the weight tensor.
    :type tensor_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    :param rng: random number generator key.
    :type rng: PRNGKey
    :param shape: shape of the weights to initialize.
    :type shape: tuple[int...]
    :param dtype: data type of the weights.
    :type dtype: Any
    """
    full_tensors = tensor_init(rng, shape[:-1] + tensor_rep.notation.full_shape, dtype)
    reduced_tensors = tensor_rep.notation.to_reduced(full_tensors)
    return tensor_rep.tensors_to_params(reduced_tensors)


class RotateSymmetricTensor(nn.module.Module):
    """Layer that rotates a symmetric tensor.
    :param dim: the dimension of the tensors to rotate.
    :type dim: int
    :param axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    :type axis: int or tuple
    :param dtype: the dtype of the computation (default: infer from input and params).
    :type dtype: Any or None
    :param param_dtype: the dtype passed to parameter initializers (default: float).
    :type param_dtype: Any
    :param precision: numerical precision of the computation see `jax.lax.Precision` for
        details.
    :type precision: None or str or jax.lax.Precision or tuple[str, str] or
        tuple[jax.lax.Precision, jax.lax.Precision]
    :param rotation_init: initializer function for the rotation matrix.
    :type rotation_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    """

    dim: int
    axis: int | Sequence[int] = -2
    dtype: jnp.dtype | None = None
    param_dtype: jnp.dtype | None = float
    precision: None | str | jax.lax.Precision | tuple[str, str] | tuple[
        jax.lax.Precision, jax.lax.Precision
    ] = None
    rotation_init: Callable[
        [jax.random.PRNGKey, tuple[int, int], Any], Any
    ] = nn.initializers.lecun_normal()
    dot_general: Callable[..., Any] = jax.lax.dot_general

    @nn.compact
    def __call__(self, inputs, transpose: bool = False):
        axis = canonicalize_tuple(self.axis)
        ndim = inputs.ndim
        axis = normalize_axes(axis, ndim)
        n_axis = len(axis)
        n_tensor_axis = 1  # Only rank 2 Mandel tensors are supported
        tensor_axis = normalize_axes(range(-n_tensor_axis, 0), ndim)
        if len(set(axis) & set(tensor_axis)) != 0:
            raise ValueError(
                "The tensor axis and the dense axis must be disjoint. "
                f"Got tensor axis {tensor_axis} and dense axis {axis}."
            )

        rot = self.param(
            "rotation_params",
            self.rotation_init,
            ((1 if self.dim == 2 else 4), 1),
        )[:, 0]

        inputs, rot = nn.dtypes.promote_dtype(inputs, rot, dtype=self.dtype)

        if self.dim == 2:
            rot_mat = angle_to_rot_mat_2d(rot[0])
        elif self.dim == 3:
            rot_mat = quat_to_rot_mat_3d(rot / jnp.linalg.norm(rot))
        else:
            raise NotImplementedError

        mandel_rotation = (
            MandelNotation(rank=4, dim=self.dim).to_reduced(
                rot_mat[:, None, :, None] * rot_mat[None, :, None, :]
                + rot_mat[None, :, :, None] * rot_mat[:, None, None, :]
            )
            / 2
        )

        out = self.dot_general(
            inputs,
            mandel_rotation,
            ((tensor_axis, (int(transpose),)), ((), ())),
        )

        return out


class DenseSymmetricTensor(nn.module.Module):
    """Dense layer for symmetric tensors.
    :param kernel_rep: the representation of the rank 4 kernel tensor.
    :type kernel_rep: SymmetricTensorRepresentation
    :param bias_rep: the representation of the rank 2 bias tensor.
    :type bias_rep: SymmetricTensorRepresentation
    :param features: the number or shape of output features.
    :type features: int or tuple
    :param axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    :type axis: int or tuple
    :param use_bias: whether to add a bias to the output (default: True).
    :type use_bias: bool
    :param dtype: the dtype of the computation (default: infer from input and params).
    :type dtype: Any or None
    :param param_dtype: the dtype passed to parameter initializers (default: float).
    :type param_dtype: Any
    :param precision: numerical precision of the computation see `jax.lax.Precision` for
        details.
    :type precision: None or str or jax.lax.Precision or tuple[str, str] or
        tuple[jax.lax.Precision, jax.lax.Precision]
    :param kernel_init: initializer function for the weight matrix.
    :type kernel_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    :param bias_init: initializer function for the bias.
    :type bias_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    """

    kernel_rep: SymmetricTensorRepresentation
    bias_rep: SymmetricTensorRepresentation
    features: int | Sequence[int]
    axis: int | Sequence[int] = -2
    use_bias: bool = True
    dtype: jnp.dtype | None = None
    param_dtype: jnp.dtype | None = float
    precision: None | str | jax.lax.Precision | tuple[str, str] | tuple[
        jax.lax.Precision, jax.lax.Precision
    ] = None
    kernel_init: Callable[
        [jax.random.PRNGKey, tuple[int, int], Any], Any
    ] = nn.initializers.lecun_normal()
    bias_init: Callable[
        [jax.random.PRNGKey, tuple[int], Any], Any
    ] = nn.initializers.zeros_init()
    dot_general: Callable[..., Any] = jax.lax.dot_general

    @nn.compact
    def __call__(self, inputs, tensor_basis: jnp.ndarray | None = None):
        features = canonicalize_tuple(self.features)
        axis = canonicalize_tuple(self.axis)
        ndim = inputs.ndim
        axis = normalize_axes(axis, ndim)
        n_axis = len(axis)
        n_features = len(features)
        n_tensor_axis = 1  # Only rank 2 Mandel tensors are supported
        tensor_axis = normalize_axes(range(-n_tensor_axis, 0), ndim)
        n_red_tensor_axis = len(self.kernel_rep.notation.reduced_shape)  # Should be 2
        if len(set(axis) & set(tensor_axis)) != 0:
            raise ValueError(
                "The tensor axis and the dense axis must be disjoint. "
                f"Got tensor axis {tensor_axis} and dense axis {axis}."
            )

        axis_shape = tuple(jnp.shape(inputs)[ax] for ax in axis)
        kernel_params = self.param(
            "kernel_params",
            lambda rng, shape, dtype=float: tensor_init_wrap(
                self.kernel_init, self.kernel_rep, rng, shape, dtype
            ),
            axis_shape + features + (self.kernel_rep.basis_size,),
        )
        contract_ind = tuple(range(n_axis)) + normalize_axes(
            tuple(range(-n_tensor_axis, 0)), n_axis + n_features + n_red_tensor_axis
        )
        if self.use_bias:
            bias_params = self.param(
                "bias_params",
                lambda rng, shape, dtype=float: tensor_init_wrap(
                    self.bias_init, self.bias_rep, rng, shape, dtype
                ),
                features + (self.bias_rep.basis_size,),
            )
        else:
            bias_params = None
        inputs, kernel_params, bias_params = nn.dtypes.promote_dtype(
            inputs, kernel_params, bias_params, dtype=self.dtype
        )

        out = self.dot_general(
            inputs,
            self.kernel_rep.params_to_tensors(kernel_params, basis=tensor_basis),
            ((axis + tensor_axis, contract_ind), ((), ())),
        )

        if self.use_bias:
            out = out + self.bias_rep.params_to_tensors(bias_params, basis=tensor_basis)

        return out


class DenseGateSymmetricTensor(nn.module.Module):
    """Dense gate layer for symmetric tensors.
    :param kernel_rep: the representation of the rank 2 kernel tensor.
    :type kernel_rep: SymmetricTensorRepresentation
    :param features: the number or shape of output features.
    :type features: int or tuple
    :param axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    :type axis: int or tuple
    :param use_bias: whether to add a bias to the output (default: True).
    :type use_bias: bool
    :param dtype: the dtype of the computation (default: infer from input and params).
    :type dtype: Any or None
    :param param_dtype: the dtype passed to parameter initializers (default: float).
    :type param_dtype: Any
    :param precision: numerical precision of the computation see `jax.lax.Precision` for
        details.
    :type precision: None or str or jax.lax.Precision or tuple[str, str] or
        tuple[jax.lax.Precision, jax.lax.Precision]
    :param kernel_init: initializer function for the weight matrix.
    :type kernel_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    :param bias_init: initializer function for the bias.
    :type bias_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    """

    kernel_rep: SymmetricTensorRepresentation
    features: int | Sequence[int]
    axis: int | Sequence[int] = -2
    use_bias: bool = True
    dtype: jnp.dtype | None = None
    param_dtype: jnp.dtype | None = float
    precision: None | str | jax.lax.Precision | tuple[str, str] | tuple[
        jax.lax.Precision, jax.lax.Precision
    ] = None
    kernel_init: Callable[
        [jax.random.PRNGKey, tuple[int, int], Any], Any
    ] = nn.initializers.lecun_normal()
    bias_init: Callable[
        [jax.random.PRNGKey, tuple[int], Any], Any
    ] = nn.initializers.zeros_init()
    dot_general: Callable[..., Any] = jax.lax.dot_general

    @nn.compact
    def __call__(self, inputs, tensor_basis: jnp.ndarray | None = None):
        features = canonicalize_tuple(self.features)
        axis = canonicalize_tuple(self.axis)
        ndim = inputs.ndim
        axis = normalize_axes(axis, ndim)
        n_axis = len(axis)
        n_features = len(features)
        n_tensor_axis = 1  # Only rank 2 Mandel tensors are supported
        tensor_axis = normalize_axes(range(-n_tensor_axis, 0), ndim)
        n_red_tensor_axis = len(self.kernel_rep.notation.reduced_shape)  # Should be 1
        if len(set(axis) & set(tensor_axis)) != 0:
            raise ValueError(
                "The tensor axis and the dense axis must be disjoint. "
                f"Got tensor axis {tensor_axis} and dense axis {axis}."
            )

        axis_shape = tuple(jnp.shape(inputs)[ax] for ax in axis)
        kernel_params = self.param(
            "kernel_params",
            lambda rng, shape, dtype=float: tensor_init_wrap(
                self.kernel_init, self.kernel_rep, rng, shape, dtype
            ),
            axis_shape + features + (self.kernel_rep.basis_size,),
        )
        contract_ind = tuple(range(n_axis)) + normalize_axes(
            tuple(range(-n_tensor_axis, 0)), n_axis + n_features + n_red_tensor_axis
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, features)
        else:
            bias = None
        inputs, kernel_params, bias = nn.dtypes.promote_dtype(
            inputs, kernel_params, bias, dtype=self.dtype
        )

        out = self.dot_general(
            inputs,
            self.kernel_rep.params_to_tensors(kernel_params, basis=tensor_basis),
            ((axis + tensor_axis, contract_ind), ((), ())),
        )

        if self.use_bias:
            out = out + bias

        return out


class GRUCellSymmetricTensor(nn.RNNCellBase):
    """GRU cell for symmetric tensors.
    :param kernel_rep: Representation of the kernel.
    :type kernel_rep: SymmetricTensorRepresentation
    :param bias_rep: Representation of the bias.
    :type bias_rep: SymmetricTensorRepresentation
    :param features: the number or shape of output features.
    :type features: int or tuple
    :param axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    :type axis: int or tuple
    :param gate_fn: activation function used for gates (default: sigmoid)
    :type gate_fn: Callable[[jnp.ndarray], jnp.ndarray]
    :param activation_fn: activation function used for output and memory update
      (default: tanh).
    :type activation_fn: Callable[[jnp.ndarray], jnp.ndarray]
    :param kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    :type kernel_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    :param recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    :type recurrent_kernel_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    :param bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    :type bias_init: Callable[[PRNGKey, tuple[int...], Any], Any]
    :param dtype: the dtype of the computation (default: None).
    :type dtype: Any
    :param param_dtype: the dtype passed to parameter initializers (default: float).
    :type param_dtype: Any
    """

    kernel_rep: SymmetricTensorRepresentation
    bias_rep: SymmetricTensorRepresentation
    features: int | Sequence[int]

    gate_fn: Callable[..., Any] = nn.activation.sigmoid
    activation_fn: Callable[..., Any] = nn.activation.tanh
    kernel_init: Callable[
        [jax.random.PRNGKey, tuple[int, int], Any], Any
    ] = nn.initializers.lecun_normal()
    recurrent_kernel_init: Callable[
        [jax.random.PRNGKey, tuple[int, int], Any], Any
    ] = nn.initializers.orthogonal()
    bias_init: Callable[
        [jax.random.PRNGKey, tuple[int], Any], Any
    ] = nn.initializers.zeros_init()
    dtype: jnp.dtype | None = None
    param_dtype: jnp.dtype | None = float
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        self.feature_notation = SymmetricTensorNotationType.MANDEL.create(
            rank=2, dim=self.kernel_rep.dim
        )

    @nn.compact
    def __call__(self, carry, inputs):
        """Gated recurrent unit (GRU) cell.
        :param carry: the previous output of the GRU cell.
        :type carry: jnp.ndarray
        :param inputs: the input to the GRU cell.
        :type inputs: jnp.ndarray
        :return: the new output of the GRU cell.
        :rtype: jnp.ndarray
        """
        h = carry

        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(
            DenseGateSymmetricTensor,
            kernel_rep=self.bias_rep,
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
        )
        dense_i = partial(
            DenseGateSymmetricTensor,
            kernel_rep=self.bias_rep,
            features=self.features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        gate_dense = partial(
            DenseGateSymmetricTensor,
            kernel_rep=self.bias_rep,
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            bias_init=self.bias_init,
        )

        tensor_dense = partial(
            DenseSymmetricTensor,
            kernel_rep=self.kernel_rep,
            bias_rep=self.bias_rep,
            features=self.features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            bias_init=self.bias_init,
        )

        r = jnp.expand_dims(
            self.gate_fn(
                gate_dense(name="ir", use_bias=True, kernel_init=self.kernel_init)(
                    inputs
                )
                + gate_dense(
                    name="hr", use_bias=False, kernel_init=self.recurrent_kernel_init
                )(h)
            ),
            -1,
        )

        z = jnp.expand_dims(
            self.gate_fn(
                gate_dense(name="iz", use_bias=True, kernel_init=self.kernel_init)(
                    inputs
                )
                + gate_dense(
                    name="hz", use_bias=False, kernel_init=self.recurrent_kernel_init
                )(h)
            ),
            -1,
        )

        n = TensorActivation(self.activation_fn, self.feature_notation)(
            tensor_dense(name="in", kernel_init=self.kernel_init)(inputs)
            + r * tensor_dense(name="hn", kernel_init=self.recurrent_kernel_init)(h)
        )
        new_h = (1.0 - z) * n + z * h
        return new_h, new_h

    @nn.nowrap
    def initialize_carry(self, rng: jax.random.PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        batch_dims = input_shape[:-2]
        mem_shape = (
            batch_dims
            + canonicalize_tuple(self.features)
            + self.feature_notation.reduced_shape
        )
        return self.carry_init(rng, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 2
