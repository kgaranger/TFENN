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

import unittest
from itertools import product

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax import random

from TFENN.core import (DenseSymmetricTensor, GRUCellSymmetricTensor,
                        RotateSymmetricTensor, SymmetricTensorNotationType,
                        SymmetricTensorRepresentation, TensorActivation,
                        TensorSymmetryClassType)
from TFENN.util.random import (uniformDihedralR2, uniformO2, uniformO3,
                               uniformOctahedralR3)

jax.config.update("jax_enable_x64", True)


def rotate_full(e_tensor, rot_mat):
    return jnp.einsum("...ij,...jk,...lk", rot_mat, e_tensor, rot_mat)


def rotate_reduced(e_notation, e_tensor, rot_mat):
    return e_notation.to_reduced(rotate_full(e_notation.to_full(e_tensor), rot_mat))


class TestActivationTensor(unittest.TestCase):
    key = random.PRNGKey(0)
    n = 100

    def test_initialization(self):
        """Test that the initializations of the tensor activations perform as
        expected."""
        for activation, feature_notation_type, dim in product(
            [nn.relu, nn.sigmoid, nn.tanh],
            [
                SymmetricTensorNotationType.FULL,
                SymmetricTensorNotationType.MANDEL,
                SymmetricTensorNotationType.VOIGT,
            ],
            [2, 3],
        ):
            with self.subTest(
                activation=activation,
                feature_notation_type=feature_notation_type,
                dim=dim,
            ):
                feature_notation = feature_notation_type.create(dim=dim, order=2)
                tensor_activation = TensorActivation(activation, feature_notation)
                self.assertEqual(tensor_activation.activation, activation, "activation")
                self.assertEqual(
                    tensor_activation.feature_notation,
                    feature_notation,
                    "feature_notation",
                )

    def test_call(self):
        """Test that the call of the tensor activations perform as expected."""
        for activation, feature_notation_type, dim in product(
            [nn.relu, nn.sigmoid, nn.tanh],
            [
                SymmetricTensorNotationType.FULL,
                SymmetricTensorNotationType.MANDEL,
                SymmetricTensorNotationType.VOIGT,
            ],
            [2, 3],
        ):
            with self.subTest(
                activation=activation,
                feature_notation_type=feature_notation_type,
                dim=dim,
            ):
                feature_notation = feature_notation_type.create(dim=dim, order=2)
                tensor_activation = TensorActivation(activation, feature_notation)
                x = random.normal(self.key, (self.n, *feature_notation.reduced_shape))
                x_full = feature_notation.to_full(x)
                v, u = jnp.linalg.eigh(x_full)
                res1 = feature_notation.to_reduced(
                    jnp.einsum("nij,nj,nkj->nik", u, activation(v), u)
                )
                res2 = tensor_activation.apply({}, x)

                self.assertEqual(x.shape, res2.shape, "shape")
                self.assertEqual(x.dtype, res2.dtype, "dtype")
                self.assertTrue(jnp.allclose(res1, res2), "value")


class TestDenseSymmetricTensor(unittest.TestCase):
    key = random.PRNGKey(0)
    in_features = (4,)
    out_features = (5,)
    n = 100

    def test_initialization(self):
        """Test that the initializations of the dense symmetric tensor perform as
        expected."""
        for sym_cls_type, dim in product(
            [
                TensorSymmetryClassType.ISOTROPIC,
                TensorSymmetryClassType.CUBIC,
                TensorSymmetryClassType.TRICLINIC,
                TensorSymmetryClassType.NONE,
            ],
            [2, 3],
        ):
            with self.subTest(sym_cls_type=sym_cls_type, dim=dim):
                kernel_rep = SymmetricTensorRepresentation(
                    dim=dim,
                    order=4,
                    notation_type=SymmetricTensorNotationType.MANDEL,
                    sym_cls_type=sym_cls_type,
                )
                bias_rep = SymmetricTensorRepresentation(
                    dim=dim,
                    order=2,
                    notation_type=SymmetricTensorNotationType.MANDEL,
                    sym_cls_type=sym_cls_type,
                )
                dense_sym_tensor = DenseSymmetricTensor(
                    kernel_rep=kernel_rep,
                    bias_rep=bias_rep,
                    features=self.out_features,
                )
                self.assertEqual(dense_sym_tensor.kernel_rep, kernel_rep, "kernel_rep")
                self.assertEqual(dense_sym_tensor.bias_rep, bias_rep, "bias_rep")
                self.assertEqual(
                    dense_sym_tensor.features, self.out_features, "features"
                )

    def test_call(self):
        """Test that the call of the dense symmetric tensor perform as expected."""
        for sym_cls_type, dim in product(
            [
                TensorSymmetryClassType.ISOTROPIC,
                TensorSymmetryClassType.CUBIC,
                TensorSymmetryClassType.TRICLINIC,
                TensorSymmetryClassType.NONE,
            ],
            [2, 3],
        ):
            with self.subTest(sym_cls_type=sym_cls_type, dim=dim):
                kernel_rep = SymmetricTensorRepresentation(
                    dim=dim,
                    order=4,
                    notation_type=SymmetricTensorNotationType.MANDEL,
                    sym_cls_type=sym_cls_type,
                )
                bias_rep = SymmetricTensorRepresentation(
                    dim=dim,
                    order=2,
                    notation_type=SymmetricTensorNotationType.MANDEL,
                    sym_cls_type=sym_cls_type,
                )
                dense_sym_tensor = DenseSymmetricTensor(
                    kernel_rep=kernel_rep,
                    bias_rep=bias_rep,
                    features=self.out_features,
                )
                x = random.normal(
                    self.key, self.in_features + bias_rep.notation.reduced_shape
                )
                params_key, self.key = random.split(self.key)
                params = dense_sym_tensor.init(
                    params_key,
                    jnp.ones(self.in_features + bias_rep.notation.reduced_shape),
                )
                res = dense_sym_tensor.apply(params, x)
                self.assertEqual(
                    res.shape,
                    self.out_features + bias_rep.notation.reduced_shape,
                    "shape",
                )
                self.assertEqual(res.dtype, x.dtype, "dtype")

    def test_equivariance(self):
        """Test that the dense symmetric tensor is equivariant."""
        for sym_cls_type, dim in product(
            [
                TensorSymmetryClassType.ISOTROPIC,
                TensorSymmetryClassType.CUBIC,
            ],
            [2, 3],
        ):
            with self.subTest(sym_cls_type=sym_cls_type, dim=dim):
                if sym_cls_type == TensorSymmetryClassType.ISOTROPIC:
                    if dim == 2:
                        O_mats = uniformO2(self.key, self.n)
                    elif dim == 3:
                        O_mats = uniformO3(self.key, self.n)
                    else:
                        raise ValueError(f"dim={dim} not supported")
                elif sym_cls_type == TensorSymmetryClassType.CUBIC:
                    if dim == 2:
                        O_mats = uniformDihedralR2(
                            self.key, 4, min(self.n, 4), replace=False
                        )
                    elif dim == 3:
                        O_mats = uniformOctahedralR3(
                            self.key, min(self.n, 48), replace=False
                        )

                kernel_rep = SymmetricTensorRepresentation(
                    dim=dim,
                    order=4,
                    notation_type=SymmetricTensorNotationType.MANDEL,
                    sym_cls_type=sym_cls_type,
                )
                bias_rep = SymmetricTensorRepresentation(
                    dim=dim,
                    order=2,
                    notation_type=SymmetricTensorNotationType.MANDEL,
                    sym_cls_type=sym_cls_type,
                )
                dense_sym_tensor = DenseSymmetricTensor(
                    kernel_rep=kernel_rep,
                    bias_rep=bias_rep,
                    features=self.out_features,
                )
                x = random.normal(
                    self.key, self.in_features + bias_rep.notation.reduced_shape
                )
                params_key, self.key = random.split(self.key)
                params = dense_sym_tensor.init(
                    params_key,
                    jnp.ones(self.in_features + bias_rep.notation.reduced_shape),
                )
                x1 = dense_sym_tensor.apply(
                    params,
                    rotate_reduced(
                        bias_rep.notation, x[None, ...], O_mats[:, None, ...]
                    ),
                )
                x2 = rotate_reduced(
                    bias_rep.notation,
                    dense_sym_tensor.apply(params, x[None, ...])[None, ...],
                    O_mats[:, None, ...],
                )

                self.assertTrue(jnp.allclose(x1, x2), f"{x1}\n{x2}")


if __name__ == "__main__":
    unittest.main()
