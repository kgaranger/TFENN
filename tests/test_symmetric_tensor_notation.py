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
from jax import Array
from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike

from TFENN.core import (MandelNotation, SymmetricTensorNotationType,
                        VoigtNotation)

jax.config.update("jax_enable_x64", True)


class TestSymmetricTensorNotation(unittest.TestCase):
    key = random.PRNGKey(0)

    def test_initialization(self):
        """Test that the initializations of the tensor notations perform as expected."""
        for notation_type, dim, order in product(
            SymmetricTensorNotationType, range(1, 4), range(2, 7, 2)
        ):
            with self.subTest(notation_type=notation_type, dim=dim, order=order):
                notation = notation_type.create(dim=dim, order=order)
                self.assertEqual(notation.dim, dim, "dim")
                self.assertEqual(notation.order, order, "order")
                self.assertEqual(notation.full_shape, (dim,) * order, "full_shape")
                self.assertIsInstance(notation.reduced_shape, tuple, "reduced_shape")
                self.assertIsInstance(
                    notation.reduced_shape[0], int, "reduced_shape[0]"
                )
                self.assertIsInstance(
                    notation.to_reduced_indices, tuple, "to_reduced_indices"
                )
                for indices in notation.to_reduced_indices:
                    self.assertIsInstance(indices, ArrayLike, "to_reduced_indices[i]")
                self.assertIsInstance(
                    notation.to_reduced_scaling, ArrayLike | None, "to_reduced_scaling"
                )
                if isinstance(notation.to_reduced_scaling, ArrayLike):
                    self.assertEqual(
                        notation.to_reduced_scaling.shape,
                        notation.reduced_shape,
                        "to_reduced_scaling:"
                        f"shape={notation.to_reduced_scaling.shape}",
                    )
                self.assertIsInstance(
                    notation.to_full_indices, tuple, "to_full_indices"
                )
                for indices in notation.to_full_indices:
                    self.assertIsInstance(indices, ArrayLike, "to_full_indices[i]")
                self.assertIsInstance(
                    notation.to_full_scaling,
                    ArrayLike | None,
                    "to_full_scaling",
                )
                if isinstance(notation.to_full_scaling, ArrayLike):
                    self.assertEqual(
                        notation.to_full_scaling.shape,
                        notation.full_shape,
                        "to_full_scaling.shape",
                    )

        for notation_type in SymmetricTensorNotationType:
            with self.subTest(notation_type=notation_type):
                self.assertRaises(ValueError, notation_type.create, dim=2, order=1)
                self.assertRaises(ValueError, notation_type.create, dim=2, order=3)
                self.assertRaises(ValueError, notation_type.create, dim=0, order=2)
                self.assertRaises(ValueError, notation_type.create, dim=-1, order=2)

    def test_minor_symmetries(self):
        """Test that the minor symmetries of the full tensor indices and scaling
        factors are respected."""
        for notation_type, dim, order in product(
            [
                notation_type
                for notation_type in SymmetricTensorNotationType
                if notation_type != SymmetricTensorNotationType.FULL
            ],
            range(1, 4),
            range(2, 7, 2),
        ):
            with self.subTest(notation_type=notation_type, dim=dim, order=order):
                notation = notation_type.create(dim=dim, order=order)
                for i in range(len(notation.reduced_shape)):
                    for j in range(0, notation.order, 2):
                        self.assertTrue(
                            jnp.all(
                                notation.to_full_indices[i]
                                == (
                                    notation.to_full_indices[i].transpose(
                                        tuple(range(0, j))
                                        + (j + 1, j)
                                        + tuple(range(j + 2, notation.order))
                                    )
                                )
                            ),
                            f"to_full_indices, i={i}, j={j}",
                        )
                if isinstance(notation.to_full_scaling, jnp.ndarray):
                    for i in range(0, notation.order, 2):
                        self.assertTrue(
                            jnp.all(
                                notation.to_full_scaling
                                == notation.to_full_scaling.transpose(
                                    tuple(range(0, i))
                                    + (i + 1, i)
                                    + tuple(range(i + 2, notation.order))
                                )
                            ),
                            f"to_full_scaling, i={i}",
                        )

    def test_voigt_shape(self):
        """Test that the Voigt notation has the correct shape."""
        for dim, order in zip(range(1, 4), range(2, 7, 2)):
            with self.subTest(dim=dim, order=order):
                notation = VoigtNotation(dim=dim, order=order)
                self.assertEqual(
                    notation.reduced_shape,
                    (dim * (dim + 1) // 2,) * (order // 2),
                    "shape",
                )

    def test_voigt_indices(self):
        """Test that the Voigt notation has the correct indices."""
        v22 = VoigtNotation(dim=2, order=2)
        self.assertTrue(
            (
                jnp.all(v22.to_full_indices[0] == jnp.array([[0, 2], [2, 1]])),
                "dim=2, order=2",
            )
        )

        v32 = VoigtNotation(dim=3, order=2)
        self.assertTrue(
            jnp.all(
                v32.to_full_indices[0] == jnp.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
            ),
            "dim=3, order=2",
        )

        v24 = VoigtNotation(dim=2, order=4)
        self.assertTrue(
            jnp.all(
                v24.to_reduced_indices[0]
                == jnp.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
            ),
            "dim=2, order=4, i=0",
        )

    def test_reversibility(self):
        """Test that the transformation between the full and reduced indices is
        reversible."""
        for notation_type, dim, order in product(
            SymmetricTensorNotationType, range(1, 4), range(2, 7, 2)
        ):
            with self.subTest(notation_type=notation_type, dim=dim, order=order):
                notation = notation_type.create(dim=dim, order=order)
                for _ in range(10):
                    x = random.normal(self.key, notation.reduced_shape)
                    y = notation.to_reduced(notation.to_full(x))
                    self.assertTrue(jnp.allclose(x, y), "to_reduced(to_full(x))")

    def test_mandel_double_contraction_equivalence(self):
        """Test that the double contraction of the tensors is equivalent to the matrix
        product of their Mandel notations."""
        for dim in range(1, 4):
            with self.subTest(dim=dim):
                notation2 = MandelNotation(dim=dim, order=2)
                notation4 = MandelNotation(dim=dim, order=4)
                for _ in range(10):
                    x = random.normal(self.key, notation2.reduced_shape)
                    y = random.normal(self.key, notation2.reduced_shape)
                    z = random.normal(self.key, notation4.reduced_shape)
                    t = random.normal(self.key, notation4.reduced_shape)
                    x_tensor = notation2.to_full(x)
                    y_tensor = notation2.to_full(y)
                    z_tensor = notation4.to_full(z)
                    t_tensor = notation4.to_full(t)
                    xy = jnp.dot(x, y)
                    zx = z @ x
                    zt = z @ t
                    xy_tensor = jnp.einsum("...ij,...ij", x_tensor, y_tensor)
                    zx_tensor = notation2.to_reduced(
                        jnp.einsum("...ijkl,...kl->...ij", z_tensor, x_tensor)
                    )
                    zt_tensor = notation4.to_reduced(
                        jnp.einsum("...ijkl,...klmn->...ijmn", z_tensor, t_tensor)
                    )
                    self.assertTrue(
                        jnp.allclose(xy, xy_tensor), f"vector/vector product, dim={dim}"
                    )
                    self.assertTrue(
                        jnp.allclose(zx, zx_tensor), f"matrix/vector product, dim={dim}"
                    )
                    self.assertTrue(
                        jnp.allclose(zt, zt_tensor), f"matrix/matrix product, dim={dim}"
                    )


if __name__ == "__main__":
    unittest.main()
