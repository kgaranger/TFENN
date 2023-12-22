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

import jax
from jax import numpy as jnp
from jax import random

from TFENN.util.random import (uniformChiralOctahedralR3, uniformCyclicR2,
                               uniformDihedralR2, uniformO2, uniformO3,
                               uniformOctahedralR3, uniformSO2, uniformSO3)

jax.config.update("jax_enable_x64", True)


class TestUtilRandom(unittest.TestCase):
    key = random.PRNGKey(0)
    shape = (5, 20)
    cycles = (1, 2, 3, 4)

    def setUp(self):
        (
            SO2_key,
            O2_key,
            SO3_key,
            O3_key,
            C2_key,
            D2_key,
            COh3_key,
            Oh3_key,
        ) = random.split(self.key, 8)
        self.SO2 = uniformSO2(SO2_key, shape=self.shape)
        self.O2 = uniformO2(O2_key, shape=self.shape)
        self.SO3 = uniformSO3(SO3_key, shape=self.shape)
        self.O3 = uniformO3(O3_key, shape=self.shape)
        self.C2 = [uniformCyclicR2(C2_key, n, shape=self.shape) for n in self.cycles]
        self.D2 = [uniformDihedralR2(D2_key, n, shape=self.shape) for n in self.cycles]
        self.COh3 = uniformChiralOctahedralR3(COh3_key, shape=self.shape)
        self.Oh3 = uniformOctahedralR3(Oh3_key, shape=self.shape)

    def test_uniformSO2_shape(self):
        """Test that the shape of the returned matrices of uniformSO2 is correct."""
        self.assertEqual(
            self.SO2.shape,
            self.shape + (2, 2),
            f"shape of matrices returned by uniformSO2 is {self.SO2.shape} instead of"
            f"{self.shape + (2, 2)}",
        )

    def test_uniformSO2_orthogonality(self):
        """Test that the matrices returned by uniformSO2 are orthogonal."""
        self.assertTrue(
            jnp.allclose(
                jnp.expand_dims(jnp.eye(2), axis=range(len(self.shape))),
                jnp.einsum("...ij,...kj->...ik", self.SO2, self.SO2),
            ),
            "Some matrices returned by uniformSO2 are not orthogonal",
        )

    def test_uniformSO2_determinant(self):
        """Test that the matrices returned by uniformSO2 have determinant 1."""
        self.assertTrue(
            jnp.allclose(
                jnp.ones(self.shape),
                jnp.linalg.det(self.SO2),
            ),
            "Some matrices returned by uniformSO2 do not have determinant 1",
        )

    def test_uniformO2_shape(self):
        """Test that the shape of the returned matrices of uniformO2 is correct."""
        self.assertEqual(
            self.O2.shape,
            self.shape + (2, 2),
            f"shape of matrices returned by uniformO2 is {self.O2.shape} instead of"
            f"{self.shape + (2, 2)}",
        )

    def test_uniformO2_orthogonality(self):
        """Test that the matrices returned by uniformO2 are orthogonal."""
        self.assertTrue(
            jnp.allclose(
                jnp.expand_dims(jnp.eye(2), axis=range(len(self.shape))),
                jnp.einsum("...ij,...kj->...ik", self.O2, self.O2),
            ),
            "Some matrices returned by uniformO2 are not orthogonal",
        )

    def test_uniformO2_determinant(self):
        """Test that some matrices returned by uniformO2 have determinant -1 and others
        1."""
        self.assertTrue(
            jnp.any(
                jnp.isclose(
                    jnp.linalg.det(self.O2),
                    1,
                )
            ),
            "All matrices returned by uniformO2 have determinant -1",
        )
        self.assertTrue(
            jnp.any(
                jnp.isclose(
                    jnp.linalg.det(self.O2),
                    -1,
                )
            ),
            "All matrices returned by uniformO2 have determinant 1",
        )

    def test_uniformO3_shape(self):
        """Test that the shape of the returned matrices of uniformO3 is correct."""
        self.assertEqual(
            self.O3.shape,
            self.shape + (3, 3),
            f"shape of matrices returned by uniformO3 is {self.O3.shape} instead of"
            f"{self.shape + (3, 3)}",
        )

    def test_uniformO3_orthogonality(self):
        """Test that the matrices returned by uniformO3 are orthogonal."""
        self.assertTrue(
            jnp.allclose(
                jnp.expand_dims(jnp.eye(3), axis=range(len(self.shape))),
                jnp.einsum("...ij,...kj->...ik", self.O3, self.O3),
            ),
            "Some matrices returned by uniformO3 are not orthogonal",
        )

    def test_uniform_CyclicR2_shape(self):
        """Test that the shape of the returned matrices of uniformCyclicR2 is
        correct."""
        for C2, n in zip(self.C2, self.cycles):
            self.assertEqual(
                C2.shape,
                self.shape + (2, 2),
                f"shape of matrices returned by uniformCyclicR2 is {C2.shape} instead"
                f"of {self.shape + (2, 2)} for n={n}",
            )

    def test_uniform_CyclicR2_orthogonality(self):
        """Test that the matrices returned by uniformCyclicR2 are orthogonal."""
        for C2, n in zip(self.C2, self.cycles):
            self.assertTrue(
                jnp.allclose(
                    jnp.expand_dims(jnp.eye(2), axis=range(len(self.shape))),
                    jnp.einsum("...ij,...kj->...ik", C2, C2),
                ),
                f"Some matrices returned by uniformCyclicR2 are not orthogonal for"
                "n={n}",
            )

    def test_uniform_CyclicR2_determinant(self):
        """Test that the matrices returned by uniformCyclicR2 have determinant 1."""
        for C2, n in zip(self.C2, self.cycles):
            self.assertTrue(
                jnp.allclose(
                    jnp.ones(self.shape),
                    jnp.linalg.det(C2),
                ),
                "Some matrices returned by uniformCyclicR2 do not have determinant 1"
                f"for n={n}",
            )

    def test_uniform_CyclicR2_nth_root(self):
        """Test that the matrices returned by uniformCyclicR2 are nth roots of unity."""
        for C2, n in zip(self.C2, self.cycles):
            self.assertTrue(
                jnp.allclose(
                    jnp.eye(2),
                    jnp.linalg.matrix_power(C2, n),
                ),
                f"Some matrices returned by uniformCyclicR2 are not solutions of x^{n}"
                "= I",
            )

    def test_uniformDihedralR2_shape(self):
        """Test that the shape of the returned matrices of uniformDihedralR2 is
        correct."""
        for D2, n in zip(self.D2, self.cycles):
            self.assertEqual(
                D2.shape,
                self.shape + (2, 2),
                f"shape of matrices returned by uniformDihedralR2 is {D2.shape} instead"
                f"of {self.shape + (2, 2)} for n={n}",
            )

    def test_uniformDihedralR2_orthogonality(self):
        """Test that the matrices returned by uniformDihedralR2 are orthogonal."""
        for D2, n in zip(self.D2, self.cycles):
            self.assertTrue(
                jnp.allclose(
                    jnp.expand_dims(jnp.eye(2), axis=range(len(self.shape))),
                    jnp.einsum("...ij,...kj->...ik", D2, D2),
                ),
                "Some matrices returned by uniformDihedralR2 are not orthogonal for"
                f"n={n}",
            )

    def test_uniformDihedralR2_determinant(self):
        """Test that some matrices returned by uniformDihedralR2 have determinant -1 and
        others
        1."""
        for D2, n in zip(self.D2, self.cycles):
            self.assertTrue(
                jnp.any(
                    jnp.isclose(
                        jnp.linalg.det(D2),
                        1,
                    )
                ),
                "All matrices returned by uniformDihedralR2 have determinant -1 for"
                f"n={n}",
            )
            self.assertTrue(
                jnp.any(
                    jnp.isclose(
                        jnp.linalg.det(D2),
                        -1,
                    )
                ),
                "All matrices returned by uniformDihedralR2 have determinant 1 for"
                f"n={n}",
            )

    def test_uniformDihedralR2_nth_root(self):
        """Test that the matrices returned by uniformDihedralR2 have order 2n."""
        for D2, n in zip(self.D2, self.cycles):
            self.assertTrue(
                jnp.allclose(
                    jnp.eye(2),
                    jnp.linalg.matrix_power(D2, 2 * n),
                ),
                (
                    "Some matrices returned by uniformDihedralR2 are not solutions of",
                    f"x^{2*n} = I for n={n}",
                ),
            )

    def test_uniformChiralOctahedralR3_shape(self):
        """Test that the shape of the returned matrices of uniformChiralOctahedralR3 is
        correct."""
        self.assertEqual(
            self.COh3.shape,
            self.shape + (3, 3),
            "shape of matrices returned by uniformChiralOctahedralR3 is"
            f"{self.COh3.shape} instead of {self.shape + (3, 3)}",
        )

    def test_uniformChiralOctahedralR3_orthogonality(self):
        """Test that the matrices returned by uniformChiralOctahedralR3 are
        orthogonal."""
        self.assertTrue(
            jnp.allclose(
                jnp.expand_dims(jnp.eye(3), axis=range(len(self.shape))),
                jnp.einsum("...ij,...kj->...ik", self.COh3, self.COh3),
            ),
            "Some matrices returned by uniformChiralOctahedralR3 are not orthogonal",
        )

    def test_uniformChiralOctahedralR3_determinant(self):
        """Test that the matrices returned by uniformChiralOctahedralR3 have determinant
        1."""
        self.assertTrue(
            jnp.allclose(
                jnp.ones(self.shape),
                jnp.linalg.det(self.COh3),
            ),
            "Some matrices returned by uniformChiralOctahedralR3 do not have"
            "determinant 1",
        )

    def test_uniformChiralOctahedralR3_nth_root(self):
        """Test that the matrices returned by uniformChiralOctahedralR3 are 12th roots
        of unity."""
        self.assertTrue(
            jnp.allclose(
                jnp.eye(3),
                jnp.linalg.matrix_power(self.COh3, 12),
            ),
            "Some matrices returned by uniformChiralOctahedralR3 are not solutions of"
            "x^{12} = I",
        )

    def test_uniformOctahedralR3_shape(self):
        """Test that the shape of the returned matrices of uniformOctahedralR3 is
        correct."""
        self.assertEqual(
            self.Oh3.shape,
            self.shape + (3, 3),
            f"shape of matrices returned by uniformOctahedralR3 is {self.Oh3.shape}"
            f"instead of {self.shape + (3, 3)}",
        )

    def test_uniformOctahedralR3_orthogonality(self):
        """Test that the matrices returned by uniformOctahedralR3 are orthogonal."""
        self.assertTrue(
            jnp.allclose(
                jnp.expand_dims(jnp.eye(3), axis=range(len(self.shape))),
                jnp.einsum("...ij,...kj->...ik", self.Oh3, self.Oh3),
            ),
            "Some matrices returned by uniformOctahedralR3 are not orthogonal",
        )

    def test_uniformOctahedralR3_determinant(self):
        """Test that some matrices returned by uniformOctahedralR3 have determinant -1
        and others 1."""
        self.assertTrue(
            jnp.any(
                jnp.isclose(
                    jnp.linalg.det(self.Oh3),
                    1,
                )
            ),
            "All matrices returned by uniformOctahedralR3 have determinant -1",
        )
        self.assertTrue(
            jnp.any(
                jnp.isclose(
                    jnp.linalg.det(self.Oh3),
                    -1,
                )
            ),
            "All matrices returned by uniformOctahedralR3 have determinant 1",
        )

    def test_uniformOctahedralR3_nth_root(self):
        """Test that the matrices returned by uniformOctahedralR3 have order 12."""
        self.assertTrue(
            jnp.allclose(
                jnp.eye(3),
                jnp.linalg.matrix_power(self.Oh3, 12),
            ),
            "Some matrices returned by octahedralR3 are not solutions of x^{12} = I",
        )


if __name__ == "__main__":
    unittest.main()
