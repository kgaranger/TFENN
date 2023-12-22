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
from functools import partial
from itertools import product

import jax
from jax import numpy as jnp
from jax import random

from TFENN.core import (SymmetricTensorNotationType,
                        SymmetricTensorRepresentation, TensorSymmetryClassType)
from TFENN.util.random import (uniformDihedralR2, uniformO2, uniformO3,
                               uniformOctahedralR3)

jax.config.update("jax_enable_x64", True)


def double_contraction(c_tensor, e_tensor):
    return jnp.einsum("...ijkl,...ij", c_tensor, e_tensor)


def contraction(c_tensor, e_tensor):
    return jnp.einsum("...ij,...j", c_tensor, e_tensor)


def rotate_full(e_tensor, rot_mat):
    return jnp.einsum("...ij,...jk,...lk", rot_mat, e_tensor, rot_mat)


def rotate_reduced(e_notation, e_tensor, rot_mat):
    return e_notation.to_reduced(rotate_full(e_notation.to_full(e_tensor), rot_mat))


class TestTensorSymmetryClass(unittest.TestCase):
    key = random.PRNGKey(0)
    n = 100

    def setUp(self):
        self.O2_mats = uniformO2(self.key, self.n)
        self.O3_mats = uniformO3(self.key, self.n)
        self.D2_mats = uniformDihedralR2(self.key, 4, min(self.n, 4), replace=False)
        self.Oh_mats = uniformOctahedralR3(self.key, min(self.n, 48), replace=False)

    def test_initialization(self):
        """Test that the initialization of the tensor symmetry classes perform as
        expected."""
        for sym_cls_type, dim in product(
            [
                TensorSymmetryClassType.ISOTROPIC,
                TensorSymmetryClassType.CUBIC,
            ],
            [2, 3],
        ):
            with self.subTest(sym_cls_type=sym_cls_type, dim=dim):
                sym_cls = sym_cls_type.create(order=4, dim=dim)

    def test_tensor_basis(self):
        """Test that the tensor basis is correctly constructed for the different
        notations."""
        for sym_cls_type, dim in product(
            [
                TensorSymmetryClassType.ISOTROPIC,
                TensorSymmetryClassType.CUBIC,
            ],
            [2, 3],
        ):
            voigt_tensor_helper = SymmetricTensorRepresentation(
                order=4,
                dim=dim,
                notation_type=SymmetricTensorNotationType.VOIGT,
                sym_cls_type=sym_cls_type,
            )
            for notation_type in SymmetricTensorNotationType:
                if notation_type != SymmetricTensorNotationType.VOIGT:
                    tensor_helper = SymmetricTensorRepresentation(
                        order=4,
                        dim=dim,
                        notation_type=notation_type,
                        sym_cls_type=sym_cls_type,
                    )
                    for k, (voigt_basis_tensor, notation_basis_tensor) in enumerate(
                        zip(
                            voigt_tensor_helper.tensor_basis.todense(),
                            tensor_helper.tensor_basis.todense(),
                        )
                    ):
                        with self.subTest(
                            notation_type=notation_type,
                            sym_cls_type=sym_cls_type,
                            dim=dim,
                            basis_tensor_id=k,
                            basis_tensor=voigt_basis_tensor,
                        ):
                            from_voigt_full_tensor = (
                                voigt_tensor_helper.notation.to_full(voigt_basis_tensor)
                            )
                            from_notation_full_tensor = tensor_helper.notation.to_full(
                                notation_basis_tensor
                            )
                            self.assertTrue(
                                jnp.allclose(
                                    from_voigt_full_tensor,
                                    from_notation_full_tensor,
                                    atol=1e-6,
                                    rtol=1e-6,
                                ),
                                msg=(
                                    f"voigt_basis_tensor:\n{voigt_basis_tensor} -->\n"
                                    f"{from_voigt_full_tensor}\n"
                                    "notation_basis_tensor:\n"
                                    f"{notation_basis_tensor} -->\n"
                                    f"{from_notation_full_tensor}"
                                ),
                            )

    def test_isotropic_symmetry(self):
        """Test that the isotropic symmetry is respected."""
        for notation_type, dim in product(
            [
                SymmetricTensorNotationType.VOIGT,
                SymmetricTensorNotationType.MANDEL,
            ],
            [2, 3],
        ):
            e_tensor_notation = notation_type.create(dim=dim, order=2)
            rotate_reduced_jit = jax.jit(
                lambda et, rm: rotate_reduced(e_tensor_notation, et, rm)
            )

            c_tensor_helper = SymmetricTensorRepresentation(
                order=4,
                dim=dim,
                notation_type=notation_type,
                sym_cls_type=TensorSymmetryClassType.ISOTROPIC,
            )

            contract_jit = jax.jit(
                lambda ct, et: contraction(ct, et)
                if notation_type == SymmetricTensorNotationType.MANDEL
                else e_tensor_notation.to_reduced(
                    double_contraction(
                        c_tensor_helper.notation.to_full(ct),
                        e_tensor_notation.to_full(et),
                    )
                )
            )

            O_mats = self.O2_mats if dim == 2 else self.O3_mats

            # First test all basis tensors individually
            for k, c_tensor in enumerate(c_tensor_helper.tensor_basis.todense()):
                with self.subTest(
                    notation_type=notation_type,
                    dim=dim,
                    basis_tensor_id=k,
                    basis_tensor=c_tensor,
                ):
                    self.key, subkey = random.split(self.key)
                    e_tensors = random.normal(
                        subkey,
                        (self.n,) + e_tensor_notation.reduced_shape,
                    )

                    res1 = contract_jit(
                        c_tensor,
                        rotate_reduced_jit(e_tensors, O_mats),
                    )
                    res2 = rotate_reduced_jit(
                        contract_jit(c_tensor, e_tensors),
                        O_mats,
                    )

                    self.assertTrue(jnp.allclose(res1, res2, atol=1e-3))

            # Then with random tensors
            with self.subTest(notation_type=notation_type, dim=dim):
                self.key, c_key, e_key = random.split(self.key, 3)
                c_tensor_params = random.normal(
                    c_key, (self.n, c_tensor_helper.sym_cls.basis_size)
                )
                c_tensors = c_tensor_helper.params_to_tensors(c_tensor_params)
                e_tensors = random.normal(
                    e_key, (self.n,) + e_tensor_notation.reduced_shape
                )

                res1 = contract_jit(
                    c_tensors,
                    rotate_reduced_jit(e_tensors, O_mats),
                )
                res2 = rotate_reduced_jit(contract_jit(c_tensors, e_tensors), O_mats)

                self.assertTrue(jnp.allclose(res1, res2, atol=1e-3))

    def test_cubic_symmetry(self):
        """Test that the cubic symmetry is respected."""
        for notation_type, dim in product(
            [
                SymmetricTensorNotationType.VOIGT,
                SymmetricTensorNotationType.MANDEL,
            ],
            [2, 3],
        ):
            e_tensor_notation = notation_type.create(dim=dim, order=2)
            rotate_reduced_jit = jax.jit(
                lambda et, rm: rotate_reduced(e_tensor_notation, et, rm)
            )

            c_tensor_helper = SymmetricTensorRepresentation(
                order=4,
                dim=dim,
                notation_type=notation_type,
                sym_cls_type=TensorSymmetryClassType.CUBIC,
            )

            contract_jit = jax.jit(
                lambda ct, et: contraction(ct, et)
                if notation_type == SymmetricTensorNotationType.MANDEL
                else e_tensor_notation.to_reduced(
                    double_contraction(
                        c_tensor_helper.notation.to_full(ct),
                        e_tensor_notation.to_full(et),
                    )
                )
            )

            O_mats = self.D2_mats if dim == 2 else self.Oh_mats

            # First test all basis tensors individually
            for k, c_tensor in enumerate(c_tensor_helper.tensor_basis.todense()):
                with self.subTest(
                    notation_type=notation_type,
                    dim=dim,
                    basis_tensor_id=k,
                    basis_tensor=c_tensor,
                ):
                    self.key, subkey = random.split(self.key)
                    e_tensors = random.normal(
                        subkey, (self.n,) + e_tensor_notation.reduced_shape
                    )

                    res1 = contract_jit(
                        c_tensor[None, None, ...],
                        rotate_reduced_jit(e_tensors[None, ...], O_mats[:, None, ...]),
                    )
                    res2 = rotate_reduced_jit(
                        contract_jit(c_tensor[None, None, ...], e_tensors[None, ...]),
                        O_mats[:, None, ...],
                    )

                    self.assertTrue(jnp.allclose(res1, res2, atol=1e-3))

            # Then with random tensors
            with self.subTest(notation_type=notation_type, dim=dim):
                self.key, c_key, e_key = random.split(self.key, 3)
                c_tensor_params = random.normal(
                    c_key, (self.n, c_tensor_helper.sym_cls.basis_size)
                )
                c_tensors = c_tensor_helper.params_to_tensors(c_tensor_params)
                e_tensors = random.normal(
                    e_key, (self.n,) + e_tensor_notation.reduced_shape
                )

                res1 = contract_jit(
                    c_tensors[None, None, ...],
                    rotate_reduced_jit(e_tensors[None, ...], O_mats[:, None, ...]),
                )
                res2 = rotate_reduced_jit(
                    contract_jit(c_tensors[None, None, ...], e_tensors[None, ...]),
                    O_mats[:, None, ...],
                )

                self.assertTrue(jnp.allclose(res1, res2, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
