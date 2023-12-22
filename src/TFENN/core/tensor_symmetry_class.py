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

import abc
import dataclasses
import logging

from jax import numpy as jnp
from jax.experimental import sparse

from TFENN.util.enum_input import EnumInputClass


class TensorSymmetryClassType(EnumInputClass):
    """Enum for the symmetry classes of tensors."""

    ISOTROPIC = "isotropic"
    CUBIC = "cubic"
    HEXAGONAL = "hexagonal"
    RHOMBOHEDRAL = "rhombohedral"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"
    NONE = None

    @classmethod
    @property
    def obj_map(
        cls,
    ) -> dict["TensorSymmetryClassType", "TensorSymmetryClass"]:
        return {
            cls.ISOTROPIC: IsotropicSymmetryClass,
            cls.CUBIC: CubicSymmetryClass,
            cls.HEXAGONAL: HexagonalSymmetryClass,
            cls.RHOMBOHEDRAL: RhombohedralSymmetryClass,
            cls.TETRAGONAL: TetragonalSymmetryClass,
            cls.ORTHORHOMBIC: OrthorhombicSymmetryClass,
            cls.MONOCLINIC: MonoclinicSymmetryClass,
            cls.TRICLINIC: TriclinicSymmetryClass,
            cls.NONE: NoSymmetryClass,
        }


@dataclasses.dataclass(frozen=True)
class TensorSymmetryClass(abc.ABC):
    """Base class for the symmetry classes of a symmetric tensor."""

    dim: int = 3
    order: int = 4
    logger: logging.Logger = dataclasses.field(
        default_factory=lambda: logging.getLogger(__name__)
    )

    def __post_init__(self):
        if self.order not in [2, 4]:
            raise ValueError("Only order 2 and 4 tensors are supported.")
        if self.dim not in [2, 3]:
            raise ValueError("Only dimensions 2 and 3 are supported.")

    @property
    @abc.abstractmethod
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the full notation of a tensor for this symmetry
        class as a sparse array where the first axis is the number of basis elements and
        the remaining axes are of the size of the tensor in Mandel notation.
        :return: the basis of the full notation
        :rtype: sparse.BCOO
        """
        pass

    @property
    def basis_size(self) -> int:
        """Return the number of parameters needed to describe a tensor in this symmetry
        class.
        :return: the number of parameters needed
        :rtype: int
        """
        return self.mandel_tensor_basis.shape[0]


@dataclasses.dataclass(frozen=True)
class IsotropicSymmetryClass(TensorSymmetryClass):
    @property
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the space of tensors equivariant to this symmetry class
        in Mandel notation.
        :return: the basis of the space of tensors equivariant to this symmetry class
        :rtype: sparse.BCOO
        """
        if self.order == 2:
            if self.dim == 2:
                return sparse.BCOO(
                    (jnp.array([1, 1]), jnp.array([[0, 0], [0, 1]])), shape=(1, 3)
                )
            elif self.dim == 3:
                return sparse.BCOO(
                    (jnp.array([1, 1, 1]), jnp.array([[0, 0], [0, 1], [0, 2]])),
                    shape=(1, 6),
                )
            else:
                raise NotImplementedError
        elif self.order == 4:
            if self.dim == 2:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 5 + (-1,)),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [0, 1, 1],
                                [0, 2, 2],
                                [1, 0, 1],
                                [1, 1, 0],
                                [1, 2, 2],
                            ]
                        ),
                    ),
                    shape=(2, 3, 3),
                )

            elif self.dim == 3:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 12 + (-1,) * 3),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [0, 1, 1],
                                [0, 2, 2],
                                [0, 3, 3],
                                [0, 4, 4],
                                [0, 5, 5],
                                [1, 0, 1],
                                [1, 0, 2],
                                [1, 1, 0],
                                [1, 1, 2],
                                [1, 2, 0],
                                [1, 2, 1],
                                [1, 3, 3],
                                [1, 4, 4],
                                [1, 5, 5],
                            ]
                        ),
                    ),
                    shape=(2, 6, 6),
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class CubicSymmetryClass(TensorSymmetryClass):
    @property
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the Mandel notation notation for the cubic symmetry.
        :return: the basis of the Mandel notation notation
        :rtype: sparse.BCOO
        """
        if self.order == 2:
            return IsotropicSymmetryClass(
                dim=self.dim, order=self.order
            ).mandel_tensor_basis
        elif self.order == 4:
            if self.dim == 2:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 5),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0],
                                [2, 2, 2],
                            ]
                        ),
                    ),
                    shape=(3, 3, 3),
                )
            elif self.dim == 3:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 12),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [0, 1, 1],
                                [0, 2, 2],
                                [1, 0, 1],
                                [1, 0, 2],
                                [1, 1, 0],
                                [1, 1, 2],
                                [1, 2, 0],
                                [1, 2, 1],
                                [2, 3, 3],
                                [2, 4, 4],
                                [2, 5, 5],
                            ]
                        ),
                    ),
                    shape=(3, 6, 6),
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class HexagonalSymmetryClass(TensorSymmetryClass):
    @property
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the Mandel notation for the hexagonal symmetry.
        :return: the basis of the Mandel notation
        :rtype: sparse.BCOO
        """
        if self.dim == 2:
            return CubicSymmetryClass(dim=2, order=4).mandel_tensor_basis
        elif self.dim == 3:
            if order == 2:
                raise NotImplementedError
            elif order == 4:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 12 + (-1,)),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0],
                                [2, 2, 2],
                                [3, 0, 2],
                                [3, 1, 2],
                                [3, 2, 0],
                                [3, 2, 1],
                                [4, 3, 3],
                                [5, 4, 4],
                                [0, 5, 5],
                                [1, 5, 5],
                            ],
                        ),
                    ),
                    shape=(6, 6, 6),
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class RhombohedralSymmetryClass(TensorSymmetryClass):
    pass


@dataclasses.dataclass(frozen=True)
class TetragonalSymmetryClass(TensorSymmetryClass):
    @property
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the Mandel notation for the tetragonal symmetry.
        :return: the basis of the Mandel notation
        :rtype: sparse.BCOO
        """
        if self.dim == 2:
            return CubicSymmetryClass(dim=2, order=4).mandel_tensor_basis
        elif self.dim == 3:
            if order == 2:
                raise NotImplementedError
            elif order == 4:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 14 + (-1,) * 2),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0],
                                [2, 2, 2],
                                [3, 0, 2],
                                [3, 1, 2],
                                [3, 2, 0],
                                [3, 2, 1],
                                [4, 3, 3],
                                [4, 4, 4],
                                [5, 5, 5],
                                [6, 0, 5],
                                [6, 5, 0],
                                [6, 1, 5],
                                [6, 5, 1],
                            ],
                        ),
                    ),
                    shape=(7, 6, 6),
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class OrthorhombicSymmetryClass(TensorSymmetryClass):
    @property
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the Mandel notation for the orthorhombic symmetry.
        :return: the basis of the Mandel notation
        :rtype: sparse.BCOO
        """
        if self.order == 2:
            if self.dim == 2:
                return sparse.BCOO(
                    (
                        jnp.array([1, 1]),
                        jnp.array([[0, 0], [1, 1]]),
                    ),
                    shape=(2, 3),
                )
            else:
                raise NotImplementedError
        elif self.order == 4:
            if self.dim == 2:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 5),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [1, 1, 1],
                                [2, 0, 1],
                                [2, 1, 0],
                                [3, 2, 2],
                            ],
                        ),
                    ),
                    shape=(4, 3, 3),
                )

            elif self.dim == 3:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 12),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [1, 1, 1],
                                [2, 0, 1],
                                [2, 1, 0],
                                [3, 2, 2],
                                [4, 0, 2],
                                [4, 2, 0],
                                [5, 1, 2],
                                [5, 2, 1],
                                [6, 3, 3],
                                [7, 4, 4],
                                [8, 5, 5],
                            ],
                        ),
                    ),
                    shape=(9, 6, 6),
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class MonoclinicSymmetryClass(TensorSymmetryClass):
    @property
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the Mandel notation for the monoclinic symmetry.
        :return: the basis of the Mandel notation
        :rtype: sparse.BCOO
        """
        if self.dim == 2:
            return TriclinicSymmetryClass(dim=2, order=4).mandel_tensor_basis

        elif self.dim == 3:
            if self.order == 2:
                raise NotImplementedError
            elif self.order == 4:
                return sparse.BCOO(
                    (
                        jnp.array((1,) * 20),
                        jnp.array(
                            [
                                [0, 0, 0],
                                [1, 1, 1],
                                [2, 0, 1],
                                [2, 1, 0],
                                [3, 2, 2],
                                [4, 0, 2],
                                [4, 2, 0],
                                [5, 1, 2],
                                [5, 2, 1],
                                [6, 3, 3],
                                [7, 4, 4],
                                [8, 3, 4],
                                [8, 4, 3],
                                [9, 5, 5],
                                [10, 0, 5],
                                [10, 5, 0],
                                [11, 1, 5],
                                [11, 5, 1],
                                [12, 2, 5],
                                [12, 5, 2],
                            ],
                        ),
                    ),
                    shape=(13, 6, 6),
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class TriclinicSymmetryClass(TensorSymmetryClass):
    @property
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the Mandel notation for the triclinic symmetry.
        :return: the basis of the Mandel notation
        :rtype: sparse.BCOO
        """
        n = 3 if self.dim == 2 else 6

        if self.order == 2:
            return sparse.BCOO(
                (jnp.array((1,) * n), jnp.array([[i, i] for i in range(n)])),
                shape=(n, n),
            )
        elif self.order == 4:
            return sparse.BCOO(
                (
                    jnp.array((1,) * (n**2)),
                    jnp.array(
                        [[i, i, i] for i in range(n)]
                        + [
                            [n + i * (i + 1) // 2 + j, i, j]
                            for i in range(n)
                            for j in range(i + 1, n)
                        ]
                        + [
                            [n + i * (i + 1) // 2 + j, j, i]
                            for i in range(n)
                            for j in range(i + 1, n)
                        ]
                    ),
                ),
                shape=(n * (n + 1) // 2, n, n),
            )
        else:
            raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class NoSymmetryClass(TensorSymmetryClass):
    @property
    def mandel_tensor_basis(self) -> sparse.BCOO:
        """Return the basis of the Mandel notation for no assumed symmetry.
        :return: the basis of the Mandel notation
        :rtype: sparse.BCOO
        """
        n = 3 if self.dim == 2 else 6

        if self.order == 2:
            return sparse.BCOO(
                (jnp.array((1,) * n), jnp.array([[i, i] for i in range(n)])),
                shape=(n, n),
            )
        elif self.order == 4:
            return sparse.BCOO(
                (
                    jnp.array((1,) * (n**2)),
                    jnp.array([[i * n + j, i, j] for i in range(n) for j in range(n)]),
                ),
                shape=(n * n, n, n),
            )
        else:
            raise NotImplementedError
