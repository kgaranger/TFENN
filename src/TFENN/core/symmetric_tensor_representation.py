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
from collections.abc import Sequence
from itertools import product
from math import prod

import jax
import numpy as np
from jax import numpy as jnp
from jax import typing as jnpt
from numpy import typing as npt

from TFENN.util.array_util import canonicalize_tuple, normalize_axes
from TFENN.util.enum_input import EnumInputClass

from .tensor_symmetry_class import TensorSymmetryClass, TensorSymmetryClassType


class SymmetricTensorNotationType(EnumInputClass):
    """Enum for the notations of a symmetric tensor."""

    FULL = "full"
    VOIGT = "voigt"
    MANDEL = "mandel"

    @classmethod
    @property
    def obj_map(
        cls,
    ) -> dict["SymmetricTensorNotationType", "SymmetricTensorNotation"]:
        return {
            cls.FULL: FullNotation,
            cls.VOIGT: VoigtNotation,
            cls.MANDEL: MandelNotation,
        }


@dataclasses.dataclass(frozen=True)
class SymmetricTensorNotation(abc.ABC):
    """Helper class to convert between full and reduced notations of tensors that verify
    minor symmetries.
    Given an order `n` tensor in `d` dimensions and a specific reduced notation,
    which is a tensor of order `m` in `d` dimensions, this class provides methods to
    convert between the two.
    For example, if tensor_notation is an instance of a derived class of
    SymmetricTensorNotation and if the scaling property attributes of the class
    are not None, then for a tensor M of appropriate shape, its reduced notation R
    is given by
    R = M[tensor_notation.to_reduced_indices] * tensor_notation.to_reduced_scaling
    and the full tensor is given by
    M = R[tensor_notation.to_full_indices] * tensor_notation.to_full_scaling
    """

    dim: int = 3
    order: int = 2
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        if self.order % 2 != 0:
            raise ValueError(f"Only even order tensors are supported, got {self.order}")
        if self.dim < 1:
            raise ValueError("dim must be positive")

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, order={self.order})"

    @property
    def full_shape(self) -> tuple[int, ...]:
        """Return the shape of the full tensor.
        :return: shape of the full tensor
        :rtype: tuple[int, ...]
        """
        return (self.dim,) * self.order

    @property
    @abc.abstractmethod
    def reduced_shape(self) -> tuple[int, ...]:
        """Return the shape of the reduced notation of the tensor.
        :return: shape of the reduced notation
        :rtype: tuple[int, ...]
        """
        pass

    @property
    @abc.abstractmethod
    def to_reduced_indices(self) -> tuple[npt.NDArray, ...]:
        """Return the indices of the full tensor shaped as the reduced notation.
        :return: a tuple of arrays with length equal to the number of axes of the
                 full notation. Each array is of the same shape as the reduced
                 notation and contains the indices of the corresponding axis of
                 the full notation
        :rtype: tuple[npt.NDArray, ...]
        """
        pass

    @property
    def to_reduced_scaling(self) -> npt.NDArray | float | None:
        """Return the scaling factor for the reduced notation of the tensor.
        :return: scaling factor for the reduced notation
        :rtype: npt.NDArray | float | None
        """
        return None

    @property
    @abc.abstractmethod
    def to_full_indices(self) -> tuple[npt.NDArray, ...]:
        """Return the indices of the reduced notation of the tensor shaped as the
        full tensor.
        :return: a tuple of arrays with length equal to the number of axes of the
                 reduced notation. Each array is of the same shape as the
                 full notation notation and contains the indices of the
                 corresponding axis of the reduced notation notation
        :rtype: tuple[npt.NDArray, ...]
        """
        pass

    @property
    def to_full_scaling(self) -> npt.NDArray | float | None:
        """Return the scaling factor for the full tensor.
        :return: scaling factor for the full tensor
        :rtype: npt.NDArray | float | None
        """
        return None

    def to_reduced(
        self, tensors: jnpt.ArrayLike, tensors_axis: int | Sequence[int] | None = None
    ) -> jax.Array:
        """Convert an array containing tensors in full notation to their reduced
        notations.
        :param tensors: tensors to be converted
        :type tensors: jnpt.ArrayLike
        :param tensors_axis: axes of the tensors to be converted, if None, the last axes
          are used, defaults to None
        :type tensors_axis: int or Sequence[int] or None
        :return: reduced notation of the tensors
        :rtype: jax.Array
        """
        if tensors_axis is None:
            tensors_axis = range(-self.order, 0)
        tensors_axis = canonicalize_tuple(tensors_axis)
        ndim = jnp.ndim(tensors)
        tensors_axis = normalize_axes(tensors_axis, ndim)
        indices = iter(self.to_reduced_indices)
        slices = tuple(
            next(indices) if i in tensors_axis else slice(None) for i in range(ndim)
        )
        non_tensors_axis = tuple(i for i in range(ndim) if i not in tensors_axis)
        if self.to_reduced_scaling is None:
            return tensors[slices]
        else:
            return tensors[slices] * jnp.expand_dims(
                self.to_reduced_scaling, non_tensors_axis
            )

    def to_full(
        self, tensors: jnpt.ArrayLike, tensors_axis: int | Sequence[int] | None = None
    ) -> jax.Array:
        """Convert an array containing tensors in reduced notation to ther full
        notation.
        :param tensors: tensors to be converted
        :type tensors: jnpt.ArrayLike
        :param tensors_axis: axes of the tensors to be converted, if None, the last axes
          are used, defaults to None
        :type tensors_axis: int or Sequence[int] or None
        :return: full notation of the tensors
        :rtype: jax.Array
        """
        if tensors_axis is None:
            tensors_axis = range(-len(self.reduced_shape), 0)
        tensors_axis = canonicalize_tuple(tensors_axis)
        ndim = jnp.ndim(tensors)
        tensors_axis = normalize_axes(tensors_axis, ndim)
        indices = iter(self.to_full_indices)
        slices = tuple(
            next(indices) if i in tensors_axis else slice(None) for i in range(ndim)
        )
        if self.to_full_scaling is None:
            return tensors[slices]
        else:
            batch_axis = tuple(i for i in range(ndim) if i not in tensors_axis)
            return tensors[slices] * jnp.expand_dims(self.to_full_scaling, batch_axis)


@dataclasses.dataclass(frozen=True)
class FullNotation(SymmetricTensorNotation):
    @property
    def reduced_shape(self) -> tuple[int, ...]:
        """Return the same shape as the full tensor.
        :return: shape of the reduced notation
        :rtype: tuple[int, ...]
        """
        return self.full_shape

    @property
    def to_reduced_indices(self) -> tuple[npt.NDArray, ...]:
        """Return the indices of the indices of the full tensor.
        full tensor.
        :return: a tuple of indices arrays
        :rtype: tuple[npt.NDArray, ...]
        """
        return tuple(np.indices(self.full_shape))

    @property
    def to_full_indices(self) -> tuple[npt.NDArray, ...]:
        """Return the indices of the indices of the full tensor.
        full tensor.
        :return: a tuple of indices arrays
        :rtype: tuple[npt.NDArray, ...]
        """
        return tuple(np.indices(self.full_shape))


@dataclasses.dataclass(frozen=True)
class VoigtNotation(SymmetricTensorNotation):
    """Helper class for symmetric tensors to convert between tensor and Voigt notation
    notations."""

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, order={self.order})"

    @property
    def voigt_dim(self) -> int:
        """Return the dimension of the Voigt notation notation of the tensor.
        :return: The dimension of the Voigt notation notation of the tensor.
        :rtype: int
        """
        return self.dim * (self.dim + 1) // 2

    @property
    def reduced_shape(self) -> tuple[int, ...]:
        """Return the shape of the Voigt notation notation of the tensor.
        :return: The shape of the Voigt notation notation of the tensor.
        :rtype: tuple[int, ...]
        """
        return (self.dim * (self.dim + 1) // 2,) * (self.order // 2)

    @property
    def tensor_indices_lists(self) -> tuple[list[int], ...]:
        """Return the tuple of indices lists of the tensor notation per axis,
        based on the Voigt notation notation.
        :return: The tuple of indices lists of the tensor notation per axis.
        :rtype: tuple[list[int], ...]
        """
        idx, vid = tuple(
            zip(
                *[
                    ((i, j), self.voigt_indices_matrix[i, j])
                    for i in range(self.dim)
                    for j in range(i, self.dim)
                ]
            )
        )
        i_s, j_s = tuple(zip(*idx))
        a_srt = np.argsort(np.array(vid))
        return (np.array(i_s)[a_srt], np.array(j_s)[a_srt])

    @property
    def voigt_indices_matrix(self) -> npt.NDArray:
        """Return the matrix of size (dim, dim) that contains the cornotationonding Voigt
        notation indices.
        :return: The matrix of Voigt notation indices.
        :rtype: npt.NDArray[int]
        """
        mat = np.empty((self.dim, self.dim), dtype=int)
        # Manually spiral through the matrix, maybe something more elegant can be done
        dir_cycle = ((1, 1), (-1, 0), (0, -1))
        dir_idx = 0
        i, j = -1, -1
        for k in range(self.dim):
            for l in range(self.dim - k):
                i += dir_cycle[dir_idx][0]
                j += dir_cycle[dir_idx][1]
                mat[i, j] = k * self.dim - k * (k - 1) // 2 + l
            dir_idx = (dir_idx + 1) % 3

        for i in range(1, self.dim):
            for j in range(i):
                mat[i, j] = mat[j, i]

        self.logger.debug(f"Voigt matrix:\n{mat}")
        return np.asarray(mat)

    @property
    def to_reduced_indices(self) -> tuple[npt.NDArray, ...]:
        """Return the indices of the full tensor shaped as the Voigt notation
        notation.
        :return: a tuple of arrays with length equal to the number of axes of the
                 full notation. Each array is of the same shape as the Voigt
                 notation notation and contains the indices of the corresponding
                 axis of the full notation
        :rtype: tuple[npt.NDArray, ...]
        """
        return tuple(
            map(
                lambda x: np.reshape(np.asarray(x), self.reduced_shape),
                zip(
                    *map(
                        lambda x: sum(x, ()),
                        product(
                            zip(*self.tensor_indices_lists), repeat=self.order // 2
                        ),
                    )
                ),
            )
        )

    @property
    def to_full_indices(self) -> tuple[npt.NDArray, ...]:
        """Return the indices of the Voigt notation notation of the tensor shaped
        as the full tensor.
        :return: a tuple of arrays with length equal to the number of axes of the Voigt
                 notation notation. Each array is of the same shape as the
                 full notation notation and contains the indices of the
                 corresponding axis of the Voigt notation notation
        :rtype: tuple[npt.NDArray, ...]
        """
        return tuple(
            map(
                lambda x: np.reshape(np.asarray(x), self.full_shape),
                zip(
                    *product(
                        self.voigt_indices_matrix.flatten(), repeat=self.order // 2
                    )
                ),
            )
        )


@dataclasses.dataclass(frozen=True)
class MandelNotation(VoigtNotation):
    """Helper class for symmetric tensors to convert between tensor and Mandel notation
    notations."""

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, order={self.order})"

    @property
    def reduced_scaling_list(self) -> list[float]:
        """Return the scaling factors for the Mandel notation notation of the
        tensor.
        :return: The scaling factors for the Mandel notation notation of the
        :rtype: list[float]
        """
        l = [1.0] * self.dim + [np.sqrt(2.0)] * (self.voigt_dim - self.dim)

        self.logger.debug(f"Mandel scaling list:\n{l}")
        return l

    @property
    def full_scaling_matrix(self) -> npt.NDArray:
        """
        Return the matrix of size (dim, dim) that contains the cornotationonding Mandel
        notation scaling factors.
        :return: The matrix of Voigt notation scaling factors.
        :rtype: npt.NDArray[float]
        """
        mat = np.empty((self.dim, self.dim), dtype=float)
        mat[np.triu_indices(self.dim, 1)] = 1 / np.sqrt(2.0)
        mat[np.diag_indices(self.dim)] = 1.0
        mat[np.tril_indices(self.dim, -1)] = 1 / np.sqrt(2.0)

        self.logger.debug(f"Mandel matrix:\n{mat}")
        return np.asarray(mat)

    @property
    def to_reduced_scaling(self) -> npt.NDArray:
        """Return the scaling factor for the Mandel notation notation.
        :return: scaling factor for the Mandel notation notation
        :rtype: npt.NDArray
        """
        return np.array(
            list(map(prod, product(self.reduced_scaling_list, repeat=self.order // 2)))
        ).reshape(self.reduced_shape)

    @property
    def to_full_scaling(self) -> npt.NDArray:
        """Return the scaling factor for the full tensor notation.
        :return: scaling factor for the full tensor notation
        :rtype: npt.NDArray
        """
        return np.array(
            list(
                map(
                    prod,
                    product(
                        self.full_scaling_matrix.flatten(),
                        repeat=self.order // 2,
                    ),
                )
            )
        ).reshape(self.full_shape)


@dataclasses.dataclass(frozen=True)
class SymmetricTensorRepresentation:
    dim: int = 3
    order: int = 2
    notation_type: dataclasses.InitVar[
        SymmetricTensorNotationType
    ] = SymmetricTensorNotationType.FULL
    sym_cls_type: dataclasses.InitVar[TensorSymmetryClassType | None] = None
    notation: SymmetricTensorNotation = dataclasses.field(init=False)
    sym_cls: TensorSymmetryClass = dataclasses.field(init=False)

    def __post_init__(self, notation_type, sym_cls_type):
        object.__setattr__(
            self,
            "notation",
            notation_type.create(dim=self.dim, order=self.order),
        )
        object.__setattr__(
            self,
            "sym_cls",
            sym_cls_type.create(dim=self.dim, order=self.order)
            if sym_cls_type is not None
            else None,
        )

    @property
    def basis_size(self) -> int:
        return self.sym_cls.basis_size

    @property
    def tensor_basis(self) -> jax.Array:
        if type(self.notation) == MandelNotation:
            return self.sym_cls.mandel_tensor_basis
        elif type(self.notation) == FullNotation:
            return MandelNotation(dim=self.dim, order=self.order).to_full(
                self.sym_cls.mandel_tensor_basis
            )
        else:
            return self.notation.to_reduced(
                MandelNotation(dim=self.dim, order=self.order).to_full(
                    self.sym_cls.mandel_tensor_basis
                )
            )

    def params_to_tensors(
        self,
        params: jnpt.ArrayLike,
        params_axis: int = -1,
        basis: jnpt.ArrayLike | None = None,
    ) -> jax.Array:
        """Return the tensor notations of the given array of parameters.
        :param params: the parameters to convert
        :type params: jnpt.ArrayLike
        :param params_axis: the axis of the parameters, defaults to -1
        :type params_axis: int, optional
        :param basis: the basis of tensors to use, defaults to None
        :type basis: jnpt.ArrayLike, optional
        :return: the tensor notation of the given parameters
        :rtype: jax.Array
        """
        ndim = jnp.ndim(params)
        (params_axis,) = normalize_axes((params_axis,), ndim)
        if basis is None:
            basis = self.tensor_basis.todense()
        return jnp.einsum(
            params,
            range(ndim),
            basis,
            [params_axis, Ellipsis],
            [k for k in range(ndim) if k != params_axis] + [Ellipsis],
        )

    def params_to_full_tensors(
        self,
        params: jnpt.ArrayLike,
        params_axis: int = -1,
        basis: jnpt.ArrayLike | None = None,
    ) -> jax.Array:
        """Return the tensor notations of the given array of parameters.
        :param params: the parameters to convert
        :type params: jnpt.ArrayLike
        :param params_axis: the axis of the parameters, defaults to -1
        :type params_axis: int, optional
        :param basis: the basis of tensors to use, defaults to None
        :type basis: jnpt.ArrayLike, optional
        :return: the tensor notation of the given parameters
        :rtype: jax.Array
        """
        return self.notation.to_full(self.params_to_tensors(params, params_axis, basis))

    def tensors_to_params(
        self,
        tensors: jnpt.ArrayLike,
        tensors_axis: tuple[int, ...] = None,
        basis: jnpt.ArrayLike | None = None,
    ) -> jax.Array:
        """Return the parameters of the given array of tensor notations.
        :param tensors: the tensor notations to convert
        :type tensors: jnpt.ArrayLike
        :param tensors_axis: the axes of the tensor notations, if None, the last axes
        are used, defaults to None
        :type tensors_axis: tuple[int, ...], optional
        :param basis: the basis of tensors to use, defaults to None
        :type basis: jnpt.ArrayLike, optional
        :return: the parameters of the given tensor notations
        :rtype: jax.Array
        """
        ndim = jnp.ndim(tensors)
        if tensors_axis is None:
            tensors_axis = tuple(range(-self.order // 2, 0))
        tensors_axis = normalize_axes(tensors_axis, ndim)
        if basis is None:
            basis = self.tensor_basis.todense()

        ts = jnp.reshape(
            jnp.moveaxis(tensors, tensors_axis, range(-self.order // 2, 0)),
            (-1, prod(basis.shape[1:])),
        )
        bs = jnp.reshape(basis, (basis.shape[0], -1))

        ps = jnp.matmul(ts, jnp.linalg.pinv(bs.astype(ts.dtype)))
        return jnp.swapaxes(
            jnp.reshape(
                ps,
                tuple(tensors.shape[k] for k in range(ndim) if k not in tensors_axis)
                + (ps.shape[-1],),
            ),
            -1,
            min(tensors_axis),
        )

    def full_tensors_to_params(
        self,
        tensors: jnpt.ArrayLike,
        tensors_axis: tuple[int, ...] | None = None,
        basis: jnpt.ArrayLike | None = None,
    ) -> jax.Array:
        """Return the parameters of the given array of full tensors.
        :param tensors: the full tensors to convert
        :type tensors: jnpt.ArrayLike
        :param tensors_axis: the axes of the full tensors, defaults to None
        :type tensors_axis: tuple[int, ...], optional
        :param basis: the basis of tensors to use, defaults to None
        :type basis: jnpt.ArrayLike, optional
        :return: the parameters of the given full tensors
        :rtype: jax.Array
        """
        return self.tensors_to_params(
            self.notation.to_reduced(tensors), tensors_axis, basis
        )
