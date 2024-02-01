# Copyright 2024 Kévin Garanger
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

from functools import partial
from typing import NamedTuple

from jax import Array, jit
from jax import numpy as jnp


class EighResult(NamedTuple):
    """Copied from jax.numpy.linalg"""

    eigenvalues: Array
    eigenvectors: Array


@partial(jit, static_argnames=("UPLO", "symmetrize_input", "add_epsilon"))
def _eigh_2x2(
    C: Array,
    UPLO: str | None = None,
    symmetrize_input: bool = True,
    add_epsilon: bool = False,
) -> EighResult:
    """Eigenvalues and eigenvectors of a 2x2 hermitian matrix.
    For some reason, this version seems to be more robsut to nan errors than the
    original jax version.
    Based on https://hal.science/hal-01501221/document
    """
    a = jnp.real(C[..., 0, 0])
    b = jnp.real(C[..., 1, 1])
    if symmetrize_input:
        c = 0.5 * (jnp.conj(C[..., 0, 1]) + C[..., 1, 0])
    elif UPLO == "U":
        c = jnp.conj(C[..., 0, 1])
    else:
        c = C[..., 1, 0]

    if add_epsilon:
        deltas = jnp.sqrt(4 * abs(c) ** 2 + (a - b) ** 2 + 1e-16)
    else:
        deltas = jnp.sqrt(4 * abs(c) ** 2 + (a - b) ** 2)

    eigvals = jnp.stack(
        [
            0.5 * (a + b - deltas),
            0.5 * (a + b + deltas),
        ],
        axis=-1,
    )

    eigvecs = jnp.stack(
        [eigvals[...] - b[..., None], c[..., None] * jnp.ones_like(eigvals)],
        axis=-2,
    )

    if add_epsilon:
        alpha = 1e-16
        eigvecs_sn = jnp.sum(eigvecs**2, axis=-2, keepdims=True)
        eigvecs += jnp.eye(2) * (alpha / (alpha + eigvecs_sn))

    eigvecs = eigvecs / jnp.linalg.norm(eigvecs, axis=-2, keepdims=True)
    return EighResult(eigvals, eigvecs)


@partial(jit, static_argnames=("UPLO", "symmetrize_input", "add_epsilon"))
def eigh(
    a: Array,
    UPLO: str | None = None,
    symmetrize_input: bool = True,
    add_epsilon: bool = False,
) -> EighResult:
    """Eigenvalues and eigenvectors of a symmetric matrix."""

    if a.shape[-2:] == (2, 2):
        return _eigh_2x2(
            a, UPLO=UPLO, symmetrize_input=symmetrize_input, add_epsilon=add_epsilon
        )
    else:
        return jnp.linalg.eigh(a)
