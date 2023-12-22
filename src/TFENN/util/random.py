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

from typing import Any, Sequence

from jax import Array
from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike

from .array_util import canonicalize_tuple
from .geometry import axis_angle_to_rot_mat_3d

Shape = Sequence[int]

KeyArrayLike = ArrayLike
RealArray = Array


def chiralOctahedralR3() -> RealArray:
    fourfold_axis_180_rot = jnp.array(
        [
            axis_angle_to_rot_mat_3d(jnp.array([1, 0, 0]), jnp.pi),
            axis_angle_to_rot_mat_3d(jnp.array([0, 1, 0]), jnp.pi),
            axis_angle_to_rot_mat_3d(jnp.array([0, 0, 1]), jnp.pi),
        ]
    )
    threefold_axis_120_rot = jnp.array(
        [
            axis_angle_to_rot_mat_3d(jnp.array([1, 1, 1]), 2 * jnp.pi / 3),
            axis_angle_to_rot_mat_3d(jnp.array([1, -1, 1]), 2 * jnp.pi / 3),
            axis_angle_to_rot_mat_3d(jnp.array([-1, 1, 1]), 2 * jnp.pi / 3),
            axis_angle_to_rot_mat_3d(jnp.array([-1, -1, 1]), 2 * jnp.pi / 3),
            axis_angle_to_rot_mat_3d(jnp.array([1, 1, -1]), 2 * jnp.pi / 3),
            axis_angle_to_rot_mat_3d(jnp.array([1, -1, -1]), 2 * jnp.pi / 3),
            axis_angle_to_rot_mat_3d(jnp.array([-1, 1, -1]), 2 * jnp.pi / 3),
            axis_angle_to_rot_mat_3d(jnp.array([-1, -1, -1]), 2 * jnp.pi / 3),
        ]
    )
    twofold_axis_180_rot = jnp.array(
        [
            axis_angle_to_rot_mat_3d(jnp.array([1, 1, 0]), jnp.pi),
            axis_angle_to_rot_mat_3d(jnp.array([1, -1, 0]), jnp.pi),
            axis_angle_to_rot_mat_3d(jnp.array([1, 0, 1]), jnp.pi),
            axis_angle_to_rot_mat_3d(jnp.array([1, 0, -1]), jnp.pi),
            axis_angle_to_rot_mat_3d(jnp.array([0, 1, 1]), jnp.pi),
            axis_angle_to_rot_mat_3d(jnp.array([0, 1, -1]), jnp.pi),
        ]
    )
    fourfold_axis_90_rot = jnp.array(
        [
            axis_angle_to_rot_mat_3d(jnp.array([1, 0, 0]), jnp.pi / 2),
            axis_angle_to_rot_mat_3d(jnp.array([0, 1, 0]), jnp.pi / 2),
            axis_angle_to_rot_mat_3d(jnp.array([0, 0, 1]), jnp.pi / 2),
            axis_angle_to_rot_mat_3d(jnp.array([-1, 0, 0]), jnp.pi / 2),
            axis_angle_to_rot_mat_3d(jnp.array([0, -1, 0]), jnp.pi / 2),
            axis_angle_to_rot_mat_3d(jnp.array([0, 0, -1]), jnp.pi / 2),
        ]
    )
    return jnp.concatenate(
        [
            jnp.eye(3)[None, ...],
            fourfold_axis_180_rot,
            threefold_axis_120_rot,
            twofold_axis_180_rot,
            fourfold_axis_90_rot,
        ],
        axis=0,
    )


def uniformSO2(key: KeyArrayLike, shape: Shape = 1, dtype: Any = float) -> RealArray:
    """Sample uniform matrices in SO2.
    :param key: random
    :type key: KeyArrayLike
    :param shape: shape of the array of matrices, defaults to 1. Output shape is
    (shape, 2, 2)
    :type shape: Shape, optional
    :param dtype: dtype of the output array, defaults to float
    :type dtype: Any, optional
    :return: random matrices in SO2
    :rtype: RealArray
    """

    shape = canonicalize_tuple(shape)
    angs = random.uniform(key, shape=shape, minval=-jnp.pi, maxval=jnp.pi)[
        ..., None, None
    ]
    return jnp.stack(
        [jnp.cos(angs), -jnp.sin(angs), jnp.sin(angs), jnp.cos(angs)], axis=-1
    ).reshape(shape + (2, 2))


def uniformO2(key: KeyArrayLike, shape: Shape = 1, dtype: Any = float) -> RealArray:
    """Sample uniform matrices in O2.
    :param key: random key
    :type key: KeyArrayLike
    :param shape: shape of the array of matrices, defaults to 1. Output shape is
    (shape, 2, 2)
    :type shape: Shape, optional
    :param dtype: dtype of the output array, defaults to float
    :type dtype: Any, optional
    :return: random matrices in O2
    :rtype: RealArray
    """

    shape = canonicalize_tuple(shape)
    so2_key, dets_key = random.split(key)
    so2 = uniformSO2(so2_key, shape=shape, dtype=dtype)
    neg_dets = random.choice(dets_key, jnp.array([True, False]), shape=shape)
    return so2.at[neg_dets, 0, :].set(so2[neg_dets, 0, :] * -1)


def uniformSO3(key: KeyArrayLike, shape: Shape = 1, dtype: Any = float) -> RealArray:
    """Sample uniform matrices in SO3.
    :param key: random key
    :type key: KeyArrayLike
    :param shape: shape of the array of matrices, defaults to 1. Output shape is
    (shape, 3, 3)
    :type shape: Shape, optional
    :param dtype: dtype of the output array, defaults to float
    :type dtype: Any, optional
    :return: random matrices in O3
    :rtype: RealArray
    """

    shape = canonicalize_tuple(shape)
    v1s_key, angs_key = random.split(key)

    v1s = random.multivariate_normal(
        v1s_key, mean=jnp.zeros(3), cov=jnp.eye(3), shape=shape
    )
    angs = random.uniform(angs_key, shape=shape, minval=-jnp.pi, maxval=jnp.pi)[
        ..., None
    ]
    v1s = v1s / jnp.linalg.norm(v1s, axis=-1)[..., None]
    v1s_x = jnp.cross(v1s, jnp.array([1, 0, 0]))
    v1s_x = v1s_x / jnp.linalg.norm(v1s_x, axis=-1)[..., None]
    v1s_y = jnp.cross(v1s, v1s_x)
    v2s = v1s_x * jnp.cos(angs) + v1s_y * jnp.sin(angs)
    v3s = -v1s_x * jnp.sin(angs) + v1s_y * jnp.cos(angs)
    return jnp.stack([v1s, v2s, v3s], axis=-1)


def uniformO3(key: KeyArrayLike, shape: Shape = 1, dtype: Any = float) -> RealArray:
    """Sample uniform matrices in O3.
    :param key: random key
    :type key: KeyArrayLike
    :param shape: shape of the array of matrices, defaults to 1. Output shape is
    (shape, 3, 3)
    :type shape: Shape, optional
    :param dtype: dtype of the output array, defaults to float
    :type dtype: Any, optional
    :return: random matrices in O3
    :rtype: RealArray
    """

    shape = canonicalize_tuple(shape)
    so3_key, dets_key = random.split(key)
    so3 = uniformSO3(so3_key, shape=shape, dtype=dtype)
    neg_dets = random.choice(dets_key, jnp.array([True, False]), shape=shape)
    return so3.at[neg_dets, 0, :].set(so3[neg_dets, 0, :] * -1)


def uniformCyclicR2(
    key: KeyArrayLike,
    n: int,
    shape: Shape = 1,
    replace: bool = True,
    dtype: Any = float,
) -> RealArray:
    """Sample matrices of SO(2) from the representation of the cyclic group C_n.
    :param key: random key
    :type key: KeyArrayLike
    :param n: order of the cyclic group
    :type n: int
    :param shape: shape of the array of matrices, defaults to 1. Output shape is
    (shape, 2, 2)
    :type shape: Shape, optional
    :param replace: whether to sample with replacement, defaults to True
    :type replace: bool, optional
    :param dtype: dtype of the output array, defaults to float
    :type dtype: Any, optional
    :return: random matrices in SO(2)
    :rtype: RealArray
    """

    shape = canonicalize_tuple(shape)
    angs = random.choice(key, n, shape=shape, replace=replace) * 2 * jnp.pi / n
    return jnp.stack(
        [
            jnp.cos(angs),
            -jnp.sin(angs),
            jnp.sin(angs),
            jnp.cos(angs),
        ],
        axis=-1,
    ).reshape(shape + (2, 2))


def uniformDihedralR2(
    key: KeyArrayLike,
    n: int,
    shape: Shape = 1,
    replace: bool = True,
    dtype: Any = float,
) -> RealArray:
    """Sample matrices of O(2) from the representation of the dihedral group D_n.
    :param key: random key
    :type key: KeyArrayLike
    :param n: order of the dihedral group
    :type n: int
    :param shape: shape of the array of matrices, defaults to 1. Output shape is
    (shape, 2, 2)
    :type shape: Shape, optional
    :param replace: whether to sample with replacement, defaults to True
    :type replace: bool, optional
    :param dtype: dtype of the output array, defaults to float
    :type dtype: Any, optional
    :return: random matrices in O(2)
    :rtype: RealArray
    """

    shape = canonicalize_tuple(shape)
    angs_dets = random.choice(key, 2 * n, shape=shape, replace=replace)
    angs = (angs_dets // 2) * 2 * jnp.pi / n
    dets = (angs_dets % 2) * 2 - 1
    return jnp.stack(
        [
            jnp.cos(angs) * dets,
            -jnp.sin(angs) * dets,
            jnp.sin(angs),
            jnp.cos(angs),
        ],
        axis=-1,
    ).reshape(shape + (2, 2))


def uniformChiralOctahedralR3(
    key: KeyArrayLike, shape: Shape = 1, replace: bool = True
) -> RealArray:
    """Sample matrices of SO(3) from the representation of the chiral octahedral group.
    :param key: random
    :type key: KeyArrayLike
    :param shape: shape of the array of matrices, defaults to 1. Output shape is
    (shape, 3, 3)
    :type shape: Shape, optional
    :param replace: whether to sample with replacement, defaults to True
    :type replace: bool, optional
    :param dtype: dtype of the output array, defaults to float
    :type dtype: Any, optional
    :return: random matrices in O(3)
    :rtype: RealArray
    """

    shape = canonicalize_tuple(shape)
    choices = random.choice(key, 24, shape=shape, replace=replace)
    return chiralOctahedralR3()[choices]


def uniformOctahedralR3(
    key: KeyArrayLike, shape: Shape = 1, replace: bool = True
) -> RealArray:
    """Sample matrices of O(3) from the representation of the octahedral group.
    :param key: random
    :type key: KeyArrayLike
    :param shape: shape of the array of matrices, defaults to 1. Output shape is
    (shape, 3, 3)
    :type shape: Shape, optional
    :param replace: whether to sample with replacement, defaults to True
    :type replace: bool, optional
    :param dtype: dtype of the output array, defaults to float
    :type dtype: Any, optional
    :return: random matrices in O(3)
    :rtype: RealArray
    """

    shape = canonicalize_tuple(shape)
    choices_negdets = random.choice(key, 48, shape=shape, replace=replace)
    choices = choices_negdets // 2
    neg_dets = choices_negdets % 2

    return (
        chiralOctahedralR3()[choices]
        .at[neg_dets, 0, :]
        .set(chiralOctahedralR3()[choices][neg_dets, 0, :] * -1)
    )
