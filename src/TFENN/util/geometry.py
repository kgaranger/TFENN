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

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike


def cross_product_mat(v: ArrayLike) -> Array:
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def angle_to_rot_mat_2d(angle: float) -> Array:
    return jnp.array(
        [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
    )


def axis_angle_to_rot_mat_3d(axis: ArrayLike, angle: float) -> Array:
    axis = axis / jnp.linalg.norm(axis)
    return (
        jnp.cos(angle) * jnp.eye(3)
        + jnp.sin(angle) * cross_product_mat(axis)
        + (1 - jnp.cos(angle)) * jnp.outer(axis, axis)
    )


def quat_to_rot_mat_3d(quat: ArrayLike) -> Array:
    return jnp.array(
        [
            [
                1 - 2 * quat[2] ** 2 - 2 * quat[3] ** 2,
                2 * quat[1] * quat[2] - 2 * quat[0] * quat[3],
                2 * quat[1] * quat[3] + 2 * quat[0] * quat[2],
            ],
            [
                2 * quat[1] * quat[2] + 2 * quat[0] * quat[3],
                1 - 2 * quat[1] ** 2 - 2 * quat[3] ** 2,
                2 * quat[2] * quat[3] - 2 * quat[0] * quat[1],
            ],
            [
                2 * quat[1] * quat[3] - 2 * quat[0] * quat[2],
                2 * quat[2] * quat[3] + 2 * quat[0] * quat[1],
                1 - 2 * quat[1] ** 2 - 2 * quat[2] ** 2,
            ],
        ]
    )
