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


from pathlib import Path

import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp

from TFENN.core import (DenseSymmetricTensor, MandelNotation,
                        SymmetricTensorNotationType,
                        SymmetricTensorRepresentation, TensorActivation,
                        TensorSymmetryClassType)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

# dataset_path = Path("data/neohookean_train.csv")  # Download link: http://tinyurl.com/2np8b8zm
dim = 3
reduced_dim = dim * (dim + 1) // 2
dataset_in_cols = (0, 1, 2, 3, 4, 5)
dataset_out_cols = (6, 7, 8, 9, 10, 11)
epochs = 1000
split = 0.8
batch_size = 32

notation_type = (
    SymmetricTensorNotationType.MANDEL
)  # Notation type used internally by the network to represent tensors symmetric as lower order tensors
symmetry_class = (
    TensorSymmetryClassType.ISOTROPIC
)  # Symmetry class enforced by the network

kernel_rep = SymmetricTensorRepresentation(
    order=4,
    dim=dim,
    notation_type=notation_type,
    sym_cls_type=symmetry_class,
)
bias_rep = SymmetricTensorRepresentation(
    order=2,
    dim=dim,
    notation_type=notation_type,
    sym_cls_type=symmetry_class,
)
feature_notation = MandelNotation(dim=dim, order=2)


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):  # type: ignore
        for _ in range(2):
            x = DenseSymmetricTensor(
                kernel_rep=kernel_rep,
                bias_rep=bias_rep,
                features=64,
                kernel_init=nn.initializers.he_normal(),
            )(x)
            x = TensorActivation(nn.leaky_relu, feature_notation)(x)
        x = DenseSymmetricTensor(
            kernel_rep=kernel_rep,
            bias_rep=bias_rep,
            features=1,
            kernel_init=nn.initializers.he_normal(),
        )(x)
        return x


def load_csv_dataset(
    path: Path, input_cols: tuple[int, ...], output_cols: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = data[:, input_cols].reshape((-1, 1, reduced_dim))
    y = data[:, output_cols].reshape((-1, 1, reduced_dim))

    # Voigt notation to Mandel notation
    x[:, 0, 3:] *= np.sqrt(2)
    y[:, 0, 3:] *= np.sqrt(2)

    # Scaling
    x_diag_mean = np.mean(x[:, 0, :3])
    y_diag_mean = np.mean(y[:, 0, :3])
    x[:, 0, :3] -= x_diag_mean
    y[:, 0, :3] -= y_diag_mean
    x_std = np.std(x)
    y_std = np.std(y)

    x /= x_std
    y /= y_std

    return x, y


def main():
    key = jax.random.PRNGKey(0)
    net_key, data_key = jax.random.split(key)
    net = Network()
    x = np.ones(
        (batch_size, 1, reduced_dim)
    )  # batch_size samples, 1 feature, dim*(dim+1)/2 components (dimxdim symmetric tensor)
    ts = train_state.TrainState.create(
        apply_fn=net.apply,
        tx=optax.adam(1e-3),
        params=net.init(key, x),
    )

    x, y = load_csv_dataset(dataset_path, dataset_in_cols, dataset_out_cols)
    shuffled_indices = jax.random.permutation(data_key, x.shape[0])
    x, y = x[shuffled_indices, ...], y[shuffled_indices, ...]

    split_index = int(split * x.shape[0])
    x_train, x_val = x[:split_index, ...], x[split_index:, ...]
    y_train, y_val = y[:split_index, ...], y[split_index:, ...]

    grad_fn = jax.grad(lambda params, x, y: jnp.mean((ts.apply_fn(params, x) - y) ** 2))
    eval_fn = jax.jit(lambda params, x, y: jnp.mean((ts.apply_fn(params, x) - y) ** 2))
    update_ts_fn = jax.jit(
        lambda ts, x, y: ts.apply_gradients(grads=grad_fn(ts.params, x, y))
    )

    for k in range(epochs):  # Training loop
        data_key, shuffle_key = jax.random.split(data_key)
        train_shuffled_indices = jax.random.permutation(shuffle_key, x_train.shape[0])
        for i in range(0, x_train.shape[0] - batch_size, batch_size):
            batch_indices = train_shuffled_indices[i : i + batch_size]
            batch_x, batch_y = x_train[batch_indices, ...], y_train[batch_indices, ...]

            ts = update_ts_fn(ts, batch_x, batch_y)

        train_loss = eval_fn(ts.params, x_train, y_train)
        val_loss = eval_fn(ts.params, x_val, y_val)
        print(f"Epoch {k}: Train loss: {train_loss}, Val loss: {val_loss}")


if __name__ == "__main__":
    main()
