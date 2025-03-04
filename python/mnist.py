"""Identify mnist digits."""
import argparse
import struct
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from tqdm import tqdm
import pickle


def get_mnist_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Return the mnist test data set in numpy arrays.

    Returns:
        (array, array): A touple containing the test
        images and labels.
    """
    with open("/home/wolter/uni/kurs/day_12_exercise_cnn_solution/data/MNIST/t10k-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_test = data.reshape((size, nrows, ncols))

    with open("/home/wolter/uni/kurs/day_12_exercise_cnn_solution/data/MNIST/t10k-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_test = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    # if gpu:
    #    return cp.array(img_data_test), cp.array(lbl_data_test)
    return img_data_test, lbl_data_test


def get_mnist_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the mnist training data set.

    Returns:
        (array, array): A touple containing the training
        images and labels.
    """
    with open("/home/wolter/uni/kurs/day_12_exercise_cnn_solution/data/MNIST/train-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_train = data.reshape((size, nrows, ncols))

    with open("/home/wolter/uni/kurs/day_12_exercise_cnn_solution/data/MNIST/train-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    # if gpu:
    #    return cp.array(img_data_train), cp.array(lbl_data_train)
    return img_data_train, lbl_data_train


def normalize(
    data: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """Normalize the input array.

    After normalization the input
    distribution should be approximately standard normal.

    Args:
        data (np.array): The input array.
        mean (float): Data mean, re-computed if None.
            Defaults to None.
        std (float): Data standard deviation,
            re-computed if None. Defaults to None.

    Returns:
        np.array, float, float: Normalized data, mean and std.
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / std, mean, std


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        """Run the forward pass."""
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=2048)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return nn.sigmoid(x)


@jax.jit
def apply_model(state, images, labels):
    """Compute gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def cross_entropy(label: jnp.ndarray, out: jnp.ndarray) -> jnp.ndarray:
    """Compute the cross entropy of one-hot encoded labels and the network output.

    Args:
        label (jnp.ndarray): The image labels of shape [batch_size, 10].
        out (jnp.ndarray): The network output of shape [batch_size, 10].

    Returns:
        jnp.ndarray, The loss scalar.
    """
    left = -label * jnp.log(out + 1e-8)
    right = -(1 - label) * jnp.log(1 - out + 1e-8)
    return jnp.mean(left + right)


@jax.jit
def forward_step(variables, img_batch, label_batch):
    """Do a forward step."""
    out = cnn.apply(variables, jnp.expand_dims(img_batch, -1))
    ce_loss = cross_entropy(nn.one_hot(label_batch, num_classes=10), out)
    return ce_loss


# set up autograd
loss_grad_fn = jax.value_and_grad(forward_step)


# set up SGD
@jax.jit
def sgd_step(variables, grads, learning_rate):
    """Update the variable in a SGD step."""
    variables = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, variables, grads
    )
    return variables


def get_acc(img_data, label_data):
    """Compute the network accuracy."""
    out = cnn.apply(variables, jnp.expand_dims(img_data, -1))
    rec = jnp.argmax(out, axis=1)
    acc = jnp.sum((rec == label_data).astype(np.float32)) / len(label_data)
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Networks on MNIST.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    args = parser.parse_args()
    print(args)

    batch_size = 50
    val_size = 1000
    epochs = 10
    img_data_train, lbl_data_train = get_mnist_train_data()
    img_data_val, lbl_data_val = img_data_train[:val_size], lbl_data_train[:val_size]
    img_data_train, lbl_data_train = (
        img_data_train[val_size:],
        lbl_data_train[val_size:],
    )
    img_data_train, mean, std = normalize(img_data_train)
    img_data_val, _, _ = normalize(img_data_val, mean, std)

    exp_list = []

    for key in (0,):
        acc_list_train = []
        acc_list_val = []

        key = jax.random.PRNGKey(key)  # type: ignore
        cnn = CNN()
        variables = cnn.init(
            key, jnp.ones([batch_size] + list(img_data_train.shape[1:]) + [1])
        )

        for e in range(epochs):
            shuffler = jax.random.permutation(key, len(img_data_train))
            img_data_train = img_data_train[shuffler]
            lbl_data_train = lbl_data_train[shuffler]

            img_batches = np.split(
                img_data_train, img_data_train.shape[0] // batch_size, axis=0
            )
            label_batches = np.split(
                lbl_data_train, lbl_data_train.shape[0] // batch_size, axis=0
            )

            for img_batch, label_batch in tqdm(
                zip(img_batches, label_batches), total=len(img_batches)
            ):
                img_batch, label_batch = (
                    jnp.array(np_array) for np_array in (img_batch, label_batch)
                )
                # cel = cross_entropy(nn.one_hot(label_batches[no], num_classes=10),
                #                    out)
                cel, grads = loss_grad_fn(variables, img_batch, label_batch)
                variables = sgd_step(variables, grads, args.lr)
            print("Epoch: {}, loss: {}".format(e, cel))

            # train_acc = get_acc(img_data_train, lbl_data_train)
            val_acc = get_acc(img_data_val, lbl_data_val)
            print(
                "Validation accuracy: {:3.3f}".format(
                    val_acc
                )
            )
            # acc_list_train.append(train_acc)
            acc_list_val.append(val_acc)

        print("Training done. Testing...")
        img_data_test, lbl_data_test = get_mnist_test_data()
        img_data_test, mean, std = normalize(img_data_test, mean, std)

        test_acc = get_acc(img_data_test, lbl_data_test)
        print("Done. Test acc: {}".format(test_acc))
        exp_list.append((test_acc, acc_list_train, acc_list_val))

        pickle.dump(variables, open('network_dump.pkl', 'wb'))
    print('done')
