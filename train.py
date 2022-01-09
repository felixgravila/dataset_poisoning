import os
import random
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from absl import app, flags
from tensorflow.keras.datasets import cifar10, cifar100, fashion_mnist, mnist


def get_dataset(dataset_name: str):
    dsn = dataset_name.lower()
    if dsn == "mnist":
        ds = mnist
    elif dsn == "fashion_mnist":
        ds = fashion_mnist
    elif dsn == "cifar10":
        ds = cifar10
    elif dsn == "cifar100":
        ds = cifar100

    (train_X, train_y), (val_X, val_y) = ds.load_data()
    train_X = train_X / 255
    val_X = val_X / 255

    if train_X.shape[-1] != 3:
        train_X = tf.expand_dims(train_X, -1)
        val_X = tf.expand_dims(val_X, -1)

    train_y = tf.squeeze(train_y)
    val_y = tf.squeeze(val_y)

    return (train_X, train_y), (val_X, val_y)


def get_wrong_label(y: int, population: List[int]):
    a, b = random.sample(population, k=2)
    if y != a:
        return a
    return b


def poison(y: List[int], population: List[int], percentage: int, biased: bool):
    """
    Poison `percentage` of dataset to be wrong
    y: the label list
    population: the label set, y \in population
    percentage: [0..100] how much of the dataset to poison
    """
    cutoff = round(len(y) * (1 - (percentage / 100)))
    to_mod = y[cutoff:]

    if biased:
        # only pick one of the first two classes
        # simulating a systematic labelling error
        population = population[:2]

    wrong_labels = [get_wrong_label(yy, population) for yy in to_mod]

    return np.concatenate([y[:cutoff], wrong_labels]).astype(np.int8)


def make_simple_conv_model(input_shape, num_classes):
    model = tf.keras.Sequential(
        [
            L.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            L.Conv2D(32, (3, 3), activation="relu"),
            L.MaxPool2D(),
            L.Conv2D(32, (3, 3), activation="relu"),
            L.Conv2D(32, (3, 3), activation="relu"),
            L.MaxPool2D(),
            L.Conv2D(32, (3, 3), activation="relu"),
            L.Conv2D(32, (3, 3), activation="relu", padding="same"),
            L.Flatten(),
            L.Dense(128, activation="relu"),
            L.Dense(128, activation="relu"),
            L.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def train_model(train_X, train_y, val_X, val_y, num_classes, save_dir, logs_dir):
    model = make_simple_conv_model(
        input_shape=train_X.shape[1:], num_classes=num_classes
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_dir, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.1, patience=5
        ),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
    ]

    train_result = model.fit(
        train_X,
        train_y,
        validation_data=(val_X, val_y),
        epochs=100,
        batch_size=256,
        validation_batch_size=256,
        callbacks=callbacks,
    )

    return train_result


def main(argv):

    ds = FLAGS.dataset.lower()

    MODEL_ROOT_DIR = f"models/{('b_' if FLAGS.biased else '')}{ds}"
    LOGS_ROOT_DIR = f"logs/{('b_' if FLAGS.biased else '')}{ds}"

    if not os.path.isdir(MODEL_ROOT_DIR):
        os.makedirs(MODEL_ROOT_DIR)
    if not os.path.isdir(LOGS_ROOT_DIR):
        os.makedirs(LOGS_ROOT_DIR)

    (train_X, train_y), (val_X, val_y) = get_dataset(ds)
    label_population = list(np.unique(train_y))

    poison_percs = map(int, FLAGS.percents)

    for poison_perc in poison_percs:

        modeldir = os.path.join(MODEL_ROOT_DIR, str(poison_perc))

        if os.path.isdir(modeldir) and not FLAGS.overwrite:
            print(f"{poison_perc}% exists, skipping...")
            continue

        print(f"Training for {poison_perc}%")

        poison_y = poison(train_y, label_population, poison_perc, FLAGS.biased)
        train_model(
            train_X,
            poison_y,
            val_X,
            val_y,
            num_classes=len(label_population),
            save_dir=modeldir,
            logs_dir=os.path.join(LOGS_ROOT_DIR, str(poison_perc)),
        )


FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "mnist", "Dataset to use.")
flags.DEFINE_list(
    "percents",
    [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100],
    "Percents to poison.",
)
flags.DEFINE_bool(
    "biased", False, "Have biased poisoning or perfectly random distribution."
)
flags.DEFINE_bool("overwrite", False, "Overwrite models.")

if __name__ == "__main__":
    app.run(main)
