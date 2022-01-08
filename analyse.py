import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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


def main(argv):

    datasets = [ds.lower() for ds in FLAGS.datasets]
    results = {}

    for ds in datasets:

        print(f"Working on {ds}...")

        MODEL_ROOT_DIR = os.path.join("models", ds)

        (_, _), (val_X, val_y) = get_dataset(ds)

        models = [
            (int(f), os.path.join(MODEL_ROOT_DIR, f))
            for f in os.listdir(MODEL_ROOT_DIR)
        ]
        models.sort(key=lambda x: x[0])

        percs, accs = [], []

        for perc, modelpath in models:
            model = tf.keras.models.load_model(modelpath)
            accuracy = model.evaluate(val_X, val_y, batch_size=256)[1]
            percs.append(perc)
            accs.append(accuracy)

        results[ds] = {"percs": percs, "accs": accs}

    plt.figure(figsize=(20, 10), facecolor="white")
    for k, v in results.items():
        plt.plot(v["percs"], [a * 100 for a in v["accs"]], label=k)
    plt.title(f"Effect of poisoning various datasets")
    plt.xlabel("Percent poisoned")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(f"{('_'.join(datasets))}.png")


FLAGS = flags.FLAGS
flags.DEFINE_list(
    "datasets", ["mnist", "fashion_mnist", "cifar10", "cifar100"], "Datasets to use"
)

if __name__ == "__main__":
    app.run(main)
