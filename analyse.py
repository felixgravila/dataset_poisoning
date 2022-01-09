import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
from absl import app, flags
from tensorflow.keras.datasets import cifar10, cifar100, fashion_mnist, mnist

avail_colours = [v for (_, v) in mcolors.TABLEAU_COLORS.items()]


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

    datasets = [(ds.lower(), ds.lower()) for ds in FLAGS.datasets]
    b_datasets = [(ds.lower(), f"b_{ds.lower()}") for ds in FLAGS.datasets]
    if FLAGS.all:
        datasets.extend(b_datasets)
    elif FLAGS.biased:
        datasets = b_datasets

    results = {}

    for (ds, model_root) in datasets:
        print(f"Working on{(' biased' if model_root.startswith('b_') else '')} {ds}...")

        MODEL_ROOT_DIR = os.path.join("models", model_root)

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

        results[model_root] = {"percs": percs, "accs": accs}

    used_colours = {}

    plt.figure(figsize=(20, 10), facecolor="white")
    for k, v in results.items():
        base_ds = k.replace("b_", "")
        if base_ds not in used_colours:
            used_colours[base_ds] = avail_colours.pop()
        col = used_colours[base_ds]
        plt.plot(
            v["percs"],
            [a * 100 for a in v["accs"]],
            ("--" if "b_" in k else "-"),
            label=k,
            c=col,
        )
    plt.title(f"Effect of poisoning various datasets")
    plt.xlabel("Percent poisoned")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{('_'.join([d[1].replace('_','') for d in datasets]))}.png")


FLAGS = flags.FLAGS
flags.DEFINE_list(
    "datasets", ["mnist", "fashion_mnist", "cifar10", "cifar100"], "Datasets to use"
)
flags.DEFINE_bool(
    "biased", False, "Use biased poisoning or perfectly random distribution."
)
flags.DEFINE_bool("all", False, "Use both random and biased poisoning.")

if __name__ == "__main__":
    app.run(main)
