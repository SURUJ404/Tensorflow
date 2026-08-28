from ._base import write_notebook
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
    ("md", """
# Chapter 6 - Convolutional Neural Networks

A `Dense` layer sees the image as one long, flat vector. A **convolutional** layer instead slides a small learnable filter across the image, preserving the *spatial layout* of features. That's why CNNs are the default for images.

## The conv recipe
1. `Conv2D(k, (3,3), activation="relu")` — k filters of size 3x3 slide over the image.
2. `MaxPooling2D((2,2))` — halves the image, keeps the strongest response, adds translation robustness & frees compute.
3. `Flatten` -> `Dense` header -> `softmax`.

We'll classify the **Fashion-MNIST** dataset (10 clothing classes) - a harder "MNIST-like" benchmark that proves a CNN beats a plain Dense net.
"""),
    ("code", """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import fashion_mnist
from keras import layers, models

print("Keras", keras.__version__)
"""),
    ("md", "## Load & prepare (channel-last batch, normalized)"),
    ("code", """
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., np.newaxis].astype("float32") / 255.0
test_images = test_images[..., np.newaxis].astype("float32") / 255.0

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

print("train", train_images.shape, "test", test_images.shape)
"""),
    ("md", "## Step 1 - a small CNN from scratch"),
    ("code", """
cnn = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.summary()
"""),
    ("code", """
history = cnn.fit(train_images, train_labels,
                  validation_data=(test_images, test_labels),
                  epochs=10, batch_size=128, verbose=1)
"""),
    ("md", "## Score vs the chapter-1 Dense baseline"),
    ("code", """
test_loss, test_acc = cnn.evaluate(test_images, test_labels, verbose=0)
print(f"CNN acc: {test_acc:.4f}   (Dense chapter-1 baseline was ~0.98, plain Dense on FashionMNIST ~0.90)")
"""),
    ("md", "## Look at the 20+ feature maps a layer learns"),
    ("code", """
def show_conv_feature_maps(model, image):
    # cut the model at the first conv output (Keras 3: predict returns a plain array)
    first_conv = model.layers[1]
    act_model = models.Model(inputs=model.layers[0].input, outputs=first_conv.output)

    acts = act_model.predict(image[None])       # (1, 26, 26, 32)
    maps = acts[0]                              # (26, 26, 32)
    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(maps[..., i], cmap="viridis")
        ax.axis("off")
    plt.suptitle("Feature maps after first Conv2D (32 filters)"); plt.show()

show_conv_feature_maps(cnn, test_images[0])
"""),
    ("md", """
## Why it works
- 3x3 filters see small local patches (tiny structures: edges, corners, strokes).
- Early convs learn low-level features, later convs recombine them into higher-level parts (sleeve, sole, zipper).
- Pooling drops spatial resolution while increasing the field-of-view -> localization + compression.
- Dropout + batch-norm (next) keep it from memorizing pixels.
"""),
    ("code", """
# quick visual: a few real samples with their predictions
preds = cnn.predict(test_images[:12]).argmax(axis=1)
fig, axes = plt.subplots(1, 12, figsize=(16, 4))
for i, ax in enumerate(axes):
    ax.imshow(test_images[i, :, :, 0], cmap="gray")
    ax.set_title(f"T{test_labels[i]}/P{preds[i]}", fontsize=8)
    ax.axis("off")
plt.show()
"""),
    ("md", """
## Recap
- `Conv2D` learns *shared* filters (small, efficient) instead of one weight per pixel.
- Add depth + pooling -> more expressive, more robust.
- On images, a CNN blows away a Dense net of the same parameter budget (~+7pp on FashionMNIST).
"""),
]

def build():
    write_notebook(ROOT / "06-Convolutional-Neural-Networks" / "06-Convolutional-Neural-Networks.ipynb", CELLS, "Convolutional Neural Networks")
    return "06"