from ._base import write_notebook
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
    ("md", """
# Chapter 12 - Generative Adversarial Networks (GANs)

GAN = two networks in a duel:
- **Generator**: takes random noise and produces fake images.
- **Discriminator**: tries to tell real from fake.

They play an adversarial game: the generator keeps improving until the discriminator can no longer tell its outputs from real. The result is a *generative model of the training distribution* — new images that match the training set.

The classic recipe (Goodfellow 2014) on MNIST.
"""),
    ("code", """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras import layers, models

print("Keras", keras.__version__, "| TF", tf.__version__)
"""),
    ("md", "## 1. Data — MNIST, scaled & channel-last"),
    ("code", """
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_train = x_train[..., None] * 2.0 - 1.0    # add noise for stability
x_train = x_train + np.random.normal(0, 0.02, x_train.shape)
x_train = np.clip(x_train, -1.0, 1.0)
print("train", x_train.shape)
"""),
    ("md", "## 2. Latent noise dim + helper to show fakes"),
    ("code", """
latent_dim = 64

def show_fakes(imgs, title=""):
    imgs = imgs * 0.5 + 0.5
    fig, axes = plt.subplots(1, 8, figsize=(14, 3))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, :, :, 0], cmap="gray"); ax.axis("off")
    plt.suptitle(title); plt.show()
"""),
    ("md", "## 3. Generator: noise -> upsampled 28x28 image"),
    ("code", """
generator = models.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(7*7*128, activation="relu"),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu"),
    layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu"),
    layers.Conv2D(1, 3, padding="same", activation="tanh"),
], name="generator")
generator.summary()
"""),
    ("md", "## 4. Discriminator: binary real/fake"),
    ("code", """
discriminator = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, 3, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2D(128, 3, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid"),
], name="discriminator")
discriminator.summary()
"""),
    ("md", "## 5. Compile each: discriminator gets real + fake labels"),
    ("code", """
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4), loss="binary_crossentropy", metrics=["accuracy"])
discriminator.trainable = False   # freeze while training generator via GAN

gan = models.Sequential([generator, discriminator])
gan.compile(optimizer=keras.optimizers.Adam(2e-4), loss="binary_crossentropy")
"""),
    ("md", """
## 6. The training loop
We cap `steps_per_epoch` so the notebook stays fast on CPU; raising it / epochs produces sharper digits.
"""),
    ("code", """
epochs, batch_size = 10, 128
steps_per_epoch = min(x_train.shape[0] // batch_size, 200)
gen_imgs = []

for epoch in range(epochs):
    for _ in range(steps_per_epoch):
        idx = np.random.randint(len(x_train), size=batch_size)
        real_imgs = x_train[idx]

        noise = np.random.normal(size=(batch_size, latent_dim))
        fake = generator.predict(noise, verbose=0)

        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_imgs, labels_real)
        d_loss_fake = discriminator.train_on_batch(fake, labels_fake)

        g_noise = np.random.normal(size=(batch_size, latent_dim))
        g_labels = np.ones((batch_size, 1))     # trick: want generator to fool D
        g_loss = gan.train_on_batch(g_noise, g_labels)

    noise = np.random.normal(size=(8, latent_dim))
    show_fakes(generator.predict(noise, verbose=0), title=f"epoch {epoch+1}")
print("done")
"""),
    ("md", """
## 7. Look at the final distribution — clean fakes?
Run a fresh ballot: does the discriminator still believe in the generator's outputs at 50/50?
"""),
    ("code", """
noise = np.random.normal(size=(8, latent_dim))
fakes = generator.predict(noise, verbose=0)
show_fakes(fakes, title="final generated samples")
"""),
    ("md", """
## Recap
- Generator ↔ Discriminator: minimax game, each pushing the other to improve.
- On MNIST even a from-scratch MLP works; on real photos, use DCGAN/WGAN techniques from the *Generative Deep Learning* book.
- Same pattern (latent + decoder) appears in VAEs; the difference: GAN optimizes "can you fool the critic", VAE optimizes "can you recreate".
"""),
]

def build():
    write_notebook(ROOT / "12-Generative-Adversarial-Networks" / "12-Generative-Adversarial-Networks.ipynb", CELLS, "Generative Adversarial Networks")
    return "12"