from ._base import write_notebook
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
    ("md", """
# Chapter 7 - One-Hot Encoding & Word Embeddings

Machines can't read "dog", "cat", "love". We must convert discrete, categorical data into numbers. This chapter explains the two dominant strategies:

1. **One-hot encoding**: a vector with a single 1 among many 0s per category.
2. **Learned embeddings**: a small dense numeric vector per word *learned* by the network during training.

Both are fundamental; embeddings are what power modern NLP.
"""),
    ("code", """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, models

print("Keras", keras.__version__)
"""),
    ("md", """
## Part 1 - One-hot by hand (the crisp, the clear, the naive)
For 4 categories (red, green, blue, yellow), index them and create a 4-dim one-hot vector.
"""),
    ("code", """
categories = ["red", "green", "blue", "yellow"]
word_to_index = {w: i for i, w in enumerate(categories)}
samples = ["red green blue blue yellow", "blue green"]

def one_hot_text(text, vocab, vocab_size):
    vectors = np.zeros((len(text.split()), vocab_size))
    for i, w in enumerate(text.split()):
        vectors[i, vocab[w]] = 1.0
    return vectors

v = one_hot_text("red green blue blue", word_to_index, len(categories))
print("one-hot for 'red green blue blue':")
print(v)
print("column = category presence (each time the word is present it 1's its column)")
"""),
    ("code", """
from keras.utils import to_categorical

labels = ["red", "blue", "green", "yellow", "yellow"]
label_ids = np.array([word_to_index[l] for l in labels])

onehot = keras.ops.one_hot(label_ids, len(categories))
print("keras.ops.one_hot(labels) shape:", onehot.shape)
print(onehot.numpy())
"""),
    ("code", """
onehot2 = to_categorical(label_ids, len(categories))
print("keras.utils.to_categorical(labels) shape:", onehot2.shape)
print(onehot2)
"""),
    ("md", """
## Part 2 - Learned embeddings (the modern, dense way)
One-hot vectors are sparse (mostly zeros).
Embeddings replace them with a **learned dense vector** per word. In Keras this is the `Embedding(vocab_size, embedding_dim)` layer: index -> vector, and gradients flow back during training.
"""),
    ("code", """
word_index = ["hello", "world", "deep", "learning", "is", "great"]
pad = 0
seq = np.array([[1, 2, 3, 4, 5, 6]])  # stopwords hidden for clarity

embed = layers.Embedding(input_dim=len(word_index)+1, output_dim=8, input_length=6)
inp = layers.Input(shape=(6,))
out = embed(inp)
em = models.Model(inp, out)
vector = em.predict(seq)
print("embedding output for the 6 words -> shape", vector.shape)
print(vector[0])
print("each row is a dense 8-dim vector; similar words will end up close in this space")
"""),
    ("md", """
## Embeddings as features: a mini sentiment classifier
Train an embedding-backed classifier on IMDB -- the baseline that the RNN chapter then beats with sequence modelling.
"""),
    ("code", """
from keras.datasets import imdb
from keras.utils import pad_sequences

vocab_size = 10000
maxlen = 100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print("shapes", x_train.shape, x_test.shape)
"""),
    ("code", """
model = models.Sequential([
    layers.Input(shape=(maxlen,)),
    layers.Embedding(vocab_size, 8),
    layers.GlobalAveragePooling1D(),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
"""),
    ("code", """
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=10, batch_size=256, verbose=1)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"embedding baseline accuracy: {acc:.4f}")
"""),
    ("md", """
## Recap
- **One-hot**: fast to build, sparse, huge, no semantic similarity.
- **Embeddings**: dense, trained for the task, capture meaning.
- Next step: stack an RNN on top of embeddings (chapter 8).
"""),
]

def build():
    write_notebook(ROOT / "07-Embeddings-One-Hot" / "07-Embeddings-One-Hot.ipynb", CELLS, "One-Hot & Embeddings")
    return "07"