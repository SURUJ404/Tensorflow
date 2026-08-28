from ._base import write_notebook
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
    ("md", """
# Chapter 3 - Multi-Class Classification

Binary classification says *yes/no*. Multi-class says *which one of k classes?*.

We'll predict the topic of Reuters newswires: 8,982 training / 2,246 test articles, each assigned to one of 46 mutually exclusive topics (classes).
"""),
    ("md", """
## The three-phase recipe
1. **Vectorize** text into a 10,000-dim bag of words.
2. **One-hot** the integer labels into a 46-dim binary vector (or use sparse categorical cross-entropy with integer labels).
3. Build a network ending in a 46-unit `softmax` layer.
"""),
    ("code", """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import reuters
from keras import layers, models

print("Keras", keras.__version__)
"""),
    ("md", """
## Load & prepare (with sensible caps)
`num_words=10000` truncates vocabulary. We also one-hot the labels the manual way first to *see* what it does (use `to_categorical` for storage later).
"""),
    ("code", """
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print("train", len(train_data), "test", len(test_data), "classes", len(set(train_labels)))
"""),
    ("code", """
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        # histogram-like: count word presence
        results[i, seq] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print("x_train", x_train.shape)
"""),
    ("code", """
from keras.utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
print("one-hot label vector for first sample:", y_train[0], "shape", y_train.shape)
"""),
    ("md", """
## The model: 64 -> 64 -> softmax(46)
A wide hidden layer (64 units always more than 46 classes) so the network has room to spread information before the final softmax.
"""),
    ("code", """
model = models.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax"),
])
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
"""),
    ("code", """
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=15, batch_size=512, verbose=1,
)
"""),
    ("md", "## Evaluate on the test set and inspect per-class quality"),
    ("code", """
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}  (46 classes -> random is {1/46:.3f})")
"""),
    ("md", """
## Classification: predictions are *distributions*, not answers
`predict` on one sample gives 46 probabilities; argmax tells the class, but the shape of the distribution tells how confident the model is.
"""),
    ("code", """
pred = model.predict(x_train[0:1])
print("pred.shape", pred.shape)
top3 = np.argsort(pred[0])[::-1][:3]
print("top-3 predicted class ids:", top3)
print("true class:", train_labels[0])
"""),
    ("md", """
## Recap
- Multi-class = `softmax` over k outputs, `categorical_crossentropy`.
- One-hot labels are standard; keras's `to_categorical` does it for integer labels.
- Must choose layer sizes >= number of classes to carry enough information.
"""),
]

def build():
    write_notebook(ROOT / "03-Multi-Class-Classification" / "03-Multi-Class-Classification.ipynb", CELLS, "Multi-Class Classification")
    return "03"