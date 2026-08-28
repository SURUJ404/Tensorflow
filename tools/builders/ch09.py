from ._base import write_notebook
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
    ("md", """
# Chapter 9 - Text Generation with LSTM

Networks can *generate* as well as classify. Here we train a character-level LSTM on a small corpus of text and let it invent new text character-by-character.

The trick: the model predicts the *probability of the next character* given all previous characters. We keep sampling (with a temperature knob for creativity) until the sequence is long enough.

We use a public-domain starter corpus so everything runs offline from scratch.
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
    ("md", "## 1. Corpus: build a small ordered text (public domain lines)"),
    ("code", """
corpus = (
    "In the beginning God created the heaven and the earth. "
    "And the earth was without form, and void; and darkness was upon the face of the deep. "
    "And the spirit of God moved upon the face of the waters. "
    "The Lord is my shepherd; I shall not want. "
    "He makes me lie down in green pastures; he leads me beside still waters. "
    "He restores my soul, and leads me in paths of righteousness. "
    "The light shines in the darkness and the darkness has not overcome it. "
    "Ask, and it will be given you. Seek, and you will find. Knock, and it will open. "
    "Every valley shall be raised and every mountain made low. "
    "A land flowing with milk and honey, a cloud by day and fire by night. "
)
print(f"corpus length: {len(corpus)} chars")
"""),
    ("code", """
chars = sorted(set(corpus))
char_index = {c: i for i, c in enumerate(chars)}
index_char = {i: c for c, i in char_index.items()}
n_chars = len(chars)
print(f"alphabet size: {n_chars}")
print("alphabet:", chars)
"""),
    ("md", "## 2. Build (input, target) character windows feed for training"),
    ("code", """
maxlen = 30
step = 3
sentences, next_chars = [], []
for i in range(0, len(corpus) - maxlen, step):
    sentences.append(corpus[i:i+maxlen])
    next_chars.append(corpus[i+maxlen])
print("windows:", len(sentences), "sample window: repr", repr(sentences[0]))
"""),
    ("code", """
from keras.utils import to_categorical

# input: one-hot (window len x vocab), target: one-hot of the next char
X = np.zeros((len(sentences), maxlen, n_chars), dtype=np.float32)
y = np.zeros((len(sentences), n_chars), dtype=np.float32)
for j, sentence in enumerate(sentences):
    for t, ch in enumerate(sentence):
        X[j, t, char_index[ch]] = 1.0
    y[j, char_index[next_chars[j]]] = 1.0
print("X", X.shape, "y", y.shape)
"""),
    ("md", "## 3. The generative LSTM: many units, softmax over alphabet"),
    ("code", """
model = models.Sequential([
    layers.Input(shape=(maxlen, n_chars)),
    layers.LSTM(128),
    layers.Dense(n_chars, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.summary()
"""),
    ("md", "## 4. Train (quick — small corpus)"),
    ("code", """
history = model.fit(X, y, epochs=150, batch_size=32, verbose=0)
plt.figure(figsize=(10, 3))
plt.plot(history.history["loss"]); plt.title("character-level LSTM loss"); plt.show()
"""),
    ("code", """
# (optional sanity check: the prediction output for one window is a distribution over chars)
preds = model.predict(X[:1], verbose=0)[0]
print("pred softmax sums to", round(float(preds.sum()), 4))
"""),
    ("md", """
## 5. Sampling function with temperature
Temperature `T`: low -> safer/greedier output, high -> more random, more "creative". The logits get divided by T before softmax.
"""),
    ("code", """
def sample_temperature(preds, temperature=0.5):
    preds = np.asarray(preds, dtype="float64").astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)[0]
    return int(np.argmax(probs))

def generate_text(model, seed, length=200, temperature=0.4):
    sentence = seed
    for _ in range(length):
        x = np.zeros((1, maxlen, n_chars))
        for t, ch in enumerate(sentence):
            x[0, t, char_index.get(ch, 0)] = 1.0
        preds = model.predict(x, verbose=0)[0]
        next_i = sample_temperature(preds, temperature)
        sentence = sentence[1:] + index_char[next_i]
    return sentence

gen = generate_text(model, sentences[0], 200, temperature=0.4)
print(gen)
"""),
    ("md", """
## 6. Same, but cooler and born from a different seed
"""),
    ("code", """
for T in (0.2, 0.4, 0.8):
    print(f"--- temperature {T} ---")
    print(generate_text(model, "the light ", length=150, temperature=T))
    print()
"""),
    ("md", """
## Recap
- A char-level LSTM predicts the distribution of the next char; sampling turns it into prose.
- `temperature` controls randomness/greediness.
- This toy corpus (small) won't produce amazing prose, but the *same pipeline* scales to books / Shakespeare.
- Next up in chapter 10 we flip from text to images.
"""),
]

def build():
    write_notebook(ROOT / "09-Text-Generation-LSTM" / "09-Text-Generation-LSTM.ipynb", CELLS, "Text Generation (LSTM)")
    return "09"