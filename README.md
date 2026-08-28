# TensorFlow Hands-On: Machine Learning & Deep Learning

A complete, reproducible, **from-scratch** journey into machine learning and deep learning using **TensorFlow 2 + Keras 3**.

Every notebook is written from scratch, designed to run end-to-end on CPU (verified), and structured as a progressive curriculum:

| # | Chapter | Topic | Notebook |
|---|---------|-------|----------|
| 1 | Neural Network | First feed-forward net on MNIST | [01-Neural-Network](01-Neural-Network/README.md) |
| 2 | Binary Classification | IMDB sentiment (sigmoid) | [02-Binary-Classification](02-Binary-Classification/README.md) |
| 3 | Multi-Class Classification | Reuters topics (softmax) | [03-Multi-Class-Classification](03-Multi-Class-Classification/README.md) |
| 4 | Regression | House price prediction (from-scratch data) | [04-Regression-House-Prices](04-Regression-House-Prices/README.md) |
| 5 | Overfitting & Underfitting | Diagnose & fix both failure modes | [05-Overfitting-Underfitting](05-Overfitting-Underfitting/README.md) |
| 6 | Convolutional Neural Networks | Feature maps on Fashion-MNIST | [06-Convolutional-Neural-Networks](06-Convolutional-Neural-Networks/README.md) |
| 7 | One-Hot Encoding & Embeddings | Text to numbers | [07-Embeddings-One-Hot](07-Embeddings-One-Hot/README.md) |
| 8 | Recurrent Neural Networks | Time-series + LSTM text | [08-Recurrent-Neural-Networks](08-Recurrent-Neural-Networks/README.md) |
| 9 | Text Generation with LSTM | Char-level GPT-ish sampling | [09-Text-Generation-LSTM](09-Text-Generation-LSTM/README.md) |
| 10 | Deep Dream | Visualize CNN features | [10-Deep-Dream](10-Deep-Dream/README.md) |
| 11 | Variational Autoencoders | Generative models | [11-Variational-Autoencoder](11-Variational-Autoencoder/README.md) |
| 12 | Generative Adversarial Networks | Generator vs Discriminator | [12-Generative-Adversarial-Networks](12-Generative-Adversarial-Networks/README.md) |

## Why "from scratch"?
- No opaque black boxes: every network is assembled layer-by-layer and explained.
- No huge downloads for chapter 4 and 10 (data generated with `numpy`).
- All chapters verified executing on CPU with `tensorflow-cpu`.

## Quick Start

```bash
git clone https://github.com/SURUJ404/Tensorflow.git
cd Tensorflow
pip install -r requirements.txt
jupyter lab
# or run every notebook non-interactively:
python tools/run_all.py
```

## Contributing & License
Improvements, fixed notebooks, and new chapter ideas are welcome.
This project is open-source (added `LICENSE`).