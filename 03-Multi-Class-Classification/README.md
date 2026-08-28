# Chapter 3 - Multi-Class Classification

## What you'll build
A network that assigns Reuters newswires to one of 46 topic categories.

## Concepts
- From binary to N classes: output layer of N units + `softmax`
- Loss = `categorical_crossentropy`
- One-hot label encoding with `to_categorical`
- Predictions are *probability distributions*, argmax picks the class

## Files
- `03-Multi-Class-Classification.ipynb`

## Expected result
~78-82% test accuracy (46 classes; random baseline ~2%).

[Back to repo](../README.md)