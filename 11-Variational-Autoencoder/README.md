# Chapter 11 - Variational Autoencoders (VAEs)

## What you'll build
A VAE that learns a smooth latent space of MNIST digits and can *generate new digits*.

## Concepts
- Autoencoder bottleneck -> latent representation
- Encoder predicts mean + log-variance (a distribution, not a point)
- Reparameterization trick (sampling made differentiable)
- KL divergence regularization toward N(0,1)
- Latent-space interpolation (morph between two digits)

## Files
- `11-Variational-Autoencoder.ipynb`

[Back to repo](../README.md)