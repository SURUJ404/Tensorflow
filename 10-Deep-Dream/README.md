# Chapter 10 - Deep Dream

## What it'll do
Amplify the patterns a CNN has learned by running **gradient ascent on the image** instead of the weights.

## Concepts
- Feature maps -> "what the network sees"
- Gradient ascent (maximize activations), not descent (minimize loss)
- Octave dreaming: refine detail across scales
- Same "gradient on input" trick powers adversarial examples

## Files
- `10-Deep-Dream.ipynb`

## Note
Trained CNN from scratch on synthetic blobs/stripes, so this runs fully offline.

[Back to repo](../README.md)