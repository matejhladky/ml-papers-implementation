# 001-gradient-based-learning-applied-to-document-recognition

This directory contains an authentic, well-documented implementation of the LeNet-5 model as described in the paper [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) by LeCun et al. (1998).

## Content
- **Paper Analysis:** A summary of the key components and experimental setup as presented in the paper, focusing on the LeNet-5 architecture.
- **Dataset & Preprocessing:** Code to prepare the MNIST dataset (emulating the “regular” database) using custom normalization and padding to replicate the original input specifications.
- **Model Architecture:** An authentic PyTorch implementation of LeNet-5 including:
  - Custom layers (e.g., scaled tanh activation, trainable subsampling, and an RBF output layer)
  - Weight initialization mimicking the original uniform distribution approach
- **Training & Evaluation:** 
  - Train/Test pipeline
  - Loss/Accuracy analysis
  - Confusion matrix visualization and analysis of misclassified examples
- **Discussion & Future Work:** Comparative analysis with the original paper’s results, a discussion of simplifications made (e.g., using SGD instead of SDLM), and suggestions for further improvements.