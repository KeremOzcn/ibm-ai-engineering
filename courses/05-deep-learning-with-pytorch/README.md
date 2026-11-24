# 05 · Deep Learning with PyTorch

Multilayer networks, initialization, optimization, normalization, CNNs and transfer learning.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- Hidden layers build representations; nonlinearities control which functions a network can express.
- Initialization and normalization influence activation and gradient scales. Compare them under the same seed, data and training budget.
- Convolution shares local filters. Pooling and strided convolutions change spatial resolution and affect the receptive field.
- Dropout and batch normalization behave differently during training and evaluation. Save the chosen model state and evaluate with the correct mode.

## Working model

A 2D convolution output size per spatial axis is floor((input + 2·padding − dilation·(kernel − 1) − 1) / stride + 1).

## Start with these labs

- [Convolutional Neural Network MNIST](notebooks/convolutional-neural-network-mnist.ipynb) — CNN classification.
- [Batch Normalization with the MNIST Dataset](notebooks/batch-normalization-with-the-mnist-dataset.ipynb) — Batch normalization.
- [Activation Functions](notebooks/activation-functions.ipynb) — Nonlinear activations.

## Revisit with a purpose

- Calculate CNN tensor shapes before executing the MNIST notebook.
- Compare initialization choices with a fixed optimizer and plot both training and validation loss.
- Use the collected Fashion-MNIST project as a separate classification study; document the label mapping and split.

## Common mistakes

- Flattening the wrong dimensions mixes examples or changes the classifier input size.
- Selecting the best epoch on the test set contaminates the final estimate.
- A saved checkpoint is useful only if it can be reloaded into the matching architecture.

## Check your understanding

- How does a convolution reduce the number of learned parameters?
- When can dropout hurt a small model?
- How would you compare two initializations fairly?

## Collected material

28 reference notebooks. Course source, authorship and copyright notices remain in the collected files. No course notebook has been certified as runnable by this curation.

| Material | Status |
|---|---|
| [Activation Functions and Max Pooling](notebooks/activation-functions-and-max-pooling.ipynb) | Reference |
| [Activation Functions](notebooks/activation-functions.ipynb) | Reference |
| [Batch Normalization with the MNIST Dataset](notebooks/batch-normalization-with-the-mnist-dataset.ipynb) | Reference |
| [Convolutional Neural Network for Anime Image Classification](notebooks/convolutional-neural-network-for-anime-image-classification.ipynb) | Reference |
| [Convolutional Neural Network MNIST](notebooks/convolutional-neural-network-mnist.ipynb) | Reference |
| [Convolutional Neural Network Simple Example](notebooks/convolutional-neural-network-simple-example.ipynb) | Reference |
| [Convolutional Neural Networks with Batch Norms](notebooks/convolutional-neural-networks-with-batch-norms.ipynb) | Reference |
| [Deep Neural Networks](notebooks/deep-neural-networks.ipynb) | Reference |
| [Deeper Neural Networks with nn.ModuleList()](notebooks/deeper-neural-networks-with-nn-modulelist.ipynb) | Reference |
| [Fashion-MNIST Project](notebooks/fashion-mnist-project.ipynb) | Reference |
| [Initialization with Same Weights](notebooks/initialization-with-same-weights.ipynb) | Reference |
| [Logistic Regression and Bad Initialization Value](notebooks/logistic-regression-and-bad-initialization-value.ipynb) | Reference |
| [Logistic Regression Cross Entropy](notebooks/logistic-regression-cross-entropy.ipynb) | Reference |
| [Momentum with Different Polynomials](notebooks/momentum-with-different-polynomials.ipynb) | Reference |
| [Multiple Input and Output Channels](notebooks/multiple-input-and-output-channels.ipynb) | Reference |
| [Neural Networks More Hidden Neurons](notebooks/neural-networks-more-hidden-neurons.ipynb) | Reference |
| [Neural Networks with Momentum](notebooks/neural-networks-with-momentum.ipynb) | Reference |
| [Neural Networks with One Hidden Layer](notebooks/neural-networks-with-one-hidden-layer.ipynb) | Reference |
| [Practice: Neural Networks with One Hidden Layer: Noisy XOR](notebooks/practice-neural-networks-with-one-hidden-layer-noisy-xor.ipynb) | Reference |
| [Simple One Hidden Layer Neural Network](notebooks/simple-one-hidden-layer-neural-network.ipynb) | Reference |
| [Softmax Classifier 1D](notebooks/softmax-classifier-1d.ipynb) | Reference |
| [Softmax Classifier](notebooks/softmax-classifier.ipynb) | Reference |
| [Test Default, He, and Uniform Initialization on MNIST Dataset with ReLU Activation](notebooks/test-default-he-and-uniform-initialization-on-mnist-dataset-with-relu-activation.ipynb) | Reference |
| [Test Default, Xavier, and Uniform Initialization on MNIST Dataset with Tanh Activation](notebooks/test-default-xavier-and-uniform-initialization-on-mnist-dataset-with-tanh-activation.ipynb) | Reference |
| [Test Sigmoid, Tanh, and Relu Activations Functions on the MNIST Dataset](notebooks/test-sigmoid-tanh-and-relu-activations-functions-on-the-mnist-dataset.ipynb) | Reference |
| [Using Dropout for Classification](notebooks/using-dropout-for-classification.ipynb) | Reference |
| [Using Dropout in Regression](notebooks/using-dropout-in-regression.ipynb) | Reference |
| [What is Convolution?](notebooks/what-is-convolution.ipynb) | Reference |
