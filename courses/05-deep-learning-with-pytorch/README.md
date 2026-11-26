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

28 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Activation Functions and Max Pooling](notebooks/activation-functions-and-max-pooling.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/activation-functions-and-max-pooling.ipynb) |
| [Activation Functions](notebooks/activation-functions.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/activation-functions.ipynb) |
| [Batch Normalization with the MNIST Dataset](notebooks/batch-normalization-with-the-mnist-dataset.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/batch-normalization-with-the-mnist-dataset.ipynb) |
| [Convolutional Neural Network for Anime Image Classification](notebooks/convolutional-neural-network-for-anime-image-classification.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/convolutional-neural-network-for-anime-image-classification.ipynb) |
| [Convolutional Neural Network MNIST](notebooks/convolutional-neural-network-mnist.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/convolutional-neural-network-mnist.ipynb) |
| [Convolutional Neural Network Simple Example](notebooks/convolutional-neural-network-simple-example.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/convolutional-neural-network-simple-example.ipynb) |
| [Convolutional Neural Networks with Batch Norms](notebooks/convolutional-neural-networks-with-batch-norms.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/convolutional-neural-networks-with-batch-norms.ipynb) |
| [Deep Neural Networks](notebooks/deep-neural-networks.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/deep-neural-networks.ipynb) |
| [Deeper Neural Networks with nn.ModuleList()](notebooks/deeper-neural-networks-with-nn-modulelist.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/deeper-neural-networks-with-nn-modulelist.ipynb) |
| [Fashion-MNIST Project](notebooks/fashion-mnist-project.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/fashion-mnist-project.ipynb) |
| [Initialization with Same Weights](notebooks/initialization-with-same-weights.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/initialization-with-same-weights.ipynb) |
| [Logistic Regression and Bad Initialization Value](notebooks/logistic-regression-and-bad-initialization-value.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/logistic-regression-and-bad-initialization-value.ipynb) |
| [Logistic Regression Cross Entropy](notebooks/logistic-regression-cross-entropy.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/logistic-regression-cross-entropy.ipynb) |
| [Momentum with Different Polynomials](notebooks/momentum-with-different-polynomials.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/momentum-with-different-polynomials.ipynb) |
| [Multiple Input and Output Channels](notebooks/multiple-input-and-output-channels.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/multiple-input-and-output-channels.ipynb) |
| [Neural Networks More Hidden Neurons](notebooks/neural-networks-more-hidden-neurons.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/neural-networks-more-hidden-neurons.ipynb) |
| [Neural Networks with Momentum](notebooks/neural-networks-with-momentum.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/neural-networks-with-momentum.ipynb) |
| [Neural Networks with One Hidden Layer](notebooks/neural-networks-with-one-hidden-layer.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/neural-networks-with-one-hidden-layer.ipynb) |
| [Practice: Neural Networks with One Hidden Layer: Noisy XOR](notebooks/practice-neural-networks-with-one-hidden-layer-noisy-xor.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/practice-neural-networks-with-one-hidden-layer-noisy-xor.ipynb) |
| [Simple One Hidden Layer Neural Network](notebooks/simple-one-hidden-layer-neural-network.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/simple-one-hidden-layer-neural-network.ipynb) |
| [Softmax Classifier 1D](notebooks/softmax-classifier-1d.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/softmax-classifier-1d.ipynb) |
| [Softmax Classifier](notebooks/softmax-classifier.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/softmax-classifier.ipynb) |
| [Test Default, He, and Uniform Initialization on MNIST Dataset with ReLU Activation](notebooks/test-default-he-and-uniform-initialization-on-mnist-dataset-with-relu-activation.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/test-default-he-and-uniform-initialization-on-mnist-dataset-with-relu-activation.ipynb) |
| [Test Default, Xavier, and Uniform Initialization on MNIST Dataset with Tanh Activation](notebooks/test-default-xavier-and-uniform-initialization-on-mnist-dataset-with-tanh-activation.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/test-default-xavier-and-uniform-initialization-on-mnist-dataset-with-tanh-activation.ipynb) |
| [Test Sigmoid, Tanh, and Relu Activations Functions on the MNIST Dataset](notebooks/test-sigmoid-tanh-and-relu-activations-functions-on-the-mnist-dataset.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/test-sigmoid-tanh-and-relu-activations-functions-on-the-mnist-dataset.ipynb) |
| [Using Dropout for Classification](notebooks/using-dropout-for-classification.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/using-dropout-for-classification.ipynb) |
| [Using Dropout in Regression](notebooks/using-dropout-in-regression.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/using-dropout-in-regression.ipynb) |
| [What is Convolution?](notebooks/what-is-convolution.ipynb) | Reference | [Worked copy](../../solutions/05-deep-learning-with-pytorch/notebooks/what-is-convolution.ipynb) |
