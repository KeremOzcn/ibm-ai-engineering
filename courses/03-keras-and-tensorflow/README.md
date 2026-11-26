# 03 · Deep Learning with Keras and TensorFlow

Functional models, transfer learning, generative models and reinforcement-learning introductions.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- Use the Functional API for models with multiple branches or shared layers. A custom Layer owns reusable computation and trainable state.
- Transfer learning starts from a pretrained representation. Freeze the backbone first, then consider a smaller learning rate for fine-tuning.
- Autoencoders reconstruct inputs; GANs learn through a generator-discriminator objective; diffusion models learn a denoising process.
- Q-learning estimates action values. A DQN replaces the value table with a neural network and needs a carefully controlled training loop.

## Working model

A custom training step computes predictions, evaluates the loss under GradientTape, differentiates with respect to trainable variables, and applies optimizer gradients.

## Start with these labs

- [Lab: Implementing the Functional API in Keras](notebooks/lab-implementing-the-functional-api-in-keras.ipynb) — Functional API.
- [Lab: Transfer Learning Implementation](notebooks/lab-transfer-learning-implementation.ipynb) — Transfer learning.
- [Waste Classification Using Transfer Learning](notebooks/waste-classification-using-transfer-learning.ipynb) — Waste classification project.

## Revisit with a purpose

- Sketch the data flow for the Functional API notebook, including each tensor shape.
- Compare a frozen feature extractor with fine-tuning using identical validation data.
- Use the waste classification starter to record augmentation, split rules and class-level errors.

## Common mistakes

- Augmentation belongs on training data; validation should reflect the intended evaluation distribution.
- Unfreezing layers without reviewing normalization behavior and learning rates can damage pretrained representations.
- One copyright-only source cell was incorrectly marked as executable code; the curated copy renders it as markdown.

## Check your understanding

- When does subclassing help more than a Sequential model?
- How would you isolate the effect of data augmentation?
- What is different about a generative objective and a classifier objective?

## Collected material

16 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Lab: Advanced Data Augmentation with Keras](notebooks/lab-advanced-data-augmentation-with-keras.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-advanced-data-augmentation-with-keras.ipynb) |
| [Lab: Building a Deep Q-Network with Keras](notebooks/lab-building-a-deep-q-network-with-keras.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-building-a-deep-q-network-with-keras.ipynb) |
| [Lab: Building Advanced Transformers](notebooks/lab-building-advanced-transformers.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-building-advanced-transformers.ipynb) |
| [Lab: Building Autoencoders](notebooks/lab-building-autoencoders.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-building-autoencoders.ipynb) |
| [Lab: Creating Custom Layers and Models](notebooks/lab-creating-custom-layers-and-models.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-creating-custom-layers-and-models.ipynb) |
| [Lab: Custom Training Loops in Keras](notebooks/lab-custom-training-loops-in-keras.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-custom-training-loops-in-keras.ipynb) |
| [Lab: Develop GANs Using Keras](notebooks/lab-develop-gans-using-keras.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-develop-gans-using-keras.ipynb) |
| [Lab: Hyperparameter Tuning with Keras Tuner](notebooks/lab-hyperparameter-tuning-with-keras-tuner.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-hyperparameter-tuning-with-keras-tuner.ipynb) |
| [Lab: Implementing Diffusion Models](notebooks/lab-implementing-diffusion-models.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-implementing-diffusion-models.ipynb) |
| [Lab: Implementing Q-Learning in Keras](notebooks/lab-implementing-q-learning-in-keras.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-implementing-q-learning-in-keras.ipynb) |
| [Lab: Implementing the Functional API in Keras](notebooks/lab-implementing-the-functional-api-in-keras.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-implementing-the-functional-api-in-keras.ipynb) |
| [Lab: Implementing Transformers for Text Generation](notebooks/lab-implementing-transformers-for-text-generation.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-implementing-transformers-for-text-generation.ipynb) |
| [Lab: Practical Application of Transpose Convolution](notebooks/lab-practical-application-of-transpose-convolution.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-practical-application-of-transpose-convolution.ipynb) |
| [Lab: Transfer Learning Implementation](notebooks/lab-transfer-learning-implementation.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/lab-transfer-learning-implementation.ipynb) |
| [Practice Project: Fruit Classification Using Transfer Learning](notebooks/practice-project-fruit-classification-using-transfer-learning.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/practice-project-fruit-classification-using-transfer-learning.ipynb) |
| [Waste Classification Using Transfer Learning](notebooks/waste-classification-using-transfer-learning.ipynb) | Reference | [Worked copy](../../solutions/03-keras-and-tensorflow/notebooks/waste-classification-using-transfer-learning.ipynb) |
