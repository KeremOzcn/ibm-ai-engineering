# 02 · Introduction to Deep Learning and Neural Networks with Keras

Forward propagation, backpropagation, activation functions and first Keras models.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- A dense layer computes an affine transformation followed by an activation. Nonlinear activations allow a network to learn nonlinear relationships.
- Backpropagation applies the chain rule to compute gradients. The optimizer updates parameters; it does not define the loss.
- Choose the output layer and loss together: logits and cross-entropy for classification, a continuous output and a regression loss for regression.
- Validation curves guide regularization and stopping. A final test set is for evaluation after architecture decisions.

## Working model

For a batch X of shape (B, d), a dense layer with h units maps XW + b to shape (B, h). Gradient descent updates θ ← θ − η∇θL.

## Start with these labs

- [Artificial Neural Networks - Forward Propagation](notebooks/artificial-neural-networks-forward-propagation.ipynb) — Forward propagation from first principles.
- [Backpropagation](notebooks/backpropagation.ipynb) — Gradient flow.
- [Aircraft Damage Classification and Captioning](notebooks/aircraft-damage-classification-and-captioning.ipynb) — Aircraft damage project.

## Revisit with a purpose

- Trace the shapes through a two-layer dense model before running it.
- Compare a linear baseline with a small MLP while fixing the split and training budget.
- Inspect the aircraft damage classification and captioning starter; distinguish discriminative and generative outputs.

## Common mistakes

- Large networks can memorize small datasets.
- Softmax probabilities and raw logits require matching loss settings.
- The final project has unfinished hyperparameter assignments; it is a study template.

## Check your understanding

- What does a vanishing gradient do to early layers?
- How do batch size and learning rate affect optimization?
- Why does low training loss fail to prove useful generalization?

## Collected material

8 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Activation Functions and Vanishing Gradients](notebooks/activation-functions-and-vanishing-gradients.ipynb) | Reference | [Worked copy](../../solutions/02-deep-learning-with-keras/notebooks/activation-functions-and-vanishing-gradients.ipynb) |
| [Aircraft Damage Classification and Captioning](notebooks/aircraft-damage-classification-and-captioning.ipynb) | Reference; 1 syntax blanks | [Worked copy](../../solutions/02-deep-learning-with-keras/notebooks/aircraft-damage-classification-and-captioning.ipynb) |
| [Artificial Neural Networks - Forward Propagation](notebooks/artificial-neural-networks-forward-propagation.ipynb) | Reference | [Worked copy](../../solutions/02-deep-learning-with-keras/notebooks/artificial-neural-networks-forward-propagation.ipynb) |
| [Backpropagation](notebooks/backpropagation.ipynb) | Reference | [Worked copy](../../solutions/02-deep-learning-with-keras/notebooks/backpropagation.ipynb) |
| [Classification Models with Keras](notebooks/classification-models-with-keras.ipynb) | Reference | [Worked copy](../../solutions/02-deep-learning-with-keras/notebooks/classification-models-with-keras.ipynb) |
| [Convolutional Neural Networks with Keras](notebooks/convolutional-neural-networks-with-keras.ipynb) | Reference | [Worked copy](../../solutions/02-deep-learning-with-keras/notebooks/convolutional-neural-networks-with-keras.ipynb) |
| [Regression Models with Keras](notebooks/regression-models-with-keras.ipynb) | Reference | [Worked copy](../../solutions/02-deep-learning-with-keras/notebooks/regression-models-with-keras.ipynb) |
| [Transformers with Keras](notebooks/transformers-with-keras.ipynb) | Reference | [Worked copy](../../solutions/02-deep-learning-with-keras/notebooks/transformers-with-keras.ipynb) |
