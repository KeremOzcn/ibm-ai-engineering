# 04 · Introduction to Neural Networks and PyTorch

Tensors, autograd, datasets and gradient-based learning.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- A tensor has a shape, dtype and device. Check all three at the boundaries of a model.
- Autograd records differentiable operations. backward() accumulates gradients, so clear them between optimization steps.
- Dataset describes samples; DataLoader groups and optionally shuffles them. Shuffle training examples without mixing the evaluation partition into training.
- For binary logits use BCEWithLogitsLoss; for multiclass logits use CrossEntropyLoss with integer class indices.

## Working model

The training loop is zero_grad → forward → loss → backward → step. Evaluation uses model.eval() together with a no-gradient context.

## Start with these labs

- [Torch Tensors in 1D](notebooks/torch-tensors-in-1d.ipynb) — Tensor operations.
- [Differentiation in PyTorch](notebooks/differentiation-in-pytorch.ipynb) — Automatic differentiation.
- [Final Project: League of Legends Match Predictor](notebooks/final-project-league-of-legends-match-predictor.ipynb) — Match prediction starter.

## Revisit with a purpose

- Run the companion PyTorch training loop and inspect the train/validation/test separation.
- Compare full-batch, stochastic and mini-batch gradient descent on the same regression problem.
- Review the match predictor starter and write down features that would leak the final match result.

## Common mistakes

- eval() changes layer behavior; it does not disable gradient recording.
- The source named Wine Classification actually loads the breast-cancer dataset; the curated title follows the content.
- The cross-entropy notebook was collected from course 05 because the course 04 page had no direct download link.

## Check your understanding

- Why do gradients accumulate by default?
- What happens if floating-point labels are passed to multiclass cross-entropy?
- Why should normalization statistics come from training data?

## Collected material

20 reference notebooks. Course source, authorship and copyright notices remain in the collected files. No course notebook has been certified as runnable by this curation.

| Material | Status |
|---|---|
| [Deep Neural Network for Breast Cancer Classification](notebooks/deep-neural-network-for-breast-cancer-classification.ipynb) | Reference |
| [Differentiation in PyTorch](notebooks/differentiation-in-pytorch.ipynb) | Reference |
| [Final Project: League of Legends Match Predictor](notebooks/final-project-league-of-legends-match-predictor.ipynb) | Reference |
| [Linear Regression 1D: Optimization in PyTorch](notebooks/linear-regression-1d-optimization-in-pytorch.ipynb) | Reference |
| [Linear Regression 1D: Prediction](notebooks/linear-regression-1d-prediction.ipynb) | Reference |
| [Linear Regression 1D: Training Two Parameter Mini-Batch Gradient Descent](notebooks/linear-regression-1d-training-two-parameter-mini-batch-gradient-descent.ipynb) | Reference |
| [Linear regression 1D: Training Two Parameter Stochastic Gradient Descent (SGD)](notebooks/linear-regression-1d-training-two-parameter-stochastic-gradient-descent-sgd.ipynb) | Reference |
| [Linear regression 1D: Training Two Parameter](notebooks/linear-regression-1d-training-two-parameter.ipynb) | Reference |
| [Linear Regression Multiple Outputs](notebooks/linear-regression-multiple-outputs.ipynb) | Reference |
| [Linear regression: Training and Validation Data](notebooks/linear-regression-training-and-validation-data.ipynb) | Reference |
| [Linear Regression with Multiple Outputs](notebooks/linear-regression-with-multiple-outputs.ipynb) | Reference |
| [Logistic Regression and Bad Initialization Value](notebooks/logistic-regression-and-bad-initialization-value.ipynb) | Reference |
| [Logistic Regression Cross Entropy](notebooks/logistic-regression-cross-entropy.ipynb) | Reference |
| [Logistic Regression](notebooks/logistic-regression.ipynb) | Reference |
| [Multiple Linear Regression](notebooks/multiple-linear-regression.ipynb) | Reference |
| [Prebuilt Datasets and Transforms](notebooks/prebuilt-datasets-and-transforms.ipynb) | Reference |
| [Simple Dataset](notebooks/simple-dataset.ipynb) | Reference |
| [Torch Tensors in 1D](notebooks/torch-tensors-in-1d.ipynb) | Reference |
| [Training Linear Regression with Multiple Outputs](notebooks/training-linear-regression-with-multiple-outputs.ipynb) | Reference |
| [Two-Dimensional Tensors](notebooks/two-dimensional-tensors.ipynb) | Reference |
