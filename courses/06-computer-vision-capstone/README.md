# 06 · AI Capstone Project with Deep Learning

Satellite land classification with Keras, PyTorch, vision transformers and CNN–ViT integration.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- Build a dataset inventory before training: class counts, image dimensions, duplicate images and split membership.
- Keep data preprocessing and class mappings consistent when comparing Keras and PyTorch classifiers.
- A vision transformer converts image patches into token embeddings and models their relationships with attention.
- A hybrid CNN–ViT combines local visual features with attention. Compare it to simpler baselines under an explicit resource budget.

## Working model

For square images of side H and patch size P, the image contributes (H/P)² patch tokens when H is divisible by P. Attention cost grows with the square of the token count.

## Start with these labs

- [Compare Memory-Based versus Generator-Based Data Loading](notebooks/compare-memory-based-versus-generator-based-data-loading.ipynb) — Dataset loading.
- [Data Loading and Augmentation Using Keras](notebooks/data-loading-and-augmentation-using-keras.ipynb) — Data preprocessing.
- [Lab: Data Loading and Augmentation Using PyTorch](notebooks/lab-data-loading-and-augmentation-using-pytorch.ipynb) — Augmentation.
- [Lab: Land Classification: CNN-Transformer Integration evaluation](notebooks/lab-land-classification-cnn-transformer-integration-evaluation.ipynb) — CNN–ViT integration.

## Revisit with a purpose

- Start with data loading, then preprocessing and augmentation, then the framework-specific classifiers.
- Compare the Keras and PyTorch experiments using the same held-out images and class ordering.
- Write a model card covering the dataset, metric, compute budget and likely geographic distribution shift.

## Common mistakes

- Similar satellite tiles can leak across a random split.
- Different preprocessing or checkpoint weights can invalidate model comparisons.
- These are the collected course templates; previously executed final-submission artifacts from another workspace are not imported here.

## Check your understanding

- How would you split images to reduce location leakage?
- What evidence would justify choosing a hybrid over a smaller CNN?
- Which preprocessing steps must be shared across framework comparisons?

## Collected material

9 reference notebooks. Course source, authorship and copyright notices remain in the collected files. No course notebook has been certified as runnable by this curation.

| Material | Status |
|---|---|
| [Compare Memory-Based versus Generator-Based Data Loading](notebooks/compare-memory-based-versus-generator-based-data-loading.ipynb) | Reference |
| [Data Loading and Augmentation Using Keras](notebooks/data-loading-and-augmentation-using-keras.ipynb) | Reference |
| [Lab: Comparative Analysis of Keras and PyTorch Models](notebooks/lab-comparative-analysis-of-keras-and-pytorch-models.ipynb) | Reference |
| [Lab: Data Loading and Augmentation Using PyTorch](notebooks/lab-data-loading-and-augmentation-using-pytorch.ipynb) | Reference |
| [Lab: Implement and Test a PyTorch-Based Classifier](notebooks/lab-implement-and-test-a-pytorch-based-classifier.ipynb) | Reference |
| [Lab: Land Classification: CNN-Transformer Integration evaluation](notebooks/lab-land-classification-cnn-transformer-integration-evaluation.ipynb) | Reference |
| [Lab: Train and Evaluate a Keras-Based Classifier](notebooks/lab-train-and-evaluate-a-keras-based-classifier.ipynb) | Reference |
| [Lab: Vision Transformers Using Keras](notebooks/lab-vision-transformers-using-keras.ipynb) | Reference |
| [Lab: Vision Transformers Using PyTorch](notebooks/lab-vision-transformers-using-pytorch.ipynb) | Reference |
