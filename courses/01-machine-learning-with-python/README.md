# 01 · Machine Learning with Python

Regression, classification, clustering and model selection with scikit-learn.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- Regression: predict a continuous target; classification: predict a class or its probability. Choose metrics that match the decision.
- A pipeline fits preprocessing on each training fold. Fit scalers and imputers after splitting to avoid leakage.
- Cross-validation selects hyperparameters on the training partition. The final test partition stays untouched until the model choice is fixed.
- For clustering, inspect stability and domain meaning as well as silhouette scores. PCA summarizes variance; t-SNE and UMAP are primarily exploratory views.

## Working model

Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2PR / (P + R). MAE and RMSE use target units; R² compares squared error to a mean predictor.

## Start with these labs

- [Machine Learning Pipelines and GridSearchCV](notebooks/machine-learning-pipelines-and-gridsearchcv.ipynb) — Model selection and preprocessing.
- [Final Project: Building a Rainfall Prediction Classifier](notebooks/final-project-building-a-rainfall-prediction-classifier.ipynb) — Rainfall classification project.
- [Lab: Applications of Principal Component Analysis (PCA)](notebooks/lab-applications-of-principal-component-analysis-pca.ipynb) — Dimensionality reduction.

## Revisit with a purpose

- Compare a DummyClassifier with a scaled logistic-regression pipeline using the same folds.
- Revisit the rainfall project; record class balance, preprocessing, split rule and confusion matrix.
- Explain when a stratified random split would be inappropriate, such as grouped customers or time series.

## Common mistakes

- Preprocessing all data before splitting leaks information.
- Training accuracy and a single favorable split can hide weak generalization.
- Several exercise cells contain deliberate syntax blanks; inspect the catalog before using Run All.

## Check your understanding

- Why does scaling belong inside the pipeline?
- Which error is more costly in a chosen application: false positives or false negatives?
- When would PR-AUC be more useful than accuracy?

## Collected material

20 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Comparing DBSCAN and HDBSCAN clustering](notebooks/comparing-dbscan-and-hdbscan-clustering.ipynb) | Reference; 1 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/comparing-dbscan-and-hdbscan-clustering.ipynb) |
| [Credit Card Fraud Detection with Decision Trees and SVM](notebooks/credit-card-fraud-detection-with-decision-trees-and-svm.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/credit-card-fraud-detection-with-decision-trees-and-svm.ipynb) |
| [Decision Trees](notebooks/decision-trees.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/decision-trees.ipynb) |
| [Evaluating Classification Models](notebooks/evaluating-classification-models.ipynb) | Reference; 1 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/evaluating-classification-models.ipynb) |
| [Evaluating k-means clustering](notebooks/evaluating-k-means-clustering.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/evaluating-k-means-clustering.ipynb) |
| [Evaluating Random Forest Performance](notebooks/evaluating-random-forest-performance.ipynb) | Reference; 1 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/evaluating-random-forest-performance.ipynb) |
| [Final Project: Building a Rainfall Prediction Classifier](notebooks/final-project-building-a-rainfall-prediction-classifier.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/final-project-building-a-rainfall-prediction-classifier.ipynb) |
| [K-Means Customer Segmentation](notebooks/k-means-customer-segmentation.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/k-means-customer-segmentation.ipynb) |
| [K-Nearest Neighbors Classification](notebooks/k-nearest-neighbors-classification.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/k-nearest-neighbors-classification.ipynb) |
| [Lab: Applications of Principal Component Analysis (PCA)](notebooks/lab-applications-of-principal-component-analysis-pca.ipynb) | Reference; 2 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/lab-applications-of-principal-component-analysis-pca.ipynb) |
| [Lab: Comparing Random Forest and XGBoost modeling performance](notebooks/lab-comparing-random-forest-and-xgboost-modeling-performance.ipynb) | Reference; 4 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/lab-comparing-random-forest-and-xgboost-modeling-performance.ipynb) |
| [Lab: Comparing tSNE and UMAP Dimension Reduction Models](notebooks/lab-comparing-tsne-and-umap-dimension-reduction-models.ipynb) | Reference; 1 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/lab-comparing-tsne-and-umap-dimension-reduction-models.ipynb) |
| [Logistic Regression with Python](notebooks/logistic-regression-with-python.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/logistic-regression-with-python.ipynb) |
| [Machine Learning Pipelines and GridSearchCV](notebooks/machine-learning-pipelines-and-gridsearchcv.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/machine-learning-pipelines-and-gridsearchcv.ipynb) |
| [Multi-class Classification](notebooks/multi-class-classification.ipynb) | Reference; 1 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/multi-class-classification.ipynb) |
| [Multiple Linear Regression](notebooks/multiple-linear-regression.ipynb) | Reference; 5 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/multiple-linear-regression.ipynb) |
| [Practice Project: Titanic Survival Prediction](notebooks/practice-project-titanic-survival-prediction.ipynb) | Reference; 2 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/practice-project-titanic-survival-prediction.ipynb) |
| [Regression Trees for Taxi Tip Prediction](notebooks/regression-trees-for-taxi-tip-prediction.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/regression-trees-for-taxi-tip-prediction.ipynb) |
| [Regularization in Linear Regression](notebooks/regularization-in-linear-regression.ipynb) | Reference | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/regularization-in-linear-regression.ipynb) |
| [Simple Linear Regression](notebooks/simple-linear-regression.ipynb) | Reference; 3 syntax blanks | [Worked copy](../../solutions/01-machine-learning-with-python/notebooks/simple-linear-regression.ipynb) |
