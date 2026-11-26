# Fraud detection under a 100-alert review budget

This is a new chronological evaluation of the course's fraud-detection problem, using original experiment code prepared with AI assistance. It replaces full-corpus preprocessing/feature-selection leakage with train-only fitting.

Remove exact repeated events and conflicting labels, then split at 60/80 percent time boundaries. Keep the natural rare-event prevalence. Time constructs the partitions but is excluded from model inputs. Fit a StandardScaler/logistic-regression pipeline, a depth6 decision tree and a prevalence baseline on training records only.

The validation scores choose a fixed cutoff under a 100-alert review budget; ties cannot exceed that validation budget. Report actual future workload under that frozen cutoff, which can drift above100. A separate top100 test ranking evaluates a fixed batch-review policy. Report average precision, ROC-AUC, precision/recall, true/false alerts and missed fraud. Average precision is the non-interpolated PR summary, not an unspecified trapezoidal PR-AUC. Save/reload the fresh local models and verify predictions.

Costs are illustrative units: one per reviewed alert and100 per missed fraud. These are not measured production costs or savings. The two-day anonymized dataset does not establish customer-group independence or the provenance of its upstream PCA fit.

Open the [standalone notebook](notebook.ipynb), [code](project.py), [fresh local execution](../../reports/second-batch-notebooks/fraud/executed.ipynb) and [second-batch release record](../../docs/SECOND_BATCH.md).

Data: [ULB/Worldline credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), fetched from the IBM teaching mirror using a pinned checksum. No transaction corpus is bundled as a public output.
