# Satellite CNN and spatial hybrid

This new experiment addresses the correctness findings in the course's [CNN classifier](../../solutions/06-computer-vision-capstone/notebooks/lab-implement-and-test-a-pytorch-based-classifier.ipynb), [hybrid construction](../../solutions/06-computer-vision-capstone/notebooks/lab-vision-transformers-using-pytorch.ipynb) and [hybrid evaluation](../../solutions/06-computer-vision-capstone/notebooks/lab-land-classification-cnn-transformer-integration-evaluation.ipynb).

- Fresh models train without the supplied course checkpoints, whose training membership is unverified.
- Train/validation/test are fixed by complete scene identifiers: T32UPE is test, T31UDQ is validation, and the other source scenes are training. Both classes must remain in every partition.
- Exact decoded-image duplicates are retained only once, with held-out copies taking priority; conflicting pixel labels are excluded and recorded. Geographic overlap and near-duplicates still need review.
- The stem retains an 8×8 feature map; patch projection produces **16 spatial tokens plus CLS**. It cannot collapse silently to one image token.
- The models share initial stem weights, sample order and deterministic image/epoch augmentation. They have different parameter counts, so this is not an isolated test of attention alone.
- Checkpoint selection uses validation macro F1. The selected weights are restored before test evaluation, and a fresh model must reproduce the selected checkpoint's probe logits exactly.

The default pilot uses 32 training, 16 validation and 16 test images. It checks implementation and evidence handling. Its scores cannot establish which architecture is better. Full-run study design still needs repeated seeds, training-budget decisions and larger held-out evaluation.

```bash
.venv/bin/python -m projects.satellite.project \
  --data-root .lab-cache/06-computer-vision-capstone/lab-vision-transformers-using-pytorch/images_dataSAT

# Separately budgeted full run, with a different artifact directory:
.venv/bin/python -m projects.satellite.project \
  --data-root PATH_TO_COMPLETE_IMAGES_DATASAT \
  --full --output .lab-cache/full-projects/satellite
```

Outputs: split manifest, selected checkpoints, per-model sample IDs/metrics, exact reload proof, runtime and source hashes. Open the [standalone notebook](notebook.ipynb) or [saved pilot evidence](../../reports/selected-projects/satellite).

## Kaggle release

The separately packaged [Satellite CNN vs Transformer Across Source Scenes](../../docs/KAGGLE_RELEASE.md) notebook includes provider-hosted data acquisition, saved runtime/source evidence and the larger evaluation budget. The local notebook above keeps its small pilot default. See the release record for public links and final cloud results.
