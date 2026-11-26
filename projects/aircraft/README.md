# Aircraft classification and caption reliability

This new implementation corrects the preprocessing, partial test evaluation and hidden caption failures found in the [course aircraft notebook](../../solutions/02-deep-learning-with-keras/notebooks/aircraft-damage-classification-and-captioning.ipynb).

- The existing train/valid/test directories are indexed and checked by decoded-image hashes. Exact duplicates and conflicting labels are handled explicitly and recorded. Source-image grouping and near-duplicates still need review.
- VGG16 receives RGB values in 0–255 followed by the official BGR/channel-centering preprocessing, without 1/255 scaling. Display images stay unchanged.
- Frozen ImageNet VGG16 features feed a newly trained classification head. Validation loss selects the saved head; the selected checkpoint is reloaded before testing.
- Feature extraction includes the final incomplete batch, so evaluation cannot silently drop the last images.
- Offline loading verifies and directly loads the cached VGG H5 file. It does not invoke Keras' downloader path. BLIP likewise uses local cached files unless downloads are explicitly enabled.
- BLIP descriptions come from a separate pretrained model. Caption failures propagate. They are not damage-localization results, a jointly trained system or independently trained summary outputs.

The default pilot uses 16 training, eight validation and eight test images. It checks code paths and reloads, not useful classifier accuracy. Two real captions are retained with **null review fields**; their relevance/hallucination quality has not been certified.

```bash
.venv/bin/python -m projects.aircraft.project \
  --data-root .lab-cache/02-deep-learning-with-keras/aircraft-damage-classification-and-captioning/aircraft_damage_dataset_v1

# Separately budgeted complete classifier run:
.venv/bin/python -m projects.aircraft.project \
  --data-root PATH_TO_AIRCRAFT_DAMAGE_DATASET \
  --full --output .lab-cache/full-projects/aircraft
```

The [official VGG preprocessing contract](https://keras.io/api/applications/vgg/vgg_preprocessing/) is the basis for the input correction. Open the [standalone notebook](notebook.ipynb) or [saved pilot evidence](../../reports/selected-projects/aircraft).

## Kaggle release

The separately packaged [Aircraft Damage and Caption Reliability](../../docs/KAGGLE_RELEASE.md) notebook includes provider-hosted data acquisition, saved runtime/source evidence and the larger evaluation budget. The local notebook above keeps its small pilot default. See the release record for public links and final cloud results.
