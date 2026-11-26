# Correctness audit of the three selected projects

The owner approved these three projects on 25 November 2026. The audit found substantive issues in the study material, so corrected, separately authored experiment implementations were added under [projects/](../projects/README.md). Original course references and the Antigravity-completed worked copies remain intact.


Subsequent work expanded the image evaluations to all source splits and LoRA to 512 training records/64 held-out prompts. All three passed private Kaggle runs and were uploaded as public versions. See the [Kaggle release](KAGGLE_RELEASE.md) for the later cloud evidence; the historical pilot findings below remain valid within their original scope.

## Original local-pilot result and scope

All three standalone project notebooks executed top to bottom in fresh local kernels using bounded pilots. The exported [evidence](../reports/selected-projects) binds source/notebook hashes, split manifests, executed notebooks, metrics and checkpoint reload checks. This is local implementation evidence. Full training, a real Kaggle run, meaningful task-quality comparisons and public publication remain unverified.

| Project | Original issue | Corrected experiment |
|---|---|---|
| Satellite | Supplied checkpoints evaluated on the whole course corpus; no independently reserved test set | Fresh models, fixed whole-scene train/validation/test partitions and persisted input hashes |
| Satellite | Six pooling stages leave one spatial image token | New stem preserves 8×8 features and produces 16 image patches plus CLS |
| Satellite | Some course metrics divide partial-run sums by the full dataset size | Metrics/loss use actual processed examples; regression includes a partially yielded loader |
| Satellite | Final evaluation does not restore the promised best checkpoint; training shuffles depend on shared global RNG | Validation selects/restores weights; independent seeded loaders and image/epoch augmentation give both models identical sample order |
| Satellite | Keras/PyTorch demos differ in freezing, head widths and architecture | New comparison stays in PyTorch; parameter counts differ and are disclosed, so an isolated attention-only effect is not claimed |
| LoRA | Main training is commented out and fresh predictions are overwritten by supplied pickles | Real optimization, fresh paired generations and no imported training logs or prediction files |
| LoRA | Response boundaries, padding and reference extraction can be misleading | Explicit response-only labels, position-based padding masks, raw source references and recorded response truncation without invented EOS |
| LoRA | Demonstration lacks independent selected-adapter reload evidence | Immutable base revision, validation-selected adapter, safe checkpoint and fresh-base reload check |
| Aircraft | 1/255 scaling does not match frozen ImageNet VGG16 preprocessing | Correct BGR/channel centering; raw RGB display inputs remain unchanged |
| Aircraft | Floor-divided evaluation drops the remainder batch | Every selected example is included; a 17-image/8-batch regression checks the last sample |
| Aircraft | Caption errors become ordinary-looking strings; model roles share a misleading interface | Direct BLIP errors propagate; classifier and captioner remain distinct; no separately trained summary model is claimed |
| Aircraft | An offline flag can still invoke Keras' downloader on corrupt or read-only caches | Verify the published VGG checksum and load its explicit file path |

## What was actually exercised

| Project | Pilot size | Training/reload proof |
|---|---|---|
| Satellite | 32 train / 16 validation / 16 test images; full scene partition has 3,383 / 93 / 2,524 images | CNN and spatial hybrid train from fresh weights; shared stem initialization/order; both selected checkpoints reproduce probe logits exactly |
| LoRA | 32 selected train / 4 validation / 4 test records; two steps process four training records | Adapter weights change; a fresh OPT base at the same revision plus saved adapter reproduces selected probe logits; predictions are regenerated locally |
| Aircraft | 16 train / 8 validation / 8 test images; complete source splits have 300 / 96 / 50 images | New classifier head trains on frozen VGG features; selected classifier reloads exactly; two BLIP descriptions are generated |

Exact decoded-image checks found no duplicate or conflicting-label exclusions in either actual image dataset. This does not rule out near-duplicates, common source photographs or geographical overlap. These broader data checks are explicitly left unverified.

## Quality findings that prevent a readiness claim

The pilots are intentionally too small to establish usable model performance. The spatial hybrid did not outperform the CNN in this pilot; that observation is not a reliable architecture ranking. Two LoRA steps do not establish a useful instruction-following model or robust improvement, regardless of a small change in response loss.

The aircraft captioner produced clear semantic failures on the two inspected pilot examples: one description identified a car where a damaged aircraft nose/cockpit is visible, and the other described a person's feet where a dented metal surface is visible. Both omit the visible damage. See the [assistant visual-review record](../reports/selected-projects/aircraft/visual-review.json). This is model-assisted inspection of two examples, not human-validated annotation or a caption-quality benchmark. The saved human-review fields remain null.

The aircraft project is therefore currently a classifier plus a **caption reliability/failure-analysis case study**, not a validated damage-description system. Task checks for generated code, caption review, repeated training seeds, appropriate training budgets and a clean Kaggle `Save & Run All` are still needed before public performance claims.

## Reproduce the checks

```bash
.venv/bin/python scripts/build_project_notebooks.py --check
.venv/bin/python scripts/check_selected_projects.py
.venv/bin/python -m unittest discover -s tests -v
.venv-legacy/bin/python -m unittest discover -s tests -p test_selected_projects.py -v
.venv/bin/ruff check scripts tests projects
.venv/bin/ruff format --check scripts tests projects
.venv/bin/python scripts/check_repository.py
```

The selected tests cover spatial token shape/gradient flow, scene separation, exact-duplicate cleanup, partially processed metrics, restoring an earlier checkpoint, instruction-group separation, prompt/padding/EOS masking, token-weighted response loss, actual tiny-model LoRA updates/reload, VGG input semantics, offline weight verification, remainder batches and caption-error propagation. Framework-specific skips are exercised in the complementary installed environments.

Implementation references: [Keras VGG input preprocessing](https://keras.io/api/applications/vgg/vgg_preprocessing/), [PEFT model/checkpoint API](https://huggingface.co/docs/peft/v0.11.0/en/package_reference/peft_model), [causal generation API](https://huggingface.co/docs/transformers/v4.45.2/en/tasks/language_modeling). The project scripts are new AI-assisted experiment code; data/model ownership remains with the original providers.
