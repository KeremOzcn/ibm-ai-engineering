# Kaggle project release

The owner authorized Kaggle publication on 25 November 2026. All three independently authored project notebooks passed private Kaggle execution before their public versions were uploaded. These are separate experiments; the IBM reference and worked assessment archive remains private.

| Project | Notebook | Evaluation scope |
|---|---|---|
| Satellite CNN versus spatial hybrid | [Open on Kaggle](https://www.kaggle.com/code/kerem0zcan/satellite-cnn-vs-transformer-across-source-scenes) | 3,383 train / 93 validation / 2,524 test images; three epochs per fresh model; one seed |
| LoRA instruction tuning | [Open on Kaggle](https://www.kaggle.com/code/kerem0zcan/lora-instruction-tuning-and-verified-reload) | 512 train / 64 validation / 64 test records; one epoch / 256 updates; paired reference likelihood and generations |
| Aircraft damage and captions | [Open on Kaggle](https://www.kaggle.com/code/kerem0zcan/aircraft-damage-and-caption-reliability) | All 300 train / 96 validation / 50 test images; ten head epochs on frozen VGG16; two separate BLIP captions |

**Verified on 26 November 2026:** all three public version-2 notebooks are `COMPLETE`. Returned Kaggle metadata confirms `is_private=false`, and published source cells match the authored release notebooks. The [release ledger](../reports/kaggle/release.json) binds downloaded metrics, manifests, plots, logs, source and metadata to their hashes. `private-validation.json` records the earlier successful private stage; `public-uploads.json` records actual upload/version receipts.

| Final Kaggle measurement | Result |
|---|---|
| Satellite held-out macro F1, 2,524 images | CNN **0.845**; spatial hybrid **0.931** |
| LoRA held-out response NLL, 64 prompts / 4,889 response targets | Base **1.935**; reloaded adapter **1.661**; 256 updates |
| Aircraft classifier, 50 test images | **72% accuracy**, **0.718 macro F1** |

The [five-page LinkedIn PDF](../output/pdf/ibm-ai-engineering-portfolio.pdf) uses these final Kaggle results and contains clickable project links. Its evaluation date is 25 November 2026; the final visibility and file checks were repeated on 26 November.

Public visibility and execution were verified through the Kaggle API. Anonymous web checks returned empty 404 responses even for Kaggle home and established control pages, so browser/anonymous-link rendering is not claimed; the diagnostic is retained in `reports/kaggle/anonymous-pages.json`.

## Artifacts and reproduction

- `kaggle/<project>/`: standalone public notebook plus API metadata.
- `reports/portfolio-evaluations/`: separately measured local module runs.
- `reports/kaggle/<project>/`: final cloud metrics, split manifest, result plot, log, published source and returned metadata.
- `output/pdf/ibm-ai-engineering-portfolio.pdf`: five-page LinkedIn attachment, finalized from completed public Kaggle evidence and visually checked after rendering.

The provider-hosted datasets are fetched at runtime with fixed SHA-256 checks. Unsafe archive entries are rejected. Provider-backed reruns re-extract every member before training, so an interrupted extraction cannot be accepted just because a directory exists. Inputs and model checkpoints live in `/kaggle/temp`; only metrics, manifests and plots are persisted as Kaggle outputs. No course assessment notebooks, source archives or pretrained weights are uploaded. Attribution and usage conditions accompany each notebook.

The LoRA code uses Torch/Transformers/PEFT directly, without legacy torchtext, datasets or TRL. The generated runtime cell pins Transformers 4.45.2 and PEFT 0.11.1 while retaining Kaggle's Torch/TensorFlow builds. Actual versions and the selected Kaggle Docker image digest are captured with the release evidence. Public updates pin the original successful runtime image.

Generate private bundles with `.venv/bin/python scripts/build_kaggle_release.py`; use `--public` only after private cloud validation. Authenticate through the Kaggle CLI's supported local credential or `KAGGLE_API_TOKEN`; never place a credential in a notebook or repository. The isolated CLI environment is ignored under `.lab-cache/kaggle-cli/`. `kaggle kernels push -p kaggle/PROJECT --timeout 3600` saves and executes a version. Check `kaggle kernels status OWNER/SLUG`, collect outputs, and verify returned visibility/source/version before claiming publication. See the [official kernel documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

## Interpretation

The image projects use all retained source images at their documented epoch budgets. The LoRA project is explicitly a 512-record bounded training experiment. Response NLL is not functional code accuracy; no generated answers are executed or certified. No consistent memory/compute-quality advantage is claimed.

The local Torch 2.8 satellite hybrid and Kaggle Torch 2.11 hybrid produced materially different scores with the same authored source/seed; the CNN score matched. Keep both saved runtime records and use the final Kaggle values in the PDF. One seed and one setting do not establish an architecture ranking. Scene/near-duplicate overlap and source-photo grouping still require broader review. BLIP captions remain qualitative examples with null human-review fields.

Fraud detection remains a reserve. Mevzuat RAG stays a separate capstone. The study collection's 130 bounded course-lab passes and eight provider-gated labs are separate from this three-project release. No new course submission, GitHub push or LinkedIn post is part of this release.
