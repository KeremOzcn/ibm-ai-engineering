# Selected Kaggle project shortlist

Assessed on 25 November 2026 against the current worked notebooks, saved bounded executions and cached course data. The owner accepted the three main projects. Corrected experiment implementations and local pilot evidence are now under [projects/](../projects/README.md); see the [correctness audit](SELECTED_PROJECT_AUDIT.md). The assessments below describe the original course foundations. The three accepted experiments subsequently passed private Kaggle runs and were uploaded as public versions; see the [release evidence](KAGGLE_RELEASE.md).

The collection contains 138 worked notebooks and 130 current bounded passes. Those passes establish execution of the adapted code. They do not establish complete training, independent model improvements or Kaggle compatibility. See [solution progress](SOLUTION_PROGRESS.md) and [full-run scope](FULL_RUNS.md).

## Recommended first batch

| Priority | Proposed project | Why it earns a portfolio slot | Preparation |
|---|---|---|---|
| 1 | Satellite land classification: CNN versus CNN–ViT under scene shift | Substantial architecture, image data, controlled comparison and a meaningful generalization question | High |
| 2 | Efficient instruction tuning: base model versus LoRA | Actual language-model adaptation, quality/compute trade-offs and checkpoint reproducibility | High |
| 3 | Aircraft damage classification and caption reliability | Combines visual classification with image-to-text inference and produces clear visual case studies | Medium |
| Reserve | Fraud detection with a limited review budget | Shows mature evaluation of rare events and operational trade-offs; simpler models can still support a strong engineering story | Medium |

Three strong case studies would cover vision, language-model training and multimodal inference. The reserve project adds classical ML breadth if needed. Mevzuat RAG remains the separate RAG capstone.

## 1. Satellite land classification

Build one project from the [PyTorch CNN classifier](../solutions/06-computer-vision-capstone/notebooks/lab-implement-and-test-a-pytorch-based-classifier.ipynb), [CNN–ViT construction/training](../solutions/06-computer-vision-capstone/notebooks/lab-vision-transformers-using-pytorch.ipynb) and [hybrid evaluation](../solutions/06-computer-vision-capstone/notebooks/lab-land-classification-cnn-transformer-integration-evaluation.ipynb). The nine course-06 notebooks are stages of this project, rather than nine separate portfolio projects.

The architecture work is already substantial: convolutional backbone, patch embedding, multi-head self-attention, transformer blocks and a hybrid classifier. The source material also supports Keras/PyTorch comparison, but framework comparisons need equivalent models and settings to be interpretable. Start with a controlled comparison in one framework.

The cached archive contains 6,000 images from four source-scene filename groups:

| Scene suffix | Non-agricultural | Agricultural |
|---|---:|---:|
| 2025-04-09 / T31UDQ | 55 | 38 |
| 2025-04-27 / T32UPE | 1,316 | 1,208 |
| 2025-04-27 / T32UQB | 1,629 | 1,641 |
| 2025-04-27 / T32UQD | 0 | 113 |

This supports a stronger question than a single accuracy number: how does performance change between a random image split and a held-out source scene? The two large groups contain both classes. The small single-class group needs separate treatment; it cannot support a two-class ROC-AUC on its own. Scene grouping alone is not proof that all geographical overlap or near-duplicate images are excluded; audit those as well.

For the public version, create a fixed split manifest, train the compared models with a documented budget, and report macro F1, per-class recall, confusion matrices, training cost and representative mistakes. Include an ablation such as frozen versus trainable backbone or transformer depth. Use fresh training or an independently sourced initialization: the supplied course checkpoints do not establish an untouched holdout or personal training provenance.

Suggested title: **CNN vs CNN–ViT: Satellite Classification Across Source Scenes**.

## 2. Efficient instruction tuning

Use [instruction tuning with OPT-350M](../solutions/11-advanced-fine-tuning/notebooks/instruction-tuning-with-llms.ipynb) as the actual language-model foundation. The [LoRA implementation lab](../solutions/10-fine-tuning-transformers/notebooks/lora-with-pytorch.ipynb) provides useful implementation explanations; its main classifier is GloVe embedding, mean pooling and an MLP, so it should not be presented as an LLM fine-tuning result by itself. The [adapter lab](../solutions/10-fine-tuning-transformers/notebooks/adapters-in-pytorch.ipynb) is a possible later comparison.

The instruction-tuning notebook currently loads supplied base/LoRA predictions and training logs from course URLs; its main `trainer.train()` call is commented out. Its displayed BLEU comparison is therefore not evidence that the local adapter improved. The bounded setup also restricts the training configuration. An original portfolio experiment must generate and evaluate both sets of outputs from the exact models being compared.

Freeze the evaluation prompts and decoding settings before training. Compare the same base model with its saved/reloaded LoRA adapter; add a full-fine-tuning or adapter baseline only if the resource budget permits. Record trainable parameters, peak memory, time and held-out task quality. For coding instructions, include task checks and failure examples rather than relying on BLEU alone. Report a negative result if the adapter does not improve.

Suggested title: **What Does LoRA Buy? Quality, Memory and Training Cost in Instruction Tuning**.

## 3. Aircraft classification and caption reliability

The [aircraft project](../solutions/02-deep-learning-with-keras/notebooks/aircraft-damage-classification-and-captioning.ipynb) combines VGG16 classification of **cracks versus dents** with pretrained BLIP captions. This gives it a distinct multimodal story and useful image/prediction panels.

The current classifier and captioner are separate models. There is no demonstrated joint training, damage localization or validated caption-quality benchmark. The `caption` and `summary` modes change the text prompt sent to the same captioning model; they are not separately trained task models.

The public version should train and evaluate the classifier on a fixed split, include a confusion matrix, and review captions for whether they describe the visible defect, omit it or invent details. Show successful and failed examples together. Explain cases where the classifier and caption disagree, and let image-processing errors propagate rather than returning a success-shaped error string.

Suggested title: **Crack or Dent? Aircraft Classification and the Limits of Image Captions**.

## Reserve: fraud detection

The [fraud notebook](../solutions/01-machine-learning-with-python/notebooks/credit-card-fraud-detection-with-decision-trees-and-svm.ipynb) is currently a decision-tree/SVM teaching comparison. Its potential comes from evaluation design rather than model novelty.

Before promotion, fit scaling and label-based feature selection only on training data; the current lab computes both from the full dataset. Keep the rare-event distribution in the test set. Select an operating threshold on validation data and report PR-AUC, precision/recall and the number of false alerts at a stated review budget. Add a simple baseline and compare models under the same split. This could be a faster first Kaggle publication if delivery speed becomes the priority, though its current form is less distinctive than the first three proposals.

Suggested title: **Fraud Detection Under a Review Budget: Precision, Recall and Alert Cost**.

## Keep as study material or later extensions

- Tensor operations, derivatives, linear/logistic regression, XOR, isolated activation/initialization exercises, MNIST/Fashion-MNIST and the basic Titanic notebook are useful references but weak standalone additions beside the proposed projects.
- The [diffusion-titled lab](../solutions/03-keras-and-tensorflow/notebooks/lab-implementing-diffusion-models.ipynb) currently trains a noisy-to-clean MNIST encoder/decoder using MSE. It has no timestep-conditioned iterative diffusion sampler. Present it as denoising study material unless that scope is substantially extended.
- The [advanced-transformer lab](../solutions/03-keras-and-tensorflow/notebooks/lab-building-advanced-transformers.ipynb) uses synthetic stock-price data; the title alone does not make it an applied forecasting project.
- [DPO](../solutions/11-advanced-fine-tuning/notebooks/direct-preference-optimization-dpo-using-hugging-face.ipynb), reward modeling and PPO are technically interesting later extensions of the language-model project. A preference-alignment case study needs a credible held-out preference evaluation and comparisons beyond a two-step smoke run.
- Keep component-level document loaders, splitters and generic RAG demonstrations in the study archive. Mevzuat RAG already has its own separate project scope.

## Preparing selected notebooks for Kaggle

Create newly authored experiment notebooks with source attribution and documented data/model provenance. Keep the IBM reference and assessment archive private as described in [attribution](ATTRIBUTION.md). The public case studies should explain the problem, experimental choices, measured results, errors and limitations in the owner's own project narrative.

Each candidate needs a clean top-to-bottom Kaggle run with versioned inputs and saved outputs. Kaggle's [notebook documentation](https://www.kaggle.com/docs/notebooks) describes its fresh `Save & Run All` execution and versioned environments. Local bounded execution does not establish that result.

The current official [Kaggle image template](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl) targets Python 3.13 paths, whereas this collection's legacy NLP profile uses Python 3.11, PyTorch 2.2.2 and torchtext 0.17.2. The language-model project needs a compatibility pass for the chosen Kaggle image, including removal/replacement of the old torchtext-dependent data path. Final runtime versions must be recorded in the actual Kaggle version used.

The selection assessment above preceded implementation/publication. The later [Kaggle release](KAGGLE_RELEASE.md) adds whole-split image experiments and a 512-record LoRA experiment; the original course foundations remain study material.

## Later approved second batch

Fraud was promoted from reserve, and translation, CartPole and the SFT/DPO extension were added after a further source review. All four have completed public Kaggle runs; see [the second-batch release](SECOND_BATCH.md). Earlier proposed/pending assessments above are historical, not current publication status.
