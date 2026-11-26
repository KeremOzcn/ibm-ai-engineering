# Completion and Verification Summary

Date: 2026-11-25

This note summarizes the completion of missing exercises, bug fixes, dependency updates, and bounded local verification across the 138 worked notebooks before committing to GitHub.

## Current Portfolio State

| Metric | Count | Reference |
|---|---:|---|
| Reference notebooks (`courses/`) | 138 | [Course catalog](LAB_CATALOG.md) |
| Worked study copies (`solutions/`) | 138 | [Worked lab progress](SOLUTION_PROGRESS.md) |
| Answer & compatibility patches | 456 | [Solution patches](../provenance/solution-patches.json) |
| Syntax or unresolved stub issues | 0 | [Progress report](../reports/solution-progress.json) |
| Bounded smoke runs passed | 130 | [Execution ledger](../reports/solution-execution.json) |
| Deferred (external API key required) | 8 | [Verification plan](../provenance/verification-plan.json) |
| Executed companion notebooks | 3 | [Validation evidence](VALIDATION.md) |

## Completed Missing Exercises

1. **Waste classification transfer-learning capstone** ([worked notebook](../solutions/03-keras-and-tensorflow/notebooks/waste-classification-using-transfer-learning.ipynb)):
   Completed Tasks 1–10 (cells 11, 21, 23, 35, 37, 45, 51, 53, 58, and 60), covering TensorFlow version output, `test_generator`, `len(train_generator)`, `model.summary()`, binary cross-entropy `model.compile()`, training/validation accuracy and loss curves for both feature-extraction and fine-tuned VGG16 models, and test-image prediction plots.
2. **PyTorch differentiation practice** ([worked notebook](../solutions/04-pytorch-foundations/notebooks/differentiation-in-pytorch.ipynb)):
   Completed cell 68 to compute the partial derivative of $f = uv + (uv)^2$ at $u = 2, v = 1$.

## Code and Compatibility Fixes

- **Text generation with TensorFlow Transformers** ([worked notebook](../solutions/03-keras-and-tensorflow/notebooks/lab-implementing-transformers-for-text-generation.ipynb)):
  Fixed the second `generate_text` helper (cell 32) to slice the final token logits (`predictions[:, -1, :]`) before `tf.random.categorical`, preventing a 3D rank error.
- **Fruit classification transfer learning** ([worked notebook](../solutions/03-keras-and-tensorflow/notebooks/practice-project-fruit-classification-using-transfer-learning.ipynb)):
  Replaced the external `wget` subprocess in `download_dataset` (cell 18) with an atomic standard-library `urllib.request.urlopen` download.
- **LoRA with PyTorch** ([worked notebook](../solutions/10-fine-tuning-transformers/notebooks/lora-with-pytorch.ipynb)):
  Moved `%%capture` to line 1 of cells 89 and 143 so IPython cell magic executes properly after `_download_lab_asset` helpers.
- **Pre-training LLMs with Hugging Face** ([worked notebook](../solutions/10-fine-tuning-transformers/notebooks/pre-training-llms-with-hugging-face.ipynb)):
  Updated cell 25 to load `wikitext-2-raw-v1` from `datasets` before cell 33 exports `wikitext_dataset_train.txt` and `wikitext_dataset_test.txt`, and updated cell 69 to unpack the 3-tuple returned by `Trainer.predict(val_encoded)`.

## Environment and Bounded Smoke Adaptations

- **Dependency pins**:
  - [requirements/labs-core.txt](../requirements/labs-core.txt): added `transformers==4.45.2` (supports Course 02 BLIP captioning alongside TensorFlow/Keras).
  - [requirements/labs-nlp-legacy.txt](../requirements/labs-nlp-legacy.txt): pinned `pyarrow==17.0.0` (compatible with `numpy==1.26.4` and `datasets==2.20.0`) and added `sacrebleu==2.4.3`.
  - [requirements/labs-document-tools.txt](../requirements/labs-document-tools.txt): added `markdown==3.7` (with system `libmagic` via Homebrew for `unstructured`).
- **Non-interactive smoke overrides** ([smoke-overrides.json](../provenance/smoke-overrides.json)):
  - Scripted `input()` prompts for `exploring-generative-ai-libraries.ipynb`, `decoder-causal-language-models.ipynb`, and `direct-preference-optimization-dpo-using-hugging-face.ipynb`.
  - Bounded `data_itr` advances in `a-transformer-model-for-language-translation.ipynb` to stay within the 12-batch smoke DataLoader.
  - Guarded `fp16=torch.cuda.is_available()` and 2-step CPU limits for `SFTConfig`/`DPOConfig` (including step-2 checkpointing and `eval_loss` logging for DPO).
  - Forced CPU execution in `fine-tuning-transformers-with-pytorch-and-hugging-face.ipynb` and `reinforcement-learning-from-human-feedback-using-ppo.ipynb` to avoid a `transformers==4.45.2` + `torch==2.2.2` Apple Silicon MPS `isin_mps_friendly` error.
- **Ledger synchronization**:
  Re-ran 7 earlier Course 01 and Course 05 notebooks whose saved artifact digests had diverged after historical path redaction, and pruned [verification-plan.json](../provenance/verification-plan.json) down to the 8 notebooks that require external provider keys.

## Remaining 8 Provider-Gated Notebooks (`not-run`)

These notebooks are syntax-audited and ready, but intentionally left `not-run` because they require live external service credentials:

1. [In-Context Learning and Prompt Templates](../solutions/12-ai-agents-and-rag/notebooks/in-context-learning-and-prompt-templates.ipynb) — IBM watsonx
2. [Lab: Building AI Agents from Scratch with Python](../solutions/12-ai-agents-and-rag/notebooks/lab-building-ai-agents-from-scratch-with-python.ipynb) — `OPENWEATHER_API_KEY`
3. [LangChain](../solutions/12-ai-agents-and-rag/notebooks/langchain.ipynb) — IBM watsonx
4. [Summarize Private Documents Using RAG, LangChain, and LLMs](../solutions/12-ai-agents-and-rag/notebooks/summarize-private-documents-using-rag-langchain-and-llms.ipynb) — IBM watsonx
5. [Create and Configure a Vector Database to Store Document Embeddings](../solutions/13-rag-application-project/notebooks/create-and-configure-a-vector-database-to-store-document-embeddings.ipynb) — IBM watsonx
6. [Develop a Retriever to Fetch Document Segments Based on Queries](../solutions/13-rag-application-project/notebooks/develop-a-retriever-to-fetch-document-segments-based-on-queries.ipynb) — IBM watsonx
7. [Embed documents using watsonx's embedding model](../solutions/13-rag-application-project/notebooks/embed-documents-using-watsonx-s-embedding-model.ipynb) — IBM watsonx
8. [Put Whole Document into Prompt and Ask the Model](../solutions/13-rag-application-project/notebooks/put-whole-document-into-prompt-and-ask-the-model.ipynb) — IBM watsonx

## Verification Commands

Run before committing or when checking the repository later:

```bash
.venv/bin/python scripts/build_solutions.py --check
.venv/bin/python scripts/build_catalog.py --check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check scripts tests
.venv/bin/ruff format --check scripts tests
.venv/bin/python scripts/check_solutions.py
.venv/bin/python scripts/check_repository.py
```

## Selected-project correctness continuation

After this course-lab summary, the owner selected satellite classification, LoRA instruction tuning and aircraft classification/caption reliability. Separate corrected implementations, standalone notebooks and three fresh-kernel bounded pilots were added under [projects/](../projects/README.md). See the [correctness audit](SELECTED_PROJECT_AUDIT.md); this newer evidence is separate from the 130 course-lab smoke passes. At that pilot stage, full training, Kaggle execution and robust model-quality claims were unverified. The later [Kaggle release](KAGGLE_RELEASE.md) records whole-split image experiments, a bounded 512-record LoRA experiment and public notebook versions, with remaining quality limitations.
