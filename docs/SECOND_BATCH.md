# Additional portfolio experiments

The owner approved the second batch on 26 November 2026 after the first three public Kaggle projects. This adds three standalone studies plus a preference-alignment extension of instruction tuning. Mevzuat RAG remains separate; course reference/assessment notebooks and first-batch publication artifacts are preserved.

| Study | Local source | Kaggle notebook | Published experiment budget |
|---|---|---|---|
| German-English translation | [Notes/code/notebook](../projects/translation/README.md) | [Open](https://www.kaggle.com/code/kerem0zcan/german-english-translation-gru-vs-transformer) | All retained official Multi30k pairs, six epochs per fresh GRU/Transformer, one seed |
| Fraud review budget | [Notes/code/notebook](../projects/fraud/README.md) | [Open](https://www.kaggle.com/code/kerem0zcan/fraud-detection-under-a-100-alert-review-budget) | Complete deduplicated chronological60/ 20/ 20 partitions;100-alert batch budget |
| CartPole target networks | [Notes/code/notebook](../projects/cartpole/README.md) | [Open](https://www.kaggle.com/code/kerem0zcan/cartpole-dqn-target-networks-across-three-seeds) | Two configurations x three training seeds x20,000 environment steps;20 test resets per seed |
| SFT versus DPO | [Extension notebook](../projects/instruction_tuning/preference_alignment.ipynb) / [Code](../projects/instruction_tuning/preference_alignment.py) | [Open](https://www.kaggle.com/code/kerem0zcan/sft-vs-dpo-with-frozen-reference-and-reload-proof) |256 training /64 validation /64 test preference pairs;128 real updates per stage |

## Evidence and current execution

All four standalone notebooks completed bounded profiles in fresh local kernels. Evidence is under `reports/second-batch-notebooks/`. Fraud uses all retained transactions even in the local profile. Translation, RL and alignment use explicit small neural pilots there; separate larger local fraud/RL runs are under `reports/second-batch-local/`.

Private Kaggle validation and public execution proceed independently per notebook. `reports/kaggle-second-batch/private-uploads.json`, `private-validation.json`, `public-uploads.json` and `status.json` retain the actual stage receipts/status. A final `release.json` is written only after all four public versions complete and source, identity, visibility, runtime freshness, split counts and reload evidence are checked. Until that ledger exists, upload acceptance is not a completed release.

The first-batch ledger remains `reports/kaggle/release.json`. It is not overwritten or expanded to imply earlier verification of new experiments.

## Data and runtime provenance

[Pinned input downloads](../provenance/second-batch-inputs.json) bind provider URLs to SHA-256 values. Multi30k and cleaned Argilla preference data use immutable upstream revisions. No downloaded data, base-model weights, trained adapter, joblib model or policy checkpoint is persisted as a public Kaggle output: those live under `/kaggle/temp`; exports contain JSON evidence, plots and fresh notebook outputs. Provider/model conditions and AI assistance are disclosed in each notebook.

Translation avoids torchtext/spaCy and uses a common Unicode tokenizer plus train-only vocabularies. Preference alignment uses direct Torch/PEFT DPO math, no TRL trainer, with frozen SFT-reference log probabilities and dropout disabled consistently. It trains a new preference-domain SFT adapter on the same OPT-350M architecture; it does not relabel the previous CodeAlpaca adapter as a new result. SFT remains eligible when DPO validation does not improve. Human response quality and generated-code correctness remain unmeasured.

## Reproduce locally

Use the core Python3.11 environment for fraud, CartPole and translation; use the validated legacy OPT/PEFT environment for alignment. Additional direct dependencies are in the second-batch requirements profiles. Neural notebook pilots use `SECOND_BATCH_FULL=0`; the Kaggle metadata uses the documented larger budget by default.

```bash
.venv/bin/python scripts/build_second_batch.py
.venv/bin/python scripts/run_second_batch.py --project fraud
.venv/bin/python scripts/run_second_batch.py --project cartpole
.venv/bin/python scripts/run_second_batch.py --project translation
.venv-legacy/bin/python scripts/run_second_batch.py --project preference_alignment
.venv/bin/python scripts/check_second_batch.py
.venv/bin/python -m unittest discover -s tests -p test_second_batch.py -v
.venv/bin/python scripts/check_repository.py
```

Do not regenerate a published notebook as private and push it inadvertently. `--project` limits generation to a single new notebook; `--public` is intended only after private validation. Authentication stays outside notebooks/reports. Local host and upstream compiler roots in stderr are redacted without changing executable cells or measured numerical outputs.

## Interpretation

The local translation pilot's near-zero BLEU is pipeline evidence only. The later corpus-wide Kaggle results below replace that pilot for the reported experiment. The local three-seed RL run favored the lagged target in this configuration; preserve the online-target collapse and report seed variance. Small SFT/DPO pilots need larger held-out comparisons; DPO loss, likelihood preference rates and training changes are not human response win rates. Fraud's natural test prevalence and actual alert counts are more informative than overall accuracy.

## Completed public release - 26 November 2026

All four version-2 notebooks are COMPLETE and public according to returned Kaggle metadata. Published source cells match the authored notebooks; the [release ledger](../reports/kaggle-second-batch/release.json) binds every cloud artifact to its checksum. [Local notebook evidence](../reports/second-batch-notebooks) uses separate bounded profiles and cannot satisfy the full-publication gate.

| Final public result | Measured value |
|---|---|
| Translation, all 1,000 test pairs | GRU corpus BLEU **11.09**; Transformer **20.61** |
| Fraud, 56,746 later transactions / 74 frauds | Logistic AP **0.747**; top 100 reviews: **58 true /42 false alerts** |
| CartPole, three training seeds / 20 test resets each | Online-target mean **9.4**; lagged-target mean **235.5**; per-seed data/variation saved |
| SFT/DPO, 64 held-out pairs | DPO loss **0.693  -> 0.685**; chosen-response NLL **1.966  -> 2.022** |

The DPO candidate passed validation selection and changed the held-out preference objective, but worsened chosen-response NLL. Its length-normalized likelihood preference rate moved from 51.56% to 53.13%, a one-pair change out of 64; no robust human response improvement is established. Both adapters reload with maximum probe-logit difference 0.0. Truncation counts are printed and saved. The original SFT baseline remains available.

The fraud frozen validation cutoff produced 112 later test alerts, versus 100 under the separate ranking policy. Keep that workload distinction visible. Translation capacities differ and only one seed was used; CartPole is a simple simulation with a bounded training budget. The negative online-target result is retained.

The [five-page supplement](../output/pdf/ibm-ai-engineering-additional-projects.pdf) contains these final cloud values and clickable notebook links. It is separate from the original three-project PDF. No GitHub commit/push or LinkedIn post was performed. Anonymous browser rendering is not claimed; identity/public visibility and version-specific completion were verified through the Kaggle API.

Validation: 56 tests in the main environment (55 pass, one PEFT-specific skip); the real SFT/DPO local notebook ran in the complementary PEFT environment and all four full Kaggle versions completed. Ruff/format, credential/path scanning, reference/solution provenance, source equality, whole-data budgets, manifest accounting and reload checks pass. The Python review found no remaining high/medium issues after fixes.
