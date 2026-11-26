# Fresh base-versus-LoRA instruction tuning

This new implementation replaces the [course demonstration's](../../solutions/11-advanced-fine-tuning/notebooks/instruction-tuning-with-llms.ipynb) supplied prediction files and commented-out training path with real local training and paired evaluation.

- Base: `facebook/opt-350m`; the resolved immutable revision is recorded and reused for the independent reload.
- Input: the course CodeAlpaca JSON/JSONL file. Only nonempty no-context instructions/answers are eligible. Normalized instruction groups are split 80/10/10 before sample selection.
- Prompt and padding labels are masked with `-100`. Real EOS remains supervised even if EOS is also used as the padding ID. A truncated answer is not given an invented EOS; truncation coverage is recorded.
- Training updates only the LoRA parameters. Validation response loss selects the saved adapter; the final test responses do not participate in selection.
- The adapter must change after optimization. Loading the saved adapter into a fresh base at the same revision must reproduce selected probe logits within the declared tolerance.
- Base/adapter generations use identical held-out instructions and deterministic decoding. Response loss is weighted by actually supervised next-token counts.

The default pilot selects 32 training examples, uses two optimizer steps (four processed training examples), and evaluates four validation/four test examples. It is evidence of a working training/reload path, not a useful model or robust improvement. Generated-code task correctness and a meaningful memory/quality trade-off study remain unmeasured. No course training logs or prediction pickles are loaded.

```bash
.venv-legacy/bin/python -m projects.instruction_tuning.project \
  --data-file .lab-cache/11-advanced-fine-tuning/instruction-tuning-with-llms/CodeAlpaca-20k.json

# Separately budgeted complete training; evaluation still uses up to 64 eligible rows per holdout:
.venv-legacy/bin/python -m projects.instruction_tuning.project \
  --data-file PATH_TO_CODEALPACA_JSON \
  --full --output .lab-cache/full-projects/instruction_tuning
```

The local runs use cached model/tokenizer files. Use `--allow-downloads` only when downloading them is intended. No model weights are published by the runner. Open the [standalone notebook](notebook.ipynb) or [saved pilot evidence](../../reports/selected-projects/instruction_tuning).

## Kaggle release

The separately packaged [LoRA Instruction Tuning and Verified Reload](../../docs/KAGGLE_RELEASE.md) notebook includes provider-hosted data acquisition, saved runtime/source evidence and the larger evaluation budget. The local notebook above keeps its small pilot default. See the release record for public links and final cloud results.

## Preference-alignment extension

The [new SFT/DPO code](preference_alignment.py) and [standalone notebook](preference_alignment.ipynb) use real SFT and DPO updates on a grouped cleaned preference dataset, frozen SFT reference log probabilities and fresh selected-adapter reloads. This is a new preference-domain experiment on OPT-350M; it does not reuse or relabel the prior CodeAlpaca adapter. The baseline is retained if DPO validation does not improve. See [second-batch scope/evidence](../../docs/SECOND_BATCH.md).
