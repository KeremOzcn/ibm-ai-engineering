# 07 · Generative AI and LLMs: Architecture and Data Preparation

Tokenization, vocabulary design and data loaders for language models.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- A tokenizer maps text into model-specific token IDs. The tokenizer, vocabulary and model checkpoint must agree.
- Padding tokens align sequence lengths; an attention mask distinguishes real input from padding.
- A causal language model predicts the next token. Shift inputs and targets consistently.
- Data splitting must prevent duplicated or overlapping documents from contaminating evaluation.

## Working model

Typical token IDs and attention masks have shape (batch, sequence_length); embeddings have shape (batch, sequence_length, hidden_size).

## Start with these labs

- [Implementing Tokenization](notebooks/implementing-tokenization.ipynb) — Tokenization.
- [Creating an NLP Data Loader](notebooks/creating-an-nlp-data-loader.ipynb) — Data loading.
- [Exploring Generative AI Libraries](notebooks/exploring-generative-ai-libraries.ipynb) — Library overview.

## Revisit with a purpose

- Inspect the tokenization of punctuation, accented text and an unfamiliar word.
- Build a tiny padded batch and verify the mask by hand.
- Trace the text-to-batch transformations in the NLP data loader.

## Common mistakes

- Token count differs from word count and affects context limits and cost.
- Legacy torchtext imports have strict binary compatibility requirements.
- Some source install cells request newer torch with old torchtext; those combinations are not a validated environment.

## Check your understanding

- Why must inference use the training tokenizer?
- How should padding be excluded from loss computation?
- How can overlapping text windows leak across a split?

## Collected material

3 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Creating an NLP Data Loader](notebooks/creating-an-nlp-data-loader.ipynb) | Reference | [Worked copy](../../solutions/07-llm-data-preparation/notebooks/creating-an-nlp-data-loader.ipynb) |
| [Exploring Generative AI Libraries](notebooks/exploring-generative-ai-libraries.ipynb) | Reference | [Worked copy](../../solutions/07-llm-data-preparation/notebooks/exploring-generative-ai-libraries.ipynb) |
| [Implementing Tokenization](notebooks/implementing-tokenization.ipynb) | Reference | [Worked copy](../../solutions/07-llm-data-preparation/notebooks/implementing-tokenization.ipynb) |
