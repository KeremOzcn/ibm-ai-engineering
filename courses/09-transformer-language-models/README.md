# 09 · Generative AI Language Modeling with Transformers

Self-attention, positional encoding, BERT-style encoders and causal decoders.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- Attention mixes value vectors according to query-key similarity. Multiple heads learn different projections.
- Position information is needed because self-attention alone does not encode sequence order.
- Encoder models support bidirectional context; a causal decoder masks future tokens. Their training objectives differ.
- Attention masks, padding masks and label shifts have distinct roles. Inspect them with a short synthetic sequence.

## Working model

Attention(Q,K,V) = softmax(QKᵀ / √dₖ + mask)V. A causal mask excludes keys at positions later than the query position.

## Start with these labs

- [Self-Attention and Positional Encoding](notebooks/self-attention-and-positional-encoding.ipynb) — Attention and positions.
- [Pretraining BERT Models](notebooks/pretraining-bert-models.ipynb) — Encoder models.
- [Decoder Causal Language Models](notebooks/decoder-causal-language-models.ipynb) — Decoder models.

## Revisit with a purpose

- Compute a small attention matrix by hand and confirm each row sums to one.
- Compare a BERT masked-token target with a decoder next-token target.
- Review the classifier and translation notebooks, identifying the task-specific output heads.

## Common mistakes

- Mask shape errors can silently allow future-token leakage.
- Applying softmax over the wrong axis changes the computation.
- Several collected transformer exercise cells are intentionally incomplete; the audit lists their locations.

## Check your understanding

- Why scale attention scores by √dₖ?
- What changes when a decoder sees future tokens during training?
- How does sequence length affect attention memory?

## Collected material

6 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [A Transformer Model for Language Translation](notebooks/a-transformer-model-for-language-translation.ipynb) | Reference; 2 syntax blanks | [Worked copy](../../solutions/09-transformer-language-models/notebooks/a-transformer-model-for-language-translation.ipynb) |
| [Applying Transformers for Classification](notebooks/applying-transformers-for-classification.ipynb) | Reference; 1 syntax blanks | [Worked copy](../../solutions/09-transformer-language-models/notebooks/applying-transformers-for-classification.ipynb) |
| [Data Loading and Text Processing for BERT](notebooks/data-loading-and-text-processing-for-bert.ipynb) | Reference | [Worked copy](../../solutions/09-transformer-language-models/notebooks/data-loading-and-text-processing-for-bert.ipynb) |
| [Decoder Causal Language Models](notebooks/decoder-causal-language-models.ipynb) | Reference | [Worked copy](../../solutions/09-transformer-language-models/notebooks/decoder-causal-language-models.ipynb) |
| [Pretraining BERT Models](notebooks/pretraining-bert-models.ipynb) | Reference | [Worked copy](../../solutions/09-transformer-language-models/notebooks/pretraining-bert-models.ipynb) |
| [Self-Attention and Positional Encoding](notebooks/self-attention-and-positional-encoding.ipynb) | Reference | [Worked copy](../../solutions/09-transformer-language-models/notebooks/self-attention-and-positional-encoding.ipynb) |
