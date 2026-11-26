# 08 · Gen AI Foundational Models for NLP and Language Understanding

N-gram models, word embeddings, feedforward language models and sequence-to-sequence models.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- An n-gram model estimates the next token from a fixed context. Smoothing helps when a context is unseen.
- Word embeddings learn continuous representations; similarity reflects the training objective and corpus.
- Sequence models track order. A sequence-to-sequence model maps an input sequence to an output sequence of potentially different length.
- Document classification aggregates token-level information into a prediction; padding and pooling choices affect the representation.

## Working model

A causal sequence likelihood factors as P(x₁…xₜ) = ∏ P(xᵢ | x₁…xᵢ₋₁). Perplexity is exp(mean token negative log-likelihood).

## Start with these labs

- [Building a Language Model Using Histogram N-Gram Analysis](notebooks/building-a-language-model-using-histogram-n-gram-analysis.ipynb) — N-gram language modeling.
- [Integrating Word2Vec Part1](notebooks/integrating-word2vec-part1.ipynb) — Word2Vec.
- [Sequence-to-Sequence RNN Models: Translation Task](notebooks/sequence-to-sequence-rnn-models-translation-task.ipynb) — Sequence-to-sequence translation.

## Revisit with a purpose

- Compare unigram and bigram predictions on a tiny corpus with unseen contexts.
- Explore nearest neighbors for Word2Vec embeddings and identify corpus bias.
- Trace teacher forcing through the translation notebook.

## Common mistakes

- A low training perplexity can reflect memorization.
- Averaging embeddings discards order and can confuse negation.
- Evaluate translation on held-out sequences rather than cherry-picked generated examples.

## Check your understanding

- What information does bag-of-words discard?
- Why is teacher forcing different from inference?
- When can embedding similarity be misleading?

## Collected material

6 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Building a Language Model Using Histogram N-Gram Analysis](notebooks/building-a-language-model-using-histogram-n-gram-analysis.ipynb) | Reference | [Worked copy](../../solutions/08-foundational-nlp-models/notebooks/building-a-language-model-using-histogram-n-gram-analysis.ipynb) |
| [Building and Training a Feedforward Neural Network for Language Modeling](notebooks/building-and-training-a-feedforward-neural-network-for-language-modeling.ipynb) | Reference | [Worked copy](../../solutions/08-foundational-nlp-models/notebooks/building-and-training-a-feedforward-neural-network-for-language-modeling.ipynb) |
| [Document Classification with Word Embeddings](notebooks/document-classification-with-word-embeddings.ipynb) | Reference | [Worked copy](../../solutions/08-foundational-nlp-models/notebooks/document-classification-with-word-embeddings.ipynb) |
| [Integrating Word2Vec Part1](notebooks/integrating-word2vec-part1.ipynb) | Reference | [Worked copy](../../solutions/08-foundational-nlp-models/notebooks/integrating-word2vec-part1.ipynb) |
| [Integrating Word2Vec Part2](notebooks/integrating-word2vec-part2.ipynb) | Reference | [Worked copy](../../solutions/08-foundational-nlp-models/notebooks/integrating-word2vec-part2.ipynb) |
| [Sequence-to-Sequence RNN Models: Translation Task](notebooks/sequence-to-sequence-rnn-models-translation-task.ipynb) | Reference | [Worked copy](../../solutions/08-foundational-nlp-models/notebooks/sequence-to-sequence-rnn-models-translation-task.ipynb) |
