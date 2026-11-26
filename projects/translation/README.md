# German-to-English translation: GRU versus Transformer

This independently authored comparison extends the course's recurrent/Transformer translation concepts. Both models train from fresh weights; supplied course checkpoints and legacy torchtext/spaCy loaders are not used.

- Inputs: official Multi30k train, validation and 2016 test sentence pairs, pinned to a repository revision and download checksums in [the input manifest](../../provenance/second-batch-inputs.json).
- Keep at most 48 tokens per side. Remove repeated normalized German source sentences with test/validation priority. Fit vocabularies only on the actual training records.
- Match initial source/target embeddings, training order, batch size, learning rate and epoch count. The GRU context-vector decoder and two-layer Transformer have different capacities/computation; this is not an isolated attention ablation.
- Mask decoder future tokens and source/target padding. Weight validation NLL by actual supervised next-token counts. Restore the validation-selected checkpoint and verify a fresh model's logits.
- Generate the entire retained test set greedily and report case-insensitive corpus SacreBLEU with its signature. Save all generated hypotheses by source ID; quote only a few input/reference examples.

The bounded local notebook profile uses512 training and64 validation/test pairs for two epochs. It verifies the pipeline; its near-zero BLEU does not demonstrate a useful translator. The Kaggle budget uses every retained pair and six epochs. One seed and short captions limit conclusions.

Open the [standalone notebook](notebook.ipynb), [code](project.py), [fresh local execution](../../reports/second-batch-notebooks/translation/executed.ipynb) and [second-batch release record](../../docs/SECOND_BATCH.md).

Data/research attribution: [Multi30k](https://github.com/multi30k/dataset). Provider conditions remain applicable. Code and documentation were prepared with AI assistance; no IBM assessment notebook or source dataset is republished.
