# Remaining full runs

The owner selected bounded local verification first. Every saved smoke notebook is an adapted execution check, not the prescribed experiment or a model-quality result. Run the worked source notebook interactively in the matching environment for full training. All full prescribed experiments remain unverified by this collection's current execution ledger.

The live [notebook progress table](SOLUTION_PROGRESS.md) identifies actual passes, failed attempts and unexecuted labs. [The execution ledger](../reports/solution-execution.json) contains exact source/output hashes and failure details. [The verification plan](../provenance/verification-plan.json) records explicit resource/service requirements for deferred notebooks. There are no real provider calls or course submissions in this work.

| Course | Full run still needed |
|---|---|
| 01 | Use the full dataset and full parameter grids/cross-validation. Recompute held-out metrics from the resulting pipeline. |
| 02 | Original training budgets and complete test splits; the aircraft lab additionally downloads an image archive, VGG16 and BLIP weights. |
| 03 | Original GAN/diffusion/autoencoder/transformer training, tuner trials and RL episodes; augmentation needs CIFAR-10 download; transfer projects need image archives and pretrained weights. |
| 04–05 | All original training epochs and all DataLoader batches; recompute correctly normalized losses and evaluation counts. |
| 06 | All original image batches/epochs and the intended pretrained checkpoints; compare frameworks on the same split with complete predictions. Cached course assets can be reused if their identity is checked. |
| 07 | Complete dataset/model/tokenizer downloads and the notebook-specific spaCy/NLTK resources. |
| 08–09 | Full corpora, embeddings, training budgets and checkpoints. Legacy torchtext must match its PyTorch environment; downloaded helper scripts run outside the smoke transform. |
| 10 | Complete model/dataset/checkpoint downloads, training, adapter reload and inference. A smoke loss does not establish a fine-tuning improvement. |
| 11 | Matching TRL/PEFT/Transformers APIs; supported accelerator for quantized examples; full instruction/reward/PPO/DPO training, checkpoint and comparison paths. |
| 12 | Model weights for local BERT/DPR/GPT-2 RAG; IBM service access for watsonx examples; OPENWEATHER_API_KEY for the weather tool. |
| 13 | Compatible LangChain integrations and actual watsonx embedding/generation access; full ingestion and retrieval checks on the intended corpus. Document-only labs can be checked locally. |

## Reading bounded results

The runner seeds local randomness, samples large CSVs to 5,000 rows, restricts grids to one setting/two folds, caps recognized epoch budgets, and limits each PyTorch DataLoader to 12 batches. TensorFlow bundled datasets use 512 training/256 test examples; fit receives at most 512 array examples or eight dataset batches. Some custom loops and downloaded scripts need their own review; this is not a universal compute sandbox. Per-cell timeouts bound attempts, and a timeout is recorded as a failed attempt.

Original lab metric denominators sometimes assume a complete dataset. Under the smoke profile, those printed values may be misleading even when code executes. Fashion-MNIST was corrected to count the actual evaluated examples; other smoke metrics are retained as execution output and are not portfolio performance claims. Use the three independently executed companions for the small measured workflow examples documented in [validation](VALIDATION.md).

A guarded display-only override in the seq2seq lab skips five rather than 1,000 DataLoader batches so its sequence illustration fits the 12-batch smoke iterator. The worked notebook retains the original display selection. The applied override is recorded in the executed header and ledger.

## Dataset overrides used for vision comparisons

The Keras/PyTorch comparison and CNN/Transformer integration checks use the first 64 sorted filenames in each of the two course image classes (128 actual course images). Both frameworks receive the same selected image directory. This keeps their full-corpus completeness assertions intact under the 12-batch profile. The cached archive and selected-file hashes are recorded in each run's `data_profile`; original full images remain in the ignored cache. These subsets are for runtime checks, not measured framework comparisons.

The text-generation smoke override constructs 32 shifted token windows while retaining the original model dimensions. Causal masking, cross-entropy from logits and last-position generation are corrected in the worked source and protected by focused behavior tests.
