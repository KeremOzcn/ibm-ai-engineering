# 10 · Generative AI Engineering and Fine-Tuning Transformers

Inference, pretraining, task adaptation, adapters and LoRA.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- Pretraining learns general representations; fine-tuning adapts parameters to a task or domain. Evaluate both on the same held-out examples.
- Adapters add trainable modules while most base weights stay frozen. LoRA learns a low-rank update to selected weight matrices.
- Parameter efficiency reduces trainable state but does not eliminate base-model memory or activation cost.
- A reproducible adapter result requires the base revision, tokenizer, adapter checkpoint, generation settings and an exact reload test.

## Working model

LoRA represents ΔW = (α/r)BA, where B has shape (out_features,r) and A has shape (r,in_features). Its trainable parameter count is r·(in_features + out_features).

## Start with these labs

- [Loading Models and Inference with Hugging Face Inferences](notebooks/loading-models-and-inference-with-hugging-face-inferences.ipynb) — Inference and model loading.
- [Adapters in PyTorch](notebooks/adapters-in-pytorch.ipynb) — Adapters.
- [LoRA with PyTorch](notebooks/lora-with-pytorch.ipynb) — Low-rank adaptation.

## Revisit with a purpose

- Count trainable versus total parameters in the adapter and LoRA labs.
- Compare the base model and adapter with identical prompts and decoding settings.
- Reload the saved adapter in a fresh process before describing it as a reusable artifact.

## Common mistakes

- Lower training loss alone does not prove a better downstream model.
- The collected Hugging Face inference notebook had 10 saved error outputs; they were cleared and recorded in provenance.
- Quantization, padding and tokenizer changes can confound base-versus-adapter comparisons.

## Check your understanding

- Which memory costs remain when base weights are frozen?
- What makes an adapter evaluation a fair comparison?
- Why does checkpoint reload matter separately from training success?

## Collected material

6 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Adapters in PyTorch](notebooks/adapters-in-pytorch.ipynb) | Reference | [Worked copy](../../solutions/10-fine-tuning-transformers/notebooks/adapters-in-pytorch.ipynb) |
| [Fine-Tuning Transformers with PyTorch and Hugging Face](notebooks/fine-tuning-transformers-with-pytorch-and-hugging-face.ipynb) | Reference | [Worked copy](../../solutions/10-fine-tuning-transformers/notebooks/fine-tuning-transformers-with-pytorch-and-hugging-face.ipynb) |
| [Loading Models and Inference with Hugging Face Inferences](notebooks/loading-models-and-inference-with-hugging-face-inferences.ipynb) | Reference | [Worked copy](../../solutions/10-fine-tuning-transformers/notebooks/loading-models-and-inference-with-hugging-face-inferences.ipynb) |
| [LoRA with PyTorch](notebooks/lora-with-pytorch.ipynb) | Reference | [Worked copy](../../solutions/10-fine-tuning-transformers/notebooks/lora-with-pytorch.ipynb) |
| [Pre-Training and Fine-Tuning with PyTorch](notebooks/pre-training-and-fine-tuning-with-pytorch.ipynb) | Reference | [Worked copy](../../solutions/10-fine-tuning-transformers/notebooks/pre-training-and-fine-tuning-with-pytorch.ipynb) |
| [Pre-training LLMs with Hugging Face](notebooks/pre-training-llms-with-hugging-face.ipynb) | Reference | [Worked copy](../../solutions/10-fine-tuning-transformers/notebooks/pre-training-llms-with-hugging-face.ipynb) |
