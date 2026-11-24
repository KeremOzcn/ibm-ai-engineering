# Study roadmap

Use this collection as an active reference. Start from a question, open the relevant course notes, inspect a lab, then reproduce a small result in your own experiment. The time boxes below are optional revision sessions, not estimates of the certificate's duration.

## Six revision sessions

| Session | Material | Produce |
|---|---|---|
| 1 · Model evaluation | [Course 01](../courses/01-machine-learning-with-python/README.md), [ML companion](../companions/01-ml-pipeline-and-evaluation.ipynb) | A split diagram, a baseline and an explanation of leakage |
| 2 · Gradients and training | [Course 02](../courses/02-deep-learning-with-keras/README.md), [Course 04](../courses/04-pytorch-foundations/README.md), [PyTorch companion](../companions/02-pytorch-training-and-checkpoint.ipynb) | A hand-worked gradient and a checkpoint reload result |
| 3 · Vision and transfer learning | [Course 03](../courses/03-keras-and-tensorflow/README.md), [Course 05](../courses/05-deep-learning-with-pytorch/README.md), [Course 06](../courses/06-computer-vision-capstone/README.md) | A fair CNN-versus-transfer-learning experiment design |
| 4 · Language representations | [Course 07](../courses/07-llm-data-preparation/README.md), [Course 08](../courses/08-foundational-nlp-models/README.md), [Course 09](../courses/09-transformer-language-models/README.md) | Token, mask and attention shapes for a short example |
| 5 · Fine-tuning and alignment | [Course 10](../courses/10-fine-tuning-transformers/README.md), [Course 11](../courses/11-advanced-fine-tuning/README.md) | A base-versus-adapter evaluation protocol and parameter count |
| 6 · Retrieval and applications | [Course 12](../courses/12-ai-agents-and-rag/README.md), [Course 13](../courses/13-rag-application-project/README.md), [retrieval companion](../companions/03-local-retrieval-and-source-tracing.ipynb) | A traced evidence packet and a taxonomy of retrieval/answer failures |

For a 45–60 minute session, spend roughly 10 minutes recalling the concept, 20 minutes tracing a notebook, 15 minutes modifying a small experiment, and 5 minutes recording what changed and what remains unclear.

## How to revisit a lab

1. Read its status in the [catalog](LAB_CATALOG.md). Reference notebooks may contain blanks, external downloads and outdated install cells.
2. State the expected input, output, shapes and success metric before running anything.
3. Use an isolated environment suited to that notebook. Restart the kernel after dependency changes.
4. Write down the baseline and change one meaningful factor at a time.
5. Record dataset/split, package versions, seed, compute budget, metric and observed failure cases.
6. Explain the result without looking at the notebook. If the explanation depends on a hidden assumption, add it to your notes.

## Compact experiment record

```text
Question:
Source lab / concept:
Data and split rule:
Baseline:
Single change:
Environment / seed / compute budget:
Metric and result:
Failure cases:
Conclusion and next experiment:
```

## Interview rehearsal

Explain these with a small example: leakage in scaling, the difference between validation and test data, logits versus probabilities, train/eval behavior, a causal attention mask, trainable versus total parameters, and retrieval quality versus answer faithfulness. Each course page supplies more targeted questions.

When a result cannot be reproduced or explained, describe the limitation directly. A clear account of a failed experiment is more useful than an unsupported performance claim.
