# 11 · Generative AI Advanced Fine-Tuning for LLMs

Instruction tuning, reward models, PPO and direct preference optimization.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- Instruction tuning learns from prompt-response examples. Template consistency and label masking are part of the training specification.
- A reward model estimates preference from comparison data; its score is a learned proxy with its own biases.
- PPO optimizes a policy using a reward signal and a constraint against drifting too far from a reference policy.
- DPO uses chosen/rejected response pairs directly. Data quality and reference-model assumptions affect its behavior.

## Working model

A preference record contains a prompt, a chosen response and a rejected response. Keep related prompts together when forming train/evaluation partitions.

## Start with these labs

- [Instruction-Tuning with LLMs](notebooks/instruction-tuning-with-llms.ipynb) — Instruction tuning.
- [Reward Modeling](notebooks/reward-modeling.ipynb) — Reward modeling.
- [Direct Preference Optimization (DPO) Using Hugging Face](notebooks/direct-preference-optimization-dpo-using-hugging-face.ipynb) — Direct preference optimization.
- [Reinforcement Learning from Human Feedback Using PPO](notebooks/reinforcement-learning-from-human-feedback-using-ppo.ipynb) — PPO.

## Revisit with a purpose

- Inspect chosen/rejected pairs for labeling errors and near duplicates.
- Explain the role of the reference model before running PPO or DPO.
- Design a held-out preference evaluation including regressions on factual and refusal behavior.

## Common mistakes

- Optimizing a reward score can exploit weaknesses in the reward model.
- Preference wins do not establish factual accuracy or domain safety.
- TRL APIs vary by version; the course pins are reference snapshots, not a tested unified environment.

## Check your understanding

- What can a reward model get wrong?
- How does DPO use the rejected response?
- Which regressions would make you retain the original model?

## Collected material

4 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Direct Preference Optimization (DPO) Using Hugging Face](notebooks/direct-preference-optimization-dpo-using-hugging-face.ipynb) | Reference | [Worked copy](../../solutions/11-advanced-fine-tuning/notebooks/direct-preference-optimization-dpo-using-hugging-face.ipynb) |
| [Instruction-Tuning with LLMs](notebooks/instruction-tuning-with-llms.ipynb) | Reference | [Worked copy](../../solutions/11-advanced-fine-tuning/notebooks/instruction-tuning-with-llms.ipynb) |
| [Reinforcement Learning from Human Feedback Using PPO](notebooks/reinforcement-learning-from-human-feedback-using-ppo.ipynb) | Reference | [Worked copy](../../solutions/11-advanced-fine-tuning/notebooks/reinforcement-learning-from-human-feedback-using-ppo.ipynb) |
| [Reward Modeling](notebooks/reward-modeling.ipynb) | Reference | [Worked copy](../../solutions/11-advanced-fine-tuning/notebooks/reward-modeling.ipynb) |
