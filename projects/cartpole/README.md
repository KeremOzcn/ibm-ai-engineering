# CartPole DQN: online versus lagged targets

This new reinforcement-learning study compares a DQN that bootstraps against its online network with a lagged target-network variant. It corrects the source labs' inconsistent200/500-step experiment limits and preserves failed baselines as measured outcomes.

Both configurations use natural CartPole-v1 rewards, a 500-step limit, the same64-unit network, replay capacity 10,000, batch size 64, warmup 256 steps, Adam 0.001, gamma 0.99 and epsilon schedule. Updates occur every four environment steps; the lagged target synchronizes every 500 steps. Each of training seeds 11/22/33 receives 20,000 environment steps. Experience differs when the policies choose different actions; identical trajectories are not claimed.

Validation reset seeds 1000-1004 select checkpoints. Evaluate greedy selected policies on reserved reset seeds 2000-2019, separately from training/validation. Termination ends value bootstrapping; a time-limit truncation ends an episode but still bootstraps. Include all per-seed results, standard deviation across training-seed means and a random policy on the same test seeds. A fresh network must exactly reproduce a selected checkpoint's probe logits.

The small notebook profile uses1,000 steps, one training seed and five test reset seeds. The larger local run and Kaggle budget use three seeds. CartPole is a simple simulation; these results do not establish general RL robustness, a solved environment or real-world control readiness.

Open the [standalone notebook](notebook.ipynb), [code](project.py), [fresh local execution](../../reports/second-batch-notebooks/cartpole/executed.ipynb) and [second-batch release record](../../docs/SECOND_BATCH.md).

Attribution: [Farama Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and [termination/time-limit semantics](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). Experiment code was prepared with AI assistance from course concepts; no course checkpoint is reused.
