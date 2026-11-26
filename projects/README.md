# Selected portfolio projects

Three separate experiments were added after the project selection and correctness audit. The IBM reference/worked notebooks remain study material; these modules and standalone notebooks implement the corrected portfolio paths with AI assistance.

| Project | Open | Local verification |
|---|---|---|
| Satellite CNN vs spatial CNN–Transformer | [Notebook](satellite/notebook.ipynb) · [Code](satellite/project.py) · [Notes](satellite/README.md) | Bounded training, source-scene holdout, actual sample metrics, selected-checkpoint reload |
| Instruction tuning with LoRA | [Notebook](instruction_tuning/notebook.ipynb) · [Code](instruction_tuning/project.py) · [Notes](instruction_tuning/README.md) | Fresh base/adapter outputs, real optimizer updates, grouped holdout and independent adapter reload |
| Aircraft classification and caption reliability | [Notebook](aircraft/notebook.ipynb) · [Code](aircraft/project.py) · [Notes](aircraft/README.md) | Correct VGG preprocessing, all remainder samples, selected classifier reload and separate BLIP outputs |

See the [correctness audit](../docs/SELECTED_PROJECT_AUDIT.md) for what was found, corrected and still requires verification. [Saved pilot evidence](../reports/selected-projects) is separate from the 130 course-lab smoke passes. Small pilots do not establish useful accuracy, robust generalization or an adapter improvement. All three subsequently passed private Kaggle runs and have public notebook versions. See the [Kaggle release](../docs/KAGGLE_RELEASE.md) for the final cloud execution ledger, runtime records and larger evaluation scope.

The notebook files contain the shared helpers and project code, so they do not depend on imports from this repository when copied to Kaggle. Attach the intended data, cache the required model weights or explicitly allow downloads, select the compatible runtime, and then run the notebook from top to bottom. Preserve source/data/model attribution; the course archive and model weights are not included in the notebook bundle.

## Repeat local verification

Use the existing core environment for the image projects and the legacy environment for the tested local OPT/PEFT combination. The new instruction-tuning implementation does not import torchtext, datasets or TRL.

```bash
.venv/bin/python scripts/build_project_notebooks.py --check
.venv/bin/python scripts/run_selected_projects.py --project satellite
.venv/bin/python scripts/run_selected_projects.py --project aircraft
.venv-legacy/bin/python scripts/run_selected_projects.py --project instruction_tuning
.venv/bin/python scripts/check_selected_projects.py
.venv/bin/python -m unittest discover -s tests -p test_selected_projects.py -v
.venv-legacy/bin/python -m unittest discover -s tests -p test_selected_projects.py -v
```

Each runner uses a fresh kernel, defaults to a bounded pilot, records source/split hashes and exports the actual executed notebook. Dataset files and trained checkpoint weights stay in ignored `.lab-cache/`. Full runs should use a separate output directory; example commands are in each project's notes. The current dependency profiles are documented in [setup](../docs/SETUP.md); The separate public notebooks under `kaggle/` use the provider download and compatible runtime paths measured on Kaggle.

## Additional studies and alignment extension

A second approved batch adds [German-English translation](translation/README.md), [fraud review budgets](fraud/README.md), [CartPole target-network evaluation](cartpole/README.md), and [SFT versus DPO](instruction_tuning/preference_alignment.ipynb) as an extension of the language-model project. See [second-batch scope and execution evidence](../docs/SECOND_BATCH.md). The initial three published case studies remain intact.
