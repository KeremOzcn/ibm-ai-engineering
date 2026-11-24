# Executed companion notebooks

These three small examples connect course ideas to inspectable workflows. They were newly created with AI assistance during repository curation and are separate from the original IBM exercises. Each has saved outputs, explicit checks and a machine-readable result in [the execution report](../reports/companion-execution.json).

| Notebook | Main concepts | Data |
|---|---|---|
| [01 · ML pipeline and evaluation](01-ml-pipeline-and-evaluation.ipynb) | Preprocessing within CV, baseline, fixed test partition | Bundled scikit-learn Wine dataset |
| [02 · PyTorch training and checkpoint](02-pytorch-training-and-checkpoint.ipynb) | Training loop, validation selection, exact reload | Locally generated two-moons classification |
| [03 · Local retrieval and source tracing](03-local-retrieval-and-source-tracing.ipynb) | Chunking, TF-IDF ranking, source IDs, unsupported queries | Six newly written short study notes |

Use the [setup instructions](../docs/SETUP.md), then run `python scripts/run_companions.py` from the repository root. The runner executes all three on CPU using the interpreter that launched it and replaces their saved outputs and report.

The first two examples produce small classification results. The third evaluates only retrieval; it does not generate answers. These examples are educational checks rather than broad benchmarks, production systems or submitted course capstones.
