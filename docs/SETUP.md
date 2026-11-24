# Environment setup and compatibility

## Verified companion environment

The three companions and repository tooling were run in an isolated Python 3.11 environment. Exact direct package versions and the observed platform are recorded in [the execution report](../reports/companion-execution.json). These are pinned direct dependencies, not a universal lockfile or a promise that every course lab runs in this environment.

From the repository root on macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On Linux or Windows, install the CPU PyTorch build first to avoid downloading CUDA dependencies for these examples:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install and verify:

```bash
python -m pip install -r requirements/companions.txt -r requirements/quality.txt
python scripts/check_repository.py
python scripts/run_companions.py
python scripts/check_repository.py
```

The runner creates a temporary kernel specification pointing to the Python interpreter that launched it. It does not register a persistent kernel or execute any course notebook. It updates the three saved companion outputs and their execution report. Dataset creation, training and retrieval are local; dependency installation needs internet access.

For interactive exploration, optionally install JupyterLab in the same environment:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

## Course reference notebooks

| Course family | Main tools | Execution considerations |
|---|---|---|
| 01 | scikit-learn, pandas, plotting; some geospatial and boosting packages | External datasets and additional per-lab dependencies; preserved exercises |
| 02–03 | Keras / TensorFlow | Install a TensorFlow build supported by your Python/platform; image downloads and training can be substantial |
| 04–06 | PyTorch / torchvision; Keras in course 06 | Review image paths, model weights and device selection; keep framework comparisons consistent |
| 07–10 | torchtext, transformers, datasets, adapters | Legacy torchtext and mixed PyTorch pins require a notebook-specific environment |
| 11 | TRL, PEFT, transformers, sometimes bitsandbytes | API versions and accelerator assumptions differ; some labs need much more memory than the companions |
| 12–13 | LangChain integrations, watsonx, embeddings, vector stores, Gradio | Some labs require the Skills Network service environment or your own provider credentials |

The [machine-readable audit](../reports/lab-audit.json) lists detected imports, first lines of installation commands, syntax issues and environment flags for each notebook. It is an inventory, not a dependency resolver. Multi-line install commands must be read in the notebook itself.

## Known compatibility issues

- Source install cells mix versions such as PyTorch 2.8 with torchtext 0.18. Do not combine all course install commands into one environment. PyTorch documents that [torchtext development stopped at 0.18](https://docs.pytorch.org/text/stable/).
- Some labs request `+cpu` wheels and explicit Linux wheel indexes. Those commands need platform-specific review before use on macOS.
- The source collection mixes older and newer LangChain import paths. Use the notebook's intended package generation or migrate one example deliberately against [current LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview).
- Source `%pip` and `!pip` cells sometimes use `--user`, upgrades or uninstall commands. Review them before running; they are preserved source instructions, not part of this repository's verified setup.
- The collected inference notebook had saved missing-package and downstream name errors. Clearing those outputs is housekeeping, not evidence that its runtime issue was fixed.
- Syntax-valid notebooks can still contain `...`, empty exercise cells, unavailable datasets or runtime failures. Use the per-course notes and catalog rather than treating schema validation as execution proof.

## Maintaining the collection

```bash
python scripts/build_catalog.py
python scripts/build_catalog.py --check
ruff check scripts tests
ruff format --check scripts tests
python scripts/check_repository.py
```

Course pages are generated from [course definitions](../provenance/courses.json); edit that file before regenerating. Course cell content is guarded by [the source manifest](../provenance/source-manifest.json). If you intentionally change a reference lab, document the change and update its curated content hash; preserve the original source hash. Keep personal experiments separate from the reference copy.
