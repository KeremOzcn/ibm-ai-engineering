# Validation evidence

## Scope

Initial local validation was performed on 24 November 2026 using Python 3.11.15 on macOS Apple Silicon. The [execution report](../reports/companion-execution.json) records package versions, notebook hashes, executed cell counts, saved image counts and measured metrics. Running the companions again replaces that report with the new run's evidence.

## Course collection

| Check | Result |
|---|---|
| Source accounting | All 149 assets map to curated paths |
| Notebook consolidation | 147 originals → 138 course notebooks; 9 within-course repeated copies consolidated |
| Source guides | Both retained byte-for-byte |
| Notebook schema | 138 course notebooks and 3 companions validate with nbformat |
| Course content preservation | Cell types/source match recorded curated hashes, with the documented copyright-cell correction |
| Course execution | None of the 138 reference notebooks was executed during curation |
| Preserved syntax blanks | 26 cells across 14 reference notebooks, after IPython input transformation |
| Saved source errors | 10 error outputs in the collected inference lab were cleared and recorded |
| Navigation | Local collection-authored file links are checked; original third-party links and source-guide anchors are outside this check |

The [lab audit](../reports/lab-audit.json) lists code cells, exercise-marker cells, syntax issues, imports and environment flags. Exercise-marker detection is heuristic. Passing syntax checks does not establish that exercises are complete or runtime dependencies are available.

## Companion execution

All 15 nonempty companion code cells executed without error outputs. Both plotting companions contain a saved PNG figure.

| Example | Initial local result | Practical limit |
|---|---|---|
| ML pipeline | 5-fold training CV macro F1 0.9854; test macro F1 1.0000 on 45 Wine examples; dummy test macro F1 0.1905 | Small bundled dataset and one held-out split |
| PyTorch | Test accuracy 94.5% on 400 synthetic examples; exact logits after state-dictionary reload | Synthetic two-moons data, CPU workflow only |
| Retrieval | Hit@1 and Hit@3 both 5/6; both unrelated unsupported queries returned no hits | Six handcrafted questions over six short notes; no generated-answer evaluation |

The retrieval miss is retained and explained in the notebook. TF-IDF does not normalize inflected forms or understand synonyms. The two easy unsupported queries do not establish reliable abstention when an irrelevant query shares vocabulary with the corpus.

## Repeat the checks

```bash
ruff check scripts tests
ruff format --check scripts tests
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
python scripts/build_catalog.py --check
python scripts/check_repository.py
python scripts/run_companions.py
python scripts/check_repository.py
```

Importer regression tests cover unchanged source files, duplicate consolidation, failed-publication rollback and retry, modified source rejection, path traversal rejection, and existing-collection protection.

The integrity checker verifies source mapping, curated content, clean reference outputs, notebook schemas, current generated pages, local navigation, companion execution state and report hashes. It also searches common credential patterns and personal filesystem paths; that scan is not a full security audit.

The [GitHub Actions workflow](../.github/workflows/validate.yml) runs collection checks and re-executes companions in separate Ubuntu/Python 3.11 jobs. Successful local execution and successful GitHub execution are separate evidence; consult the actual workflow run for its result.

No provider-backed notebook execution, full prescribed course experiment, course submission, certificate credential or Mevzuat RAG experiment is verified by this work. Bounded TensorFlow/PyTorch runs are documented separately below.

## Worked lab verification

The 25 November continuation adds 138 generated worked copies, guarded answer/compatibility patches, exercise-review decisions and actual bounded notebook runs. The current counts are generated in [solution progress](SOLUTION_PROGRESS.md); the [ledger](../reports/solution-execution.json) records successes and failed attempts individually.

| Evidence | What is checked |
|---|---|
| Answer preparation | Every inventory location is patched or explicitly reviewed; patch source hashes match; zero syntax/stub issues in generated worked copies. |
| Reference preservation | The original course content hashes remain guarded independently of the worked copies. |
| Saved execution | Each current pass points to its exact worked-source hash and saved executed notebook; schema, executed code cells, absence of error outputs, and output-file hashes are checked. |
| Profile identity | New runs record transformed-code hashes, package versions, explicit smoke overrides and any cached-data manifest. Earlier runs have saved executable code but incomplete environment capture. |
| Stale evidence | Old outputs remain historical evidence when a source changes; current-pass counts require the current worked-source hash. |
| Regression behavior | Tests cover importer rollback/provenance; stub detection; bounded-budget transformations; stale/corrupt proof; Fashion input shape; multi-page PDF translation; selective unfreezing; and isolated retrieval corpora. |

The notebook runner uses asynchronous kernel management, creates an interpreter-specific temporary kernel, cleans it up, and returns a failing exit status when any selected notebook fails. Reports are written after each attempted notebook. Run one runner process at a time. Output paths are redacted before saving; private caches and environments remain ignored.

The Python reviewer examined the solution tooling and selected answer corrections. Review and regression checks do not establish every answer's semantic correctness. Full experiments, large asset/model runs and provider execution remain listed in [remaining full runs](FULL_RUNS.md).

```bash
python scripts/build_solutions.py
python scripts/build_solutions.py --check
python scripts/check_solutions.py
python scripts/check_repository.py
```
