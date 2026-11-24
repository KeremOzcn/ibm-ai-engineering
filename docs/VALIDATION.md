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

Five importer regression tests cover unchanged source files, duplicate consolidation, failed-publication rollback and retry, modified source rejection, path traversal rejection, and existing-collection protection.

The integrity checker verifies source mapping, curated content, clean reference outputs, notebook schemas, current generated pages, local navigation, companion execution state and report hashes. It also searches common credential patterns and personal filesystem paths; that scan is not a full security audit.

The [GitHub Actions workflow](../.github/workflows/validate.yml) runs collection checks and re-executes companions in separate Ubuntu/Python 3.11 jobs. Successful local execution and successful GitHub execution are separate evidence; consult the actual workflow run for its result.

No cloud-provider notebook, TensorFlow training run, full course solution set, course submission, certificate credential or Mevzuat RAG experiment was validated as part of this repository preparation.
