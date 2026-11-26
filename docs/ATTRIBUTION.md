# Attribution and provenance

## Course material

The reference notebooks and two project guides were collected from the [IBM AI Engineering Professional Certificate on Coursera](https://www.coursera.org/professional-certificates/ai-engineer) and its Skills Network lab workspace. The course notebooks retain their original author, contributor, dataset attribution and copyright passages. Many explicitly state that IBM reserves all rights.

No blanket open-source license is applied to the downloaded material. This repository is a private study collection. A future public edition should review redistribution permissions and omit assessment solutions unless sharing is expressly permitted; [Coursera's Honor Code](https://www.coursera.org/about/terms) restricts unauthorized sharing of assessment resources and solutions.

## Curation performed here

- Organized course material under descriptive paths and filenames.
- Consolidated nine within-course duplicates with identical cell types and source text, ignoring output and environment metadata.
- Added a visible reference status and navigation header to each notebook.
- Normalized notebook schemas and cell IDs, removed transient metadata, and cleared saved outputs and execution counts.
- Converted one copyright-only cell from code to markdown; its notice is preserved.
- Added course study notes, revision questions, setup guidance, an inventory and validation tooling.
- Added three distinct companion notebooks created with AI assistance, with local execution evidence.

Worked exercise answers and compatibility corrections were subsequently added under `solutions/`, without modifying the reference layer. The answers use embedded instructor explanations where available and additional AI-assisted implementations. They were not submitted or graded. The collected files do not establish the owner's certificate status or individual authorship of all notebook code.

## Trace every source

[source-manifest.json](../provenance/source-manifest.json) maps each of the 149 original source assets to its curated target. It includes the original byte SHA-256, a cell-content digest for duplicate detection, the curated content digest, and recorded changes. It omits personal download paths. Original source filenames remain in the mapping for comparison with course pages.

The untouched download archive sits outside this Git repository. The 147 original notebooks become 138 course notebooks; both source guides are retained byte-for-byte. Shared material across different courses is retained in each course's folder.

## Collection limits

All 13 course folders contain material, but this is not a claim that every lab in every syllabus revision was collected. Original labels sometimes differed from the actual downloaded content. The breast-cancer notebook's curated name follows its dataset rather than its misleading Wine filename. The introductory cross-entropy page had no direct download link, so that notebook was collected from the advanced PyTorch course; this is recorded in the manifest.

The RAG application project is present as instructional guides and component notebooks. The source archive did not contain a saved runnable application folder. Mevzuat RAG remains a separate project and is not copied into this collection.

## Companion material

The companion notebooks and repository documentation were newly prepared with AI assistance for this collection. They illustrate concepts independently of the course assignments and include their own limitations. The Wine companion uses the bundled scikit-learn dataset and links its documentation; the PyTorch example uses generated data; the retrieval example uses newly written short notes. No third-party model weights or datasets are redistributed here.

## Worked-copy provenance

[Guarded patches](../provenance/solution-patches.json) record each original cell hash, replacement, origin and explanation. [Exercise inventory](../provenance/exercise-inventory.json) and [review decisions](../provenance/solution-reviewed.json) distinguish exercises from existing implementations and false-positive markers. The build verifies all recorded locations and source hashes before applying a patch.

Nine earlier vision-capstone answer sets were reused after every template cell matched the current references. Historical outputs were not imported as current execution evidence. New bounded outputs are recorded separately, with the exact worked-source hash. The supplied answers are study material; syntax checks and smoke execution do not certify independent authorship or full model performance.
