# 13 · Project: Generative AI Applications with RAG and LangChain

Document loaders, splitting, embeddings, vector stores, retrievers and a Gradio QA interface.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- Keep ingestion separate from answering so documents are not repeatedly parsed and embedded for every query.
- Choose chunk size and overlap using retrieval evidence, not the maximum model context alone.
- Persist source identifiers alongside embeddings, then return source references with answers.
- A usable QA app needs failure handling for empty files, unsupported questions, provider errors and private data.

## Working model

The application flow is document → loader → chunks → embeddings → index → retriever → prompt context → model response → source references.

## Start with these labs

- [Load Documents Using LangChain for Different Sources](notebooks/load-documents-using-langchain-for-different-sources.ipynb) — Document loading.
- [Apply Text Splitting Techniques to Enhance Model Responsiveness](notebooks/apply-text-splitting-techniques-to-enhance-model-responsiveness.ipynb) — Text splitting.
- [Create and Configure a Vector Database to Store Document Embeddings](notebooks/create-and-configure-a-vector-database-to-store-document-embeddings.ipynb) — Vector storage.
- [Develop a Retriever to Fetch Document Segments Based on Queries](notebooks/develop-a-retriever-to-fetch-document-segments-based-on-queries.ipynb) — Retrieval.
- [QA Bot Project Guide](guides/qa-bot-project-guide.md) — QA bot integration guide.

## Revisit with a purpose

- Study the loader, splitter, vector-store and retriever notebooks in that order.
- Read the QA bot and Gradio guides as integration references; they are not a saved runnable application.
- Compare the course architecture with a future separate Mevzuat RAG case study, focusing on evaluation and citations.

## Common mistakes

- Passing a whole document into a prompt can exceed limits or bury useful evidence.
- The Skills Network project identifier is lab infrastructure, not a personal production credential.
- The collected archive contains project guides rather than an application source directory; Mevzuat RAG will be linked separately.

## Check your understanding

- Which step must be repeated when the embedding model changes?
- How would you show that an answer is supported by a particular page?
- How would you evaluate unsupported questions and document updates?

## Collected material

6 reference notebooks and 2 source guides. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Gradio Interface Guide](guides/gradio-interface-guide.md) | Source guide | — |
| [QA Bot Project Guide](guides/qa-bot-project-guide.md) | Source guide | — |
| [Apply Text Splitting Techniques to Enhance Model Responsiveness](notebooks/apply-text-splitting-techniques-to-enhance-model-responsiveness.ipynb) | Reference | [Worked copy](../../solutions/13-rag-application-project/notebooks/apply-text-splitting-techniques-to-enhance-model-responsiveness.ipynb) |
| [Create and Configure a Vector Database to Store Document Embeddings](notebooks/create-and-configure-a-vector-database-to-store-document-embeddings.ipynb) | Reference | [Worked copy](../../solutions/13-rag-application-project/notebooks/create-and-configure-a-vector-database-to-store-document-embeddings.ipynb) |
| [Develop a Retriever to Fetch Document Segments Based on Queries](notebooks/develop-a-retriever-to-fetch-document-segments-based-on-queries.ipynb) | Reference | [Worked copy](../../solutions/13-rag-application-project/notebooks/develop-a-retriever-to-fetch-document-segments-based-on-queries.ipynb) |
| [Embed documents using watsonx's embedding model](notebooks/embed-documents-using-watsonx-s-embedding-model.ipynb) | Reference | [Worked copy](../../solutions/13-rag-application-project/notebooks/embed-documents-using-watsonx-s-embedding-model.ipynb) |
| [Load Documents Using LangChain for Different Sources](notebooks/load-documents-using-langchain-for-different-sources.ipynb) | Reference | [Worked copy](../../solutions/13-rag-application-project/notebooks/load-documents-using-langchain-for-different-sources.ipynb) |
| [Put Whole Document into Prompt and Ask the Model](notebooks/put-whole-document-into-prompt-and-ask-the-model.ipynb) | Reference | [Worked copy](../../solutions/13-rag-application-project/notebooks/put-whole-document-into-prompt-and-ask-the-model.ipynb) |
