# 12 · Fundamentals of AI Agents Using RAG and LangChain

Prompt templates, retrieval-augmented generation, document summarization and tool use.

[Repository home](../../README.md) · [Full catalog](../../docs/LAB_CATALOG.md) · [Environment setup](../../docs/SETUP.md)

## Concepts to retain

- A retriever selects evidence; a generator writes an answer conditioned on that evidence. Evaluate those stages separately.
- Chunk boundaries, overlap and metadata determine what can be retrieved and cited. Preserve stable source IDs.
- An agent selects actions or tools. Tool calls need explicit inputs, observable results and bounded execution.
- Prompt text from a retrieved document is untrusted data; it should not redefine application instructions.

## Working model

Recall@k is the fraction of relevant items found among the top k results. For queries with one expected source, Hit@k checks whether that source appears in the top k.

## Start with these labs

- [RAG with PyTorch](notebooks/rag-with-pytorch.ipynb) — Retrieval concepts in PyTorch.
- [Enhance LLMs using RAG and Hugging Face](notebooks/enhance-llms-using-rag-and-hugging-face.ipynb) — Hugging Face RAG.
- [LangChain](notebooks/langchain.ipynb) — LangChain components.

## Revisit with a purpose

- Run the local retrieval companion to inspect ranking, source attribution and an unsupported query.
- Trace a document through loading, chunking, indexing, retrieval and answer construction.
- Review the weather-agent starter and identify required credentials and external requests before execution.

## Common mistakes

- A fluent answer does not prove that the retrieved evidence supports it.
- Retrieval scores are ranking signals rather than calibrated confidence probabilities.
- Cloud-backed labs depend on provider credentials and current APIs; no live calls were made during curation.

## Check your understanding

- How do you tell a retrieval failure from a generation failure?
- What should happen when no useful source is found?
- Which metadata is needed to trace an answer back to its source?

## Collected material

6 reference notebooks. Course source, authorship and copyright notices remain in the collected files. Separate worked copies contain AI-assisted study answers; consult [solution progress](../../docs/SOLUTION_PROGRESS.md) for runtime evidence.

| Material | Reference status | Study answer |
|---|---|---|
| [Enhance LLMs using RAG and Hugging Face](notebooks/enhance-llms-using-rag-and-hugging-face.ipynb) | Reference | [Worked copy](../../solutions/12-ai-agents-and-rag/notebooks/enhance-llms-using-rag-and-hugging-face.ipynb) |
| [In-Context Learning and Prompt Templates](notebooks/in-context-learning-and-prompt-templates.ipynb) | Reference | [Worked copy](../../solutions/12-ai-agents-and-rag/notebooks/in-context-learning-and-prompt-templates.ipynb) |
| [Lab: Building AI Agents from Scratch with Python](notebooks/lab-building-ai-agents-from-scratch-with-python.ipynb) | Reference; 1 syntax blanks | [Worked copy](../../solutions/12-ai-agents-and-rag/notebooks/lab-building-ai-agents-from-scratch-with-python.ipynb) |
| [LangChain](notebooks/langchain.ipynb) | Reference | [Worked copy](../../solutions/12-ai-agents-and-rag/notebooks/langchain.ipynb) |
| [RAG with PyTorch](notebooks/rag-with-pytorch.ipynb) | Reference | [Worked copy](../../solutions/12-ai-agents-and-rag/notebooks/rag-with-pytorch.ipynb) |
| [Summarize Private Documents Using RAG, LangChain, and LLMs](notebooks/summarize-private-documents-using-rag-langchain-and-llms.ipynb) | Reference | [Worked copy](../../solutions/12-ai-agents-and-rag/notebooks/summarize-private-documents-using-rag-langchain-and-llms.ipynb) |
