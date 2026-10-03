# NotebookLM Clone — Project Spec

## 1) Goal
Build a lightweight NotebookLM/Gemini-notebook style app that lets a user create multiple notebooks, ingest PDFs, PPTX, TXT, and URLs, chat with the notebook using RAG with citations, generate markdown artifacts, and run a two-method retrieval evaluation. The app must run in a Hugging Face Space and auto-deploy from GitHub via GitHub Actions.

## 2) Requirements
- Notebook management: create, rename, delete, switch; each notebook has a unique ID.
- Per-notebook storage for sources, chat history, artifacts, and vector data.
- Ingestion pipeline: extract text, split into chunks, embed, index in a vector database, store citations metadata.
- Chat flow: user asks a question, retrieve relevant chunks, send them to the LLM, return answer with visible citations.
- Persist chat history and reload it per notebook.
- Artifacts: generate a report markdown and quiz markdown with answer key; allow view and download.
- Storage survives app restarts; if persistence is limited in Hugging Face, document the constraint and keep the design simple.
- UI: Gradio-based app with notebook manager, source upload, URL ingestion, chat window, citations, artifact generation buttons, artifact view/download, and clear errors.
- Retrieval comparison: compare at least two methods (vector-only vs vector + reranker) on a small evaluation set and log method, retrieved chunks, latency, answer quality, and conclusion.
- Deployment: GitHub repo, GitHub Actions, HF Space deployment, secrets managed through GitHub/HF env vars; no secrets committed in repo.
- Deliverables: README, requirements file, workflow, .gitignore, architecture doc, HF Space URL, evaluation write-up, demo recording.

## 3) Components
- UI layer: Gradio app for notebook management, uploads, URL ingestion, chat, citations, and artifacts.
- Storage layer: notebook metadata, source metadata, chat persistence, artifact persistence, vector index storage.
- Ingestion service: file decoding, text extraction, chunking, embedding, vector indexing.
- Retrieval layer: vector search and reranker-based retrieval.
- Generation layer: LLM answer synthesis and artifact generation.
- Evaluation layer: test questions, retrieval comparison, logging and scoring.
- Deployment layer: GitHub Action that deploys HF Space build using repo + secrets.

## 4) Data flow
1. User creates or selects a notebook.
2. User adds a source (PDF, PPTX, TXT, or URL).
3. Source is ingested into the notebook’s storage path.
4. Text is extracted and chunked.
5. Chunks are embedded and stored in a notebook-scoped vector database along with metadata.
6. User asks a question in the notebook chat.
7. Query is embedded and retrieved against the notebook vector store.
8. Retrieved chunks are optionally reranked; top chunks are passed to the LLM with citations metadata.
9. LLM returns a grounded answer and the app displays citations.
10. Chat history is saved and later reloaded.
11. User creates report or quiz artifact from notebook context; file is persisted and viewable/downloadable.

## 5) Notebook data model
Notebook record:
- id: unique string
- name: string
- created_at: timestamp
- updated_at: timestamp
- sources: list of source records
- chat_history: list of chat messages
- artifacts: list of generated artifact records

Source record:
- id: unique string
- type: pdf | pptx | txt | url
- name: string
- path: string or URL
- status: queued | ingested | failed
- created_at: timestamp
- metadata: page/slide/url, chunk_count, etc.

Chunk record:
- id: unique string
- notebook_id: string
- source_id: string
- chunk_index: int
- text: string
- page_or_slide: int or string
- url: string or null
- embedding_vector_id: string

Chat message:
- id: string
- role: user | assistant
- content: string
- timestamp: timestamp
- citations: list of relevant source metadata

Artifact record:
- id: string
- type: report | quiz
- filename: string
- path: string
- created_at: timestamp

## 6) RAG pipeline
- Query enters the notebook-specific retriever.
- Step A: vector search retrieves top candidates from notebook embeddings.
- Step B: optional reranker reorders candidates using pairwise relevance scoring.
- Step C: top passages are assembled with citation metadata.
- Step D: system prompt instructs the model to answer using only the retrieved context and cite each claim using source metadata.
- Outputs: answer text + citations + saved chat history.

## 7) API / model dependencies
Planned defaults:
- LLM: Groq-hosted GPT-OSS (openai/gpt-oss-120b) via environment variable key
- Embeddings: sentence-transformers local model
- Vector DB: ChromaDB, one collection per notebook
- Reranker: cross-encoder
- UI: Gradio
- Parsing: PyMuPDF / pdfplumber for PDFs, python-pptx for PPTX, standard text parsing for TXT, requests/BeautifulSoup for URLs

## 8) Deployment approach
- GitHub repo contains app code, requirements, workflow, docs, and ignore rules.
- GitHub Actions runs on push to main, installs dependencies, optionally builds smoke checks, and deploys to a Hugging Face Space using a HF token stored in GitHub Secrets.
- HF Space hosts the app using the repository as the source and uses environment variables for the Groq API key and optional model config.
- App should persist notebook data in a workspace-local directory under `data/` by default. If HF persistence is limited, the app must document this limitation and remain simple.

## 9) Storage and persistence
- Default local structure:
  - `data/notebooks/<notebook_id>/raw/`
  - `data/notebooks/<notebook_id>/text/`
  - `data/notebooks/<notebook_id>/chroma/`
  - `data/notebooks/<notebook_id>/chats.json`
  - `data/notebooks/<notebook_id>/artifacts/`
- A small storage abstraction will centralize reading/writing notebook records, chat files, and artifact files to keep the app consistent.
- Persistence chapter in README must clearly state whether the HF Space retains data across restarts and what is expected under the platform limits.

## 10) Open decisions to confirm
Before finalizing this spec, we need your approval on the defaults below.

## 11) Approved defaults
- LLM: Groq API (model `openai/gpt-oss-120b`) via `GROQ_API_KEY` environment variable. (Changed from Gemini at user request.)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Chunking: 700–1000 characters per chunk with 100–150 character overlap.
- Reranker: cross-encoder reranker applied after vector retrieval.
- Storage: `data/notebooks/<notebook_id>/...` with a thin storage abstraction for notebook metadata, sources, chats, and artifacts.
- HF persistence: local `data/` directory with clear documentation that persistence is best effort and data may reset on the platform.

This spec is now approved for implementation.
