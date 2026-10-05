# Architecture

```
            Gradio UI (ui/gradio_app.py, app.py)
   notebooks | sources | chat + citations | artifacts
        |            |            |              |
        v            v            v              v
   StorageManager  SourceIngestor  ChatService   ArtifactService
   (storage/)      (ingestion/)    (generation/) (generation/)
        |            |   \            |   \           |
        |            |    \           v    \          v
        |            |     \      Retriever \     GroqClient --> Groq API
        |            |      \    (retrieval/) \--------^
        |            v       v        |
        |      LocalEmbedder  NotebookVectorStore (ChromaDB)
        |      (MiniLM-L6)    CrossEncoderReranker
        v
   data/notebooks/<id>/{info.json, chats.json, raw/, text/, chroma/, artifacts/}
```

## Modules
| Module | Responsibility |
|---|---|
| `storage/` | Thin abstraction over the per-notebook folder: notebook CRUD, sources, chats, artifacts metadata. Pure JSON + files. |
| `ingestion/` | Extract text per page/slide (pypdf, python-pptx, TXT, requests + BeautifulSoup), chunk (900 chars, 120 overlap, word-boundary breaks), embed in batch, index in Chroma with citation metadata. Failed sources are stored with `status=failed` and the error. |
| `retrieval/` | `LocalEmbedder` (sentence-transformers), `NotebookVectorStore` (one Chroma collection per notebook, cosine), `Retriever` with two methods: `vector` and `rerank`. |
| `generation/` | `ChatService` (retrieve -> numbered context -> LLM -> answer + citations -> persist), `ArtifactService` (report / quiz markdown from sampled notebook chunks), `GroqClient`. |
| `ui/` | Gradio Blocks wiring only; no business logic. |
| `eval/` | Two-method retrieval comparison script, corpus and question set. |

## Data flow
**Ingest:** file/URL -> raw copy in `raw/` -> text per page/slide -> `text/<source_id>.txt` -> chunks -> embeddings -> Chroma (`notebook_id`, `source_id`, `source_name`, `page_or_slide`, `chunk_index`, `url` as metadata) -> source record appended to `info.json`.

**Chat:** question -> embed -> Chroma top-N -> (optional cross-encoder rerank) -> top-5 chunks -> prompt with numbered passages and last 4 messages -> LLM answers using only the context with `[n]` citations -> UI shows answer plus citation list (source, page/slide, URL, chunk, snippet) -> messages appended to `chats.json`.

**Artifacts:** all chunks of the notebook, sampled evenly up to 30k chars -> LLM with report or quiz prompt -> `.md` saved in `artifacts/`, record in `info.json`, viewable and downloadable in the UI.

## Notebook data model
- `info.json`: `id`, `name`, `created_at`, `updated_at`, `sources[]` (`id`, `type`, `name`, `path`, `status`, `metadata`), `artifacts[]` (`id`, `type`, `filename`, `path`, `created_at`).
- `chats.json`: list of `{id, role, content, timestamp, citations[], method, latency_s}`.
- Notebook isolation: each notebook has its own folder and its own Chroma collection, so retrieval can never cross notebooks.

## Key design decisions
- **Local embeddings + reranker**, hosted LLM: only the generation step needs a network key and secret.
- **Plain files + Chroma persistent client** instead of a database: transparent, easy to inspect, zero extra infrastructure.
- **Lazy model loading** so the app and tests start fast; models load on first use.
- **Per-page chunking** so every citation can point to a page or slide.
- **LLM provider isolated** in `generation/llm.py`; switching providers is a one-file change (it was changed from Gemini to Groq during development).

## Deployment
GitHub push to `main` -> GitHub Actions runs tests -> builds the Docker image (CPU-only torch, both models baked in) -> pushes to Artifact Registry -> `gcloud run deploy` to Cloud Run (2 GiB, 2 vCPU, max 1 instance, scale to zero). `GCP_SA_KEY` is a GitHub secret; `GROQ_API_KEY` is set on the Cloud Run service and never stored in the repo.

## Persistence limits
Cloud Run's container filesystem is ephemeral: `data/` survives while an instance lives, and **is wiped when the instance is replaced** (scale-to-zero after idle, every deploy, restarts). Notebooks are therefore best effort in the deployed app. `--max-instances 1` prevents one user's notebooks from being split across instances. Options (not implemented, by design): mount a Cloud Storage bucket (GCS FUSE volume) at `data/`, or use Filestore. The storage root is a single constructor argument of `StorageManager`, so either is a small change.
