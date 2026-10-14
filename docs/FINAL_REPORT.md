# Final Report: NotebookLM / Gemini Notebook Clone

CS/IT 5010 project. Himanshi Shrivas.

| Deliverable | Where |
|---|---|
| GitHub repository | https://github.com/darpan02-cypher/NotebookLM-Gemini-Notebook-Clone |
| Deployed app | https://notebooklm-clone-4ygpfpzriq-uc.a.run.app (Google Cloud Run) |
| RAG evaluation | [docs/EVALUATION.md](EVALUATION.md), raw data in `eval/results/v2/` |
| Architecture | [docs/ARCHITECTURE.md](ARCHITECTURE.md) |
| Demo recording (1 to 2 min) | add link here; script in [DEMO_CHECKLIST.md](DEMO_CHECKLIST.md) |

## 1. What was built
A full-stack NotebookLM-style app. A user creates multiple notebooks, adds PDF, PPTX, TXT or web-page sources, asks questions answered by retrieval-augmented generation with visible citations, and generates a Report or a Quiz with answer key as downloadable Markdown. It is containerised and auto-deployed from GitHub to Google Cloud Run by GitHub Actions.

**Stack:** Gradio UI, ChromaDB (one collection per notebook), `all-MiniLM-L6-v2` embeddings, `ms-marco-MiniLM-L-6-v2` cross-encoder reranker, Groq-hosted `openai/gpt-oss-120b` for generation.

## 2. Architecture
Layers, each in its own package: `ui/` (Gradio wiring only), `storage/` (per-notebook files and metadata behind `StorageManager`), `ingestion/` (extract, chunk, embed, index), `retrieval/` (embedder, vector store, reranker), `generation/` (chat, artifacts, LLM client), `eval/` (retrieval comparison). The UI holds no business logic, and the LLM provider is isolated in one file (it was switched from Gemini to Groq with a one-file change).

Data flow, module responsibilities, the notebook data model and design decisions are in [ARCHITECTURE.md](ARCHITECTURE.md).

## 3. Rubric cross-check
Self-assessment against the grading rubric. Points are not claimed; this shows the evidence for each "Excellent" criterion.

| Category (max) | Required for "Excellent" | Evidence | Gaps |
|---|---|---|---|
| RAG implementation & evaluation (15) | Chunking, embeddings, vector DB, top-k retrieval; grounded answers with clear citations | 900-char chunks, 120 overlap; MiniLM embeddings; Chroma cosine search; top-5 into the prompt; the model must cite `[n]`; UI lists source, page/slide, chunk and snippet. Two methods compared on a 40-question set with statistics (section 5) | None known |
| System architecture & design (15) | Modular, separated concerns, documented data flow, design decisions, README with setup | Section 2; README covers local run, Docker, config, CI/CD setup | None known |
| Ingestion pipeline (12) | PDF, PPTX, TXT, single-page URL; reliable extraction; chunking; citation metadata; vector DB integration | All four types in `source_ingestor.py`; per-page/slide chunking; metadata `notebook_id, source_id, source_name, page_or_slide, chunk_index, url`; failures stored as `status=failed` with the error | No OCR for scanned PDFs; no JavaScript-rendered pages (both stated in README) |
| Artifact generation (12) | Report, quiz plus answer key; clean formatting; downloadable; grounded | `ArtifactService` builds both from notebook chunks (sampled evenly up to 30k chars); saved as `.md`, viewable and downloadable in the UI | Artifact grounding was not formally evaluated |
| Storage & notebook data (10) | Organised per-notebook structure (raw, text, vectors, chats, artifacts); abstraction layer | `data/notebooks/<id>/{info.json, chats.json, raw/, text/, chroma/, artifacts/}` behind `StorageManager`; secrets and `data/` gitignored | Cloud Run disk is ephemeral, so deployed notebooks are lost when the instance is replaced. Documented, with upgrade options (GCS mount, Filestore) |
| UI & workflow (8) | Notebook manager, upload and URL ingest, chat with citations, artifact buttons and download, helpful errors | All present in `ui/gradio_app.py`; three themes; per-action status messages (unsupported type, bad URL, missing notebook) | None known |
| Documentation & deliverables (8) | Architecture, module roles, live link and repo link, multi-technique RAG comparison, trade-off discussion | README, ARCHITECTURE, EVALUATION; latency and model-size trade-offs discussed | Demo link still to be added to the table above |

## 4. Deployment and CI/CD
`.github/workflows/deploy.yml` runs on push to `main`: install, run the offline tests (27), build the image (CPU-only torch, models baked in), push to Artifact Registry, and `gcloud run deploy` (2 GiB, 2 vCPU, one instance, scale to zero). The service-account key is a GitHub Secret; `GROQ_API_KEY` is a Cloud Run environment variable. No secrets are in the repo. The workflow retries the base-image pull to survive Docker Hub rate limits.

## 5. RAG evaluation
Full detail: [EVALUATION.md](EVALUATION.md). Summary of the round to cite (v2):

**Test questions.** 40 questions in `eval/questions_v2.json` over an 8-document, 112-chunk corpus: 12 direct, 14 paraphrased, 14 confusable (a look-alike passage with a different value exists). Written before either method was run. Gold labels are chunk IDs with a hash check, so the labels cannot silently drift. An earlier 18-question pilot could not separate the methods (both 18/18), which is why v2 was built.

**Methods.** `vector`: Chroma cosine ranking. `rerank`: the 20 nearest vector candidates reordered by the cross-encoder. Both return a top 10; the top 5 go to the LLM.

**Retrieved chunks.** Every question's retrieved chunk IDs, scores and gold rank are in `eval/results/v2/eval_log.json`. Example, q01 "Who were the three crew members of Apollo 12?" (gold chunk `apollo_missions_extended.txt#004`):
- vector: gold chunk not in the top 10. Top result was `apollo_missions_extended.txt#011`. The model answered "I could not find the names...".
- rerank: gold chunk at rank 2. The model answered "Charles 'Pete' Conrad, Alan Bean, and Richard Gordon" and cited it.

**Retrieval results (n = 40).**

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Mean retrieval latency |
|---|---|---|---|---|---|
| vector | 24/40 | 34/40 | 34/40 | 0.731 | 19.8 ms |
| rerank | 31/40 | 40/40 | 40/40 | 0.871 | 98.4 ms |

Paired bootstrap: MRR@10 difference +0.140, 95% interval [+0.023, +0.258], which excludes 0. Hit@1 difference +0.175, interval [0.000, +0.350], which touches 0, so it is suggestive only. The gain is concentrated in confusable questions (MRR +0.243, interval excludes 0; exploratory).

**Response times.** Retrieval: about 20 ms (vector) vs 98 ms (rerank). End to end including the LLM: 9.4 s vs 9.8 s. The extra reranking cost, about 80 ms, is small next to the LLM call, and the reranker adds a second model of about 90 MB.

**Answer quality.** Two LLM judges (qwen3.8-27b and TypeSafe JEV), neither the answer-writing model and both blind to the method name:

| Judge | Method | Faithfulness | Correctness | Pairwise (rerank / vector / tie) |
|---|---|---|---|---|
| qwen | vector | 4.33 | 4.48 | 9 / 3 / 28 |
| qwen | rerank | 4.65 | 5.00 | |
| JEV | vector | 4.39 | 4.45 | 7 / 1 / 32 |
| JEV | rerank | 4.61 | 4.94 | |

Both judges prefer rerank, but the pairwise sign tests are not significant (p = 0.146 and 0.070).

**Final conclusions.**
1. Reranking retrieved the gold chunk earlier on this harder set (statistically supported for MRR), and put it in the LLM's context for 40/40 questions vs 34/40.
2. Judged answer quality points the same way, but the pairwise evidence is suggestive, not conclusive.
3. The cost is small: about 80 ms and one extra model. Reranking is kept as the default and `vector` stays selectable in the UI.
4. Reranking only reorders vector's top 20, so it cannot recover a chunk outside them.

**Limitations of the evaluation.** One self-written corpus and question set (n = 40); gold labels written by us; single run; answer quality rests on LLM judges, and the 10-item blind human spot-check (`judge_spotcheck.csv`) has not been filled in, so judge reliability is unmeasured.

## 6. Known limitations
- Deployed storage is ephemeral (see rubric table); notebooks reset when the Cloud Run instance is replaced.
- Scanned PDFs (no OCR) and JavaScript-rendered pages are unsupported.
- First request after idle has a cold start of several seconds.
- Evaluation limitations as above.

## 7. Before submitting
1. Fill in the demo-recording link at the top.
2. Record the demo (script in [DEMO_CHECKLIST.md](DEMO_CHECKLIST.md)); it must show a green GitHub Actions run.
3. Optional: fill in the human spot-check and run `python -m eval.judge --agreement`.
4. Push to `main` and confirm the Actions run is green and the live URL works.
