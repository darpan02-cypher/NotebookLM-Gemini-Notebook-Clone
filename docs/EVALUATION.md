# RAG Evaluation: vector vs. vector + cross-encoder rerank

## Setup
- **Corpus:** 2 short documents in `eval/corpus/` (Apollo program, photosynthesis), ingested into one notebook (about 17 chunks of ~900 chars).
- **Questions:** 18 in `eval/questions.json`: 10 direct questions and 8 paraphrased ones with little word overlap with the source text.
- **Methods** (top-k = 3):
  - `vector`: embed query with all-MiniLM-L6-v2, Chroma cosine top-3.
  - `rerank`: Chroma top-20 candidates, re-scored by cross-encoder `ms-marco-MiniLM-L-6-v2`, top-3 kept.
- **Generation:** same LLM (Groq `openai/gpt-oss-120b`) and prompt for both methods.
- **Run:** `python -m eval.run_eval` (full per-question log with retrieved chunks, answers and timings: `eval/results/eval_log.json`).
- **Metrics:** Hit@3 (gold passage in top-3), rank of the gold passage, MRR, keyword coverage of the generated answer, retrieval latency (models warmed up first), and end-to-end latency (retrieval + LLM).

## Results

| Method | Hit@3 | Mean rank of hit | MRR | Answer keyword coverage | Mean retrieval latency |
|---|---|---|---|---|---|
| vector | 18/18 | 1.11 | 0.94 | 1.00 | 113 ms |
| rerank | 18/18 | 1.11 | 0.96 | 0.97 | 142 ms |

Per-question differences (the only two rows where methods disagreed):

| Question | vector rank | rerank rank |
|---|---|---|
| How much material did the first moonwalkers bring back to Earth? | 2 | **1** |
| What happens to plant sugar production when it gets extremely hot? | **2** | 3 |

End-to-end latency averaged 5.5 s (vector) and 5.6 s (rerank) per question and is dominated by the LLM call, not retrieval.

## Answer quality (manual check)
`manual_quality` in the log is left for hand scoring. I read a sample of answers from both methods: they were correct and grounded in the retrieved passages, and all 36 answers contain `[n]` citations. The one keyword-coverage miss (rerank, 0.50 on the "extremely hot" question) is a false negative of the metric: the answer says the heat "damages the enzymes" instead of using the word "denature".

## Conclusion
- **On this corpus the two methods are effectively tied.** Both found the gold passage for 18/18 questions, and answer quality was identical. Reranking improved one question (rank 2 -> 1) and hurt one (rank 2 -> 3), so MRR moved 0.94 -> 0.96, which is within noise at n = 18.
- **Cost:** reranking added little latency here (about 30 ms on average in this run) because only 20 candidates are scored on a CPU with a small model. The cross-encoder adds a second model download (~90 MB) and memory.
- **Why no clear win:** the notebook is tiny, so top-3 of ~17 chunks almost always contains the answer. Reranking is expected to matter more with many sources, where the first-stage top-k misses relevant chunks.
- **Recommendation:** keep reranking available (it is the UI default and helps on paraphrased queries such as the moonwalkers one), but this evaluation does not prove it is better. A larger corpus and question set would be needed for a statistically meaningful comparison.

## Limitations
- Small, self-written corpus and question set; results are not generalizable.
- "Gold passage" detection uses exact-phrase matching, which can undercount a correct chunk phrased differently.
- Keyword coverage is a coarse proxy for answer quality; the LLM is non-deterministic (temperature 0.2).
