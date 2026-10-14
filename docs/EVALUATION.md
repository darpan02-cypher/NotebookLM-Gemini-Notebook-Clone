# RAG Evaluation: vector vs. vector + cross-encoder rerank

Two rounds. The **pilot (v1)** was too easy to separate the methods (both 18/18). **v2** uses a harder corpus and question set and is the one to cite.

## v2: harder corpus, 40 questions, paired comparison

### Setup
- **Corpus:** 8 documents in `eval/corpus/` (112 chunks at 900 chars / 120 overlap): the 2 pilot documents plus 6 new ones written for this evaluation in our own words (more Apollo missions, Mercury/Gemini, Skylab/ISS, Voyager, cellular respiration with mitosis/meiosis, C3/C4/CAM). They overlap on purpose (dates, crews, durations, enzymes, locations), so look-alike distractor passages exist.
- **Questions:** 40 in `eval/questions_v2.json`: 12 direct, 14 paraphrased (little word overlap with the source), 14 confusable (a similar passage with a different value exists elsewhere and the question names the one it wants). Written before either method was run and not edited afterwards.
- **Gold labels:** by chunk ID (`<file>#<chunk_index>`), not phrase matching. A chunk is gold if it contains the full answer-bearing sentence, so 40 questions carry 56 gold IDs. Each label stores the sha256 of its chunk text, and `python -m eval.validate_questions` fails if the corpus or chunking changes.
- **Methods:** `vector` = Chroma cosine order. `rerank` = the 20 nearest vector candidates reordered by cross-encoder `ms-marco-MiniLM-L-6-v2`. Each returns a ranked top 10.
- **Generation:** Groq `openai/gpt-oss-120b`, top-5 context (as in the app), no chat history, answers cached.
- **Metrics:** Hit@1/3/5, MRR@10, mean rank of the first gold chunk (questions with no gold chunk in the top 10 are excluded from this mean and counted as misses), retrieval latency (models warmed up first).
- **Run:** `python -m eval.run_eval_v2 --retrieval-only` (no LLM), `python -m eval.run_eval_v2` (full), `python -m eval.paired_stats` (statistics). Outputs: `eval/results/v2/`.

### Retrieval results (n = 40)

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Mean gold rank | Missed top 10 | Mean latency |
|---|---|---|---|---|---|---|---|
| vector | 24/40 | 34/40 | 34/40 | 0.731 | 2.08 | 1 | 19.8 ms |
| rerank | 31/40 | 40/40 | 40/40 | 0.871 | 1.32 | 0 | 98.4 ms |

MRR@10 by question type (point estimates; 12 to 14 questions each):

| Type | vector | rerank |
|---|---|---|
| direct (12) | 0.792 | 0.861 |
| paraphrased (14) | 0.736 | 0.833 |
| confusable (14) | 0.673 | 0.917 |

### Paired comparison (rerank minus vector)
Question by question, rerank was better on 13, worse on 5 and equal on 22 (reciprocal rank). On Hit@1 it won 11, lost 4 and tied 25.

Paired bootstrap over questions (10,000 resamples, fixed seed, 95% percentile interval):

| Difference | Estimate | 95% interval | Includes 0? |
|---|---|---|---|
| MRR@10 | +0.140 | [+0.023, +0.258] | No |
| Hit@1 | +0.175 | [+0.000, +0.350] | Borderline: the lower end is exactly 0.000, so we treat it as including 0 |

By type (exploratory, not corrected for multiple comparisons): confusable MRR +0.243 [+0.053, +0.437] and Hit@1 +0.357 [+0.071, +0.643] exclude 0; direct and paraphrased intervals include 0.

### Generation (top-5 context, automatic signals only)

| Method | Gold chunk in context | Answer cites a gold chunk | Mean end-to-end latency |
|---|---|---|---|
| vector | 34/40 | 28/40 | 9.4 s |
| rerank | 40/40 | 37/40 | 9.8 s |

These automatic signals measure whether the right evidence reached the model and was cited, not whether the answer is correct. Answer quality is scored by two LLM judges in the next section.

### Answer quality (two LLM judges)
Both methods' answers (top-5 context) were scored by two judges that did not write the answers and never saw the method name: `qwen/qwen3.8-27b` on Groq (temperature 0) and TypeSafe JEV (`typesafe/jev-1.13`, via OpenRouter). Pointwise scores are faithfulness and correctness, each 1 to 5 against the reference answer and cited passages. Pairwise shows both answers as A/B in a balanced, seeded order (seed 4242) and asks for a winner or tie. Run: `python -m eval.judge --judge both`. Outputs: `eval/results/v2/judge_summary.csv`, `judgments_*.json`.

| Judge | Method | Faithfulness (mean) | Correctness (mean) |
|---|---|---|---|
| qwen | vector | 4.325 | 4.475 |
| qwen | rerank | 4.650 | 5.000 |
| JEV | vector | 4.393 | 4.453 |
| JEV | rerank | 4.607 | 4.936 |

Pairwise (n = 40):

| Judge | rerank wins | vector wins | Ties | Identical answers (within ties) | Sign test p (two-sided, ties dropped) |
|---|---|---|---|---|---|
| qwen | 9 | 3 | 28 | 5 | 0.146 |
| JEV | 7 | 1 | 32 | 5 | 0.070 |

- **Both judges agree on direction.** Rerank scored higher on faithfulness and correctness, and won more pairwise comparisons than it lost. Correctness was at or above 4.5 for 40/40 rerank answers against 34/40 (qwen) and 35/40 (JEV) for vector.
- **The pairwise result is suggestive, not significant.** Most comparisons were ties, and the sign test on the decided ones does not reach p < 0.05 for either judge. Pointwise differences were not given a paired interval.
- **This is consistent with the retrieval result:** vector put the gold chunk in the top-5 context for 34/40 questions and rerank for 40/40, and the judged gap is about that size. We did not isolate the cause for each question.
- **Human spot-check (10 blind items).** The author scored 10 randomly chosen answers (method hidden) with the same rubrics. Agreement with the judges: correctness exact 100% (qwen) and 100% (JEV); faithfulness exact 80% (qwen) and 70% (JEV), within one point 90% for both. Caveat: the author scored all 10 answers 5/5, so the sample has no weak answers and cannot show whether the judges catch bad ones; it only shows they do not under-rate good ones. The disagreements are faithfulness cases where a judge scored below 5, notably an answer that cited nothing valid (S297), which the judge rubric penalises and the human score did not. The first qwen pass used an unbalanced A/B order; it is kept as `firstpass_qwen_unbalanced_order.json` and was replaced by the balanced run above.

### Conclusion
- **On this harder set, reranking retrieved the gold chunk earlier.** MRR@10 rose from 0.731 to 0.871 and the paired interval excludes 0. Hit@3 was 40/40 against 34/40.
- **The Hit@1 gain is suggestive, not established:** +0.175 with an interval touching 0.
- **The gain is concentrated in confusable questions** (look-alike passages), which is where a cross-encoder reading query and passage together is expected to help. The per-type split is exploratory.
- **Cost:** about 80 ms extra retrieval latency per query in this run (98 vs 20 ms), negligible next to the ~9 s LLM call, plus a second model (~90 MB).
- **Limit of reranking here:** it only reorders vector's top 20, so it cannot recover a gold chunk outside them.
- **Judged answer quality points the same way** (correctness 4.94 to 5.00 vs 4.45 to 4.48 for vector), but the pairwise wins are not significant and judge-human agreement was not measured.
- **Recommendation:** keep reranking as the default. The evidence is consistent with a real retrieval improvement on this set but is not proof it generalises.

### Limitations
- n = 40 questions, self-written, one small corpus (112 chunks); the corpus, questions and gold labels were all written by us, so they may reflect our expectations about what is hard.
- Gold is "chunk contains the answer sentence". Overlap can duplicate a sentence in two neighbouring chunks (both then count as gold), and a few questions also count a later summary paragraph that restates the fact.
- Single run; latency comes from one pass on one CPU. Generation uses temperature 0.2 and cached answers.
- Answer quality comes from LLM judges only; the human spot-check covers only 10 items, so judge bias is only lightly checked (10 items, all scored 5/5 by the author).

## Pilot (v1): 2 documents, 18 questions
Kept unchanged below for reference (only heading levels were demoted). Both methods scored Hit@3 = 18/18, so the pilot could not separate them, which is why v2 was built. Note that `eval/corpus/` now also holds the v2 documents, so rerunning `python -m eval.run_eval` would no longer reproduce these numbers.

### Setup
- **Corpus:** 2 short documents in `eval/corpus/` (Apollo program, photosynthesis), ingested into one notebook (about 17 chunks of ~900 chars).
- **Questions:** 18 in `eval/questions.json`: 10 direct questions and 8 paraphrased ones with little word overlap with the source text.
- **Methods** (top-k = 3):
  - `vector`: embed query with all-MiniLM-L6-v2, Chroma cosine top-3.
  - `rerank`: Chroma top-20 candidates, re-scored by cross-encoder `ms-marco-MiniLM-L-6-v2`, top-3 kept.
- **Generation:** same LLM (Groq `openai/gpt-oss-120b`) and prompt for both methods.
- **Run:** `python -m eval.run_eval` (full per-question log with retrieved chunks, answers and timings: `eval/results/eval_log.json`).
- **Metrics:** Hit@3 (gold passage in top-3), rank of the gold passage, MRR, keyword coverage of the generated answer, retrieval latency (models warmed up first), and end-to-end latency (retrieval + LLM).

### Results

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

### Answer quality (manual check)
`manual_quality` in the log is left for hand scoring. I read a sample of answers from both methods: they were correct and grounded in the retrieved passages, and all 36 answers contain `[n]` citations. The one keyword-coverage miss (rerank, 0.50 on the "extremely hot" question) is a false negative of the metric: the answer says the heat "damages the enzymes" instead of using the word "denature".

### Conclusion
- **On this corpus the two methods are effectively tied.** Both found the gold passage for 18/18 questions, and answer quality was identical. Reranking improved one question (rank 2 -> 1) and hurt one (rank 2 -> 3), so MRR moved 0.94 -> 0.96, which is within noise at n = 18.
- **Cost:** reranking added little latency here (about 30 ms on average in this run) because only 20 candidates are scored on a CPU with a small model. The cross-encoder adds a second model download (~90 MB) and memory.
- **Why no clear win:** the notebook is tiny, so top-3 of ~17 chunks almost always contains the answer. Reranking is expected to matter more with many sources, where the first-stage top-k misses relevant chunks.
- **Recommendation:** keep reranking available (it is the UI default and helps on paraphrased queries such as the moonwalkers one), but this evaluation does not prove it is better. A larger corpus and question set would be needed for a statistically meaningful comparison.

### Limitations
- Small, self-written corpus and question set; results are not generalizable.
- "Gold passage" detection uses exact-phrase matching, which can undercount a correct chunk phrased differently.
- Keyword coverage is a coarse proxy for answer quality; the LLM is non-deterministic (temperature 0.2).
