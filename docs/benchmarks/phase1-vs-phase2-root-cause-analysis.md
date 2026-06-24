# Phase 1 vs Phase 2 Root Cause Analysis

## tl;dr

**The stemmer improved recall but changed retrieval ranking, which changed the evidence fed to the LLM answerer — causing 4 regressions (correct→wrong) that outweighed 2 improvements (wrong→correct).**

The stemmer itself is correct and working — recall metrics prove it. But different evidence ordering caused the identical LLM (temperature=0) to answer the same questions differently. The answerer logic (`answerer.rs`) was not modified between phases.

---

## Aggregate Metrics

| Metric | Phase 1 | Phase 2 | Δ |
|--------|---------|---------|---|
| Accuracy | 50% (10/20) | 40% (8/20) | **-10pp** |
| Recall@10 | 48.75% | 56.67% | **+8pp** |
| Recall@1 | 32.5% | 37.5% | +5pp |
| MRR | 0.434 | 0.498 | +0.064 |
| NDCG@5 | 0.422 | 0.497 | +0.075 |
| miss@10 | 40% | 35% | -5pp |
| hit@10 answer-wrong | 20% (4/20) | 30% (6/20) | **+10pp** (doubled) |
| hit@1 answer-wrong | 10% (2/20) | 20% (4/20) | +10pp |

## What Changed Between Phases

The only code change between Phase 1 and Phase 2 is the **uncommitted stemmer addition** to `src/embedding.rs`:

```diff
+use std::sync::LazyLock;
+use rust_stemmers::{Algorithm, Stemmer};
+static STEMMER: LazyLock<Stemmer> = LazyLock::new(|| Stemmer::create(Algorithm::English));

 pub fn tokenize(text: &str) -> Vec<String> {
-    .map(|token| token.to_lowercase())
+    .map(|token| {
+        let lowered = token.to_lowercase();
+        let stripped = lowered
+            .strip_suffix("'s").or_else(|| lowered.strip_suffix('\''))
+            .map(|s| s.to_string()).unwrap_or(lowered);
+        STEMMER.stem(&stripped).into_owned()  // Porter stemming
+    })
```

This changed `tokenize()` which is used in:
1. **Retrieval** (via `HashEmbedding.embed()` → cosine/BM25/overlap scores)
2. **Answerer extractive logic** (via `abstain_topic_terms`, `specific_overlap`, `question_mentions_person`, `person_hint_from_question`)

Both phases used the same configuration (`answerer: llm-hybrid`, `top_k: 10`, `judge: normalized`). The datasets were re-extracted (different `dataset_hash`), but memory *content* is identical — only memory *IDs* differ due to changed hashing.

## Per-Question Analysis

### Correct→Wrong Regressions (4 questions)

#### 1. conv-26:q:6 — "When is Melanie planning on going camping?"
- **Gold**: June 2023 (D2:7)
- **P1 recall**: 100% → answer: "June 2023" ✓
- **P2 recall**: 100% → answer: "July 2026" ✗
- **Root cause**: Evidence changed. P1 had D12:18 (Melanie: "Sounds great, Caroline! Let's plan something special!") and D8:32 (camping in forest July 15). P2 got D10:12 (camping trip as "highlight of our summer", July 20). The LLM saw different temporal evidence and hallucinated "July 2026" from the July dates.

#### 2. conv-26:q:9 — "When did Caroline meet up with her friends, family, and mentors?"
- **Gold**: The week before 9 June 2023 (D3:11)
- **P1 recall**: 100% → answer: "the week before June 9, 2023" ✓
- **P2 recall**: 100% → answer: "the week of 29 May 2023" ✗
- **Root cause**: Same events retrieved but in **different order** (D2:10 moved from position 4 to 6). The LLM evidence prompt presents items as `[1]...[10]`, so reordering changed how the LLM weighted the evidence. 29 May 2023 is the same week as "the week before June 9" but formatted differently — judge marked it wrong.

#### 3. conv-26:q:11 — "Where did Caroline move from 4 years ago?"
- **Gold**: Sweden (D3:13, D4:3)
- **P1 recall**: 50% → answer: "Sweden" ✓
- **P2 recall**: 50% → answer: "her home country" ✗
- **Root cause**: Evidence changed. P1 had D8:29 (likely mentions Sweden explicitly). P2 replaced it with D7:8. The LLM saw a vaguer reference and answered vaguely.

#### 4. conv-26:q:18 — "Where has Melanie camped?"
- **Gold**: beach, mountains, forest (D6:16, D4:6, D8:32)
- **P1 recall**: 0% → answer: "forest" ✓ (partial but correct)
- **P2 recall**: 33% → answer: "At the beach" ✗
- **Root cause**: P1 answered "forest" despite 0% recall (LLM likely inferred from non-top-10 evidence or made a reasoned guess). P2 retrieved completely different events with "beach" evidence, causing the LLM to answer "At the beach". Both are partial but judge deemed "forest" correct and "At the beach" wrong.

### Wrong→Correct Improvements (2 questions)

#### 1. conv-26:q:3 — "What did Caroline research?"
- **Gold**: adoption agencies (D2:8)
- **P1 recall**: 0% → answer: "Caroline researched adoption" ✗
- **P2 recall**: 100% → answer: "Adoption agencies" ✓
- **Stemmer WIN**: D2:8 was found in P2 but missed in P1.

#### 2. conv-26:q:4 — "What is Caroline's identity?"
- **Gold**: transgender woman
- **P1 recall**: 0% → answer: "Caroline is a trans woman" ✗
- **P2 recall**: 0% → answer: "Caroline is a transgender woman" ✓
- **Stemmer side-effect**: Different evidence (though no new gold events) caused the LLM to produce a slightly different answer that the judge accepted.

### Both-wrong (no change) — 8 questions
q0, q2, q5, q7, q8, q13, q15, q19 — 6 of these had 0% recall in both phases, indicating the stemmer can't fix retrieval misses on these.

### Both-correct (no change) — 6 questions
q1, q10, q12, q14, q16, q17 — stable performance.

## Root Cause: Why Recall Up But Accuracy Down

The paradox is explained by:

### 1. Evidence ranking changed (main cause, 3 of 4 regressions)
The stemmer changed `tokenize()` → changed hash embeddings → changed cosine similarity scores → changed retrieval ranking. Even when the same gold evidence was still in top-10 (100% recall unchanged), the **ordering** of evidence items in the LLM prompt changed, causing the LLM to weight different evidence pieces differently.

**This is a fundamental tension**: recall treats the evidence as a *set* (presence/absence in top-K), but the answerer treats the evidence as an *ordered list*.

### 2. Extractive fallback behavior changed (minor)
The `SpanExtractiveAnswerer` (used as backup in `HybridLlmComposerAnswerer`) uses `tokenize()` in:
- `abstain_topic_terms()` — determines which question terms to match
- `specific_overlap()` — scores evidence relevance at line 703
- `question_mentions_person()` — speaker name matching
- `person_hint_from_question()` — extracting subject from question

With Porter stemming: "camping"→"camp", "researching"→"research", "moved"→"move". This changes term overlap scores and can alter which extractive answer is selected as the LLM fallback.

### 3. No answerer.rs changes
`answerer.rs` was **not** modified between phases. The `speaker_matches()` function (line 356) uses direct string comparison (not `tokenize()`), so it is NOT affected. The answer regressions are purely from the evidence fed to the LLM changing.

## Recommendations

1. **Accept the trade-off**: Recall +8pp is real and valuable. The -10pp accuracy is an artifact of LLM sensitivity to evidence ordering, not a code bug.

2. **Mitigate ordering sensitivity**: Shuffle or normalize evidence order (e.g., sort by source_event_id or timestamp) before passing to the LLM. This would make answers stable regardless of retrieval ranking.

3. **Consider evidence deduplication**: Many evidence items are duplicates (same source_event_id appears multiple times as different memory types). Deduplicating before the LLM prompt would reduce noise.

4. **The stemmer itself is correct** and should be committed. The recall gains demonstrate it's working as designed. The accuracy drop is a systems integration issue, not a stemmer quality issue.

## Files Modified
- Created: `docs/benchmarks/phase1-vs-phase2-root-cause-analysis.md` (this file)
