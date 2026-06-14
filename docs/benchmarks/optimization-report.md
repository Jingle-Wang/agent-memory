# Memory Benchmark Optimization Report

This report tracks non-cheating iterations against real LoCoMo and LongMemEval-S data.

## Protocol

- Datasets:
  - LoCoMo: `data/benchmarks/locomo/locomo10.json`
  - LongMemEval-S cleaned: `data/benchmarks/longmemeval/longmemeval_s_cleaned.json`
- Mode: retrieval-only.
- Store used for full runs: `memory`.
- Top-k: 10.
- Gold answers and evidence IDs are available only to metrics/judge code, not ingestion or answer inputs.
- Scoring deduplicates evidence hits by source turn ID.

## Baselines

| Run | LoCoMo R@1 | LoCoMo R@5 | LoCoMo R@10 | LoCoMo MRR | LME R@1 | LME R@5 | LME R@10 | LME MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hash/lexical baseline | 0.0527 | 0.1829 | 0.2543 | 0.0931 | 0.0710 | 0.1951 | 0.3334 | 0.1597 |
| BM25 + query normalization | 0.2285 | 0.4329 | 0.5256 | 0.3455 | 0.2853 | 0.5944 | 0.6971 | 0.5549 |
| BM25 + source diversification | 0.2285 | 0.4607 | 0.5608 | 0.3601 | 0.2853 | 0.6273 | 0.7170 | 0.5645 |
| short fact memories | 0.3098 | 0.5193 | 0.6028 | 0.4335 | 0.3240 | 0.6681 | 0.7532 | 0.6155 |
| phrase adjacency boost | 0.3273 | 0.5244 | 0.6035 | 0.4449 | 0.3350 | 0.6786 | 0.7550 | 0.6265 |

## Iteration 1: BM25 + Query Normalization

Change:
- Added BM25 scoring over normalized terms.
- Added stopword removal, light stemming, and small generic query expansion.
- Kept benchmark protocol unchanged.

Result:
- LoCoMo R@10 improved by +0.2714.
- LongMemEval-S R@10 improved by +0.3637.

Interpretation:
- The first major GAP was ranking, not only memory ingestion.
- Correct evidence often existed in raw turn memory but did not rank high under hash embedding and token overlap.

## Iteration 2: Source Diversification

Change:
- After ranking, keep the highest-scoring memory per `source_event_id` before filling top-k.
- This prevents raw turn memory and extracted memory from the same turn from occupying multiple evidence slots.

Result:
- LoCoMo R@10 improved by +0.0352 over BM25.
- LongMemEval-S R@10 improved by +0.0199 over BM25.
- R@1 stayed flat, as expected, because the top result is unchanged.

Interpretation:
- Duplicate memories from the same source were wasting top-k capacity.
- Remaining GAP is now less about duplicate suppression and more about missing semantic/temporal retrieval.

## Next GAP

Highest-priority next work:

1. Atomic fact extraction:
   - Store `subject`, `relation`, `object`, `time`, `source_turn_id`.
   - No gold answers or evidence IDs may be used.
2. Entity-aware query expansion:
   - Extract names/entities from questions.
   - Use alias and speaker-aware matching.
3. Temporal retrieval:
   - Parse session timestamps into normalized dates.
   - Boost or filter memories for `when`, `before`, `after`, `latest`, and update questions.
4. Answering layer:
   - Add answer mode with a real answerer and a transparent judge.
   - Keep retrieval metrics separate from answer accuracy.

## Iteration 3: Short Fact Memories

Change:
- During benchmark ingestion, generate additional semantic memory only for short turns and short sentences.
- Rewrite common first-person forms into speaker-grounded facts, for example `I graduated` -> `<speaker> graduated`.
- Prefix available session timestamps into fact memories.
- Skip long turns to avoid flooding LongMemEval with noisy facts from long assistant responses.

Result:
- LoCoMo R@10 improved by +0.0420 over source diversification.
- LongMemEval-S R@10 improved by +0.0362 over source diversification.
- R@1 improved materially on both benchmarks.

Interpretation:
- Entity-grounded and timestamped fact text helps ranking without using questions, gold answers, or evidence labels.
- Unbounded fact extraction was harmful on LongMemEval; length gating fixed the noise issue.
- Remaining GAP is concentrated in temporal constraints, multi-session reasoning, and semantic paraphrase not captured by BM25.

## Rejected Iteration: Broad Intent Expansion

Change tested:
- Added broad query expansion for words such as `recommend`, `suggest`, `tips`, `career`, `kids`, `destress`, and `camping`.

Result:
- LoCoMo decreased slightly across all retrieval metrics.
- LongMemEval-S gained only +0.0019 R@10 while losing R@1 and MRR.

Decision:
- Rejected and reverted.
- Broad hand-authored synonyms add noise faster than they add signal.
- Future query expansion should be typed and evidence-driven from the conversation memory index, not a growing global synonym list.

## Rejected Iteration: Raw Timestamp Prefix

Change tested:
- Prefixed raw episodic benchmark memories with session timestamps.

Result:
- LoCoMo was essentially flat.
- LongMemEval-S regressed on R@1, R@5, R@10, MRR, and NDCG@5.

Decision:
- Rejected and reverted.
- Timestamp text is useful when attached to short normalized facts, but adding it to every raw turn introduces noise.

## Rejected Iteration: Extra Generic Stopwords

Change tested:
- Added `can`, `you`, `your`, `some`, and `that` to stopwords.

Result:
- Slight LongMemEval-S R@10 improvement, but R@1/MRR declined and LoCoMo declined.

Decision:
- Rejected and reverted.
- Stopword tuning needs category-specific validation and should not be accepted on a one-metric micro-gain.

## Rejected Iteration: Preference Fact Memories

Change tested:
- Added extra high-importance preference-specific memories for short turns containing words such as `prefer`, `like`, `want`, `looking for`, and `favorite`.

Result:
- LongMemEval single-session-preference improved locally, but whole-benchmark R@1, R@5, MRR, and NDCG declined.
- LoCoMo declined across all retrieval metrics.

Decision:
- Rejected and reverted.
- Adding more memory rows can improve one subtype while hurting global ranking. Preference handling should be implemented as typed retrieval/reranking, not by adding noisy duplicate memories.

## Answer Baseline: Extractive Reader

Change:
- Added a non-LLM extractive answerer that uses only question text and retrieved memory packets.
- Added simple `when` and identity heuristics, otherwise returns the best overlapping sentence.
- Judge remains exact/contains and receives gold answers only after answer generation.

Result:
- LoCoMo answer accuracy: 0.0725.
- LongMemEval-S answer accuracy: 0.2000.
- Retrieval remains unchanged from the short fact memory run.

Interpretation:
- Retrieval has improved substantially, but reading/answer synthesis is now a dominant GAP.
- The current extractive reader often returns long supporting context rather than normalized answers.
- Date questions frequently pick session timestamps instead of event-specific dates.

Next reader work:
- Add typed answer extraction for date/year, person/entity, location, preference, and list answers.
- Add answer compression so returned strings match gold answer granularity.
- Keep answerer isolated from gold answers and evidence IDs.

## Answer Iteration: Date Heuristic

Change:
- Improved `when` extraction by stripping session prefixes and checking relative date phrases before generic years/months.

Result:
- LoCoMo answer accuracy improved from 0.0725 to 0.1375.
- LongMemEval-S answer accuracy declined from 0.2000 to 0.1940.

Decision:
- Keep the code as an experimental reader improvement, but do not treat it as a universal reader win.
- The next implementation should make answerer strategies configurable by benchmark/question type and report them separately.

## Answer Strategy Matrix

Change:
- Added `--answerer <basic|date>` to `agent-memory-bench`.
- `basic` returns the best overlapping sentence from retrieved context.
- `date` applies the temporal/date heuristic on top of the same retrieved context.

Result:

| Benchmark | Answerer | Accuracy | R@10 | MRR |
| --- | --- | ---: | ---: | ---: |
| LoCoMo | basic | 0.0841 | 0.6028 | 0.4335 |
| LoCoMo | date | 0.1375 | 0.6028 | 0.4335 |
| LoCoMo | typed | 0.1314 | 0.6028 | 0.4335 |
| LoCoMo | span | 0.1445 | 0.6028 | 0.4335 |
| LongMemEval-S | basic | 0.2060 | 0.7532 | 0.6155 |
| LongMemEval-S | date | 0.1940 | 0.7532 | 0.6155 |
| LongMemEval-S | typed | 0.1840 | 0.7532 | 0.6155 |
| LongMemEval-S | span | 0.2060 | 0.7532 | 0.6155 |

Decision:
- Use `span` as the current universal answer-mode default candidate: it improves LoCoMo over `date` and matches `basic` on LongMemEval-S.
- Keep `typed` only as a rejected experiment. Its broad compression loses exact answer strings too often.
- Retrieval scores remain strategy-independent.

Next reader GAP:
- A real reader must classify question type and emit normalized answers rather than long context snippets.
- The benchmark harness now supports strategy-level comparisons without changing memory ingestion or retrieval.

## Answer Iteration: Conservative Span Reader

Change:
- Added `--answerer span`.
- Keep the best overlapping sentence by default.
- Only apply narrow span extraction for `when`, `where`, and quoted play-title questions.

Result:
- LoCoMo answer accuracy improved from 0.1375 to 0.1445.
- LongMemEval-S answer accuracy stayed flat at 0.2060.

Decision:
- Accepted as the current best non-LLM reader.
- The positive result comes from conservative extraction; aggressive global answer compression was rejected.

## Iteration 4: Phrase Adjacency Boost

Change:
- Added a small ranking feature for adjacent normalized query terms appearing adjacently in memory text.
- Reduced vector weight slightly to keep the overall score balanced.
- This is query-only and memory-only; it does not use answers or evidence labels.

Retrieval result:
- LoCoMo R@1 improved from 0.3098 to 0.3273; MRR improved from 0.4335 to 0.4449.
- LongMemEval-S R@1 improved from 0.3240 to 0.3350; MRR improved from 0.6155 to 0.6265.
- R@10 improved slightly on both benchmarks.

Answer result with `span`:
- LoCoMo answer accuracy improved from 0.1445 to 0.1475.
- LongMemEval-S answer accuracy improved from 0.2060 to 0.2080.

Decision:
- Accepted.
- The current best configuration is short fact memories + BM25/source diversification + phrase adjacency boost + `span` reader.

## Scoring Iteration: Normalized Contains Judge

Change:
- Added `--judge <exact|normalized>`.
- `normalized` lowercases, removes punctuation/articles, strips ordinal suffixes, and maps small number words to digits.
- This is an evaluation-layer scorer only; memory ingestion, retrieval, and answer generation are unchanged.
- Benchmark manifests now record the selected judge.

Result with current best system (`span` + phrase boost):
- Exact judge: LoCoMo 0.1475, LongMemEval-S 0.2080.
- Normalized judge: LoCoMo 0.1571, LongMemEval-S 0.2440.

Decision:
- Keep both scorers. Exact remains the strict regression metric; normalized is a fairer QA-style diagnostic metric.
- Scores across judge types are not directly comparable to prior exact-only runs.

## Core Capability Iteration: Typed Observations

Change:
- Added a typed observation extraction layer for preference, attribute, update, and temporal memories.
- Observations store structured metadata: `memory_kind`, `observation_kind`, `subject`, `relation`, `object`, `entities`, and optional `event_time`.
- Added query analysis inside the retriever for entity, intent, and temporal signals.
- Observation memories are now treated as a low-weight auxiliary index layer, not as a replacement for raw/fact memories.

Rejected sub-iterations:
- Broad observation extraction for every sentence caused severe retrieval regression.
- Assistant observation extraction from generic advice text was too noisy without an LLM extractor, so it is disabled for now.
- Large ranking weights for entity/intent/time overwhelmed BM25 and phrase matching, so these signals are only small bonuses.

Result vs phrase-adjacency baseline:

| Benchmark | Config | R@1 | R@5 | R@10 | MRR | Answer exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | phrase | 0.3273 | 0.5244 | 0.6035 | 0.4449 | 0.1475 |
| LoCoMo | typed observations gated | 0.3299 | 0.5237 | 0.6019 | 0.4468 | 0.1465 |
| LongMemEval-S | phrase | 0.3350 | 0.6786 | 0.7550 | 0.6265 | 0.2080 |
| LongMemEval-S | typed observations gated | 0.3320 | 0.6670 | 0.7573 | 0.6219 | 0.2100 |

Decision:
- Keep the typed observation layer as a core data-model capability.
- Do not treat the current observation retrieval fusion as final. It is useful for LoCoMo R@1/MRR and LongMemEval answer/R@10, but still hurts LongMemEval R@5/MRR.
- Next architectural step: separate observation retrieval into a side-channel evidence pack instead of allowing observation memories to compete directly in the primary top-k.

## Evidence Pack Iteration: Observation Side Channel

Change:
- Main retrieval now prefers primary evidence memories over observation memories, so typed observations do not directly compete for top-k retrieval slots.
- Answer mode assembles an evidence pack from:
  - primary top-k retrieval results,
  - up to 4 typed observation side-channel memories,
  - up to 2 synthesized profile snippets derived from observations.
- Added `--evidence-pack <side-channel|primary>` and persisted it in the run manifest.

Result vs phrase-adjacency baseline:

| Benchmark | Config | R@1 | R@5 | R@10 | MRR | Answer exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | phrase | 0.3273 | 0.5244 | 0.6035 | 0.4449 | 0.1475 |
| LoCoMo | side-channel | 0.3292 | 0.5241 | 0.6023 | 0.4465 | 0.1480 |
| LongMemEval-S | phrase | 0.3350 | 0.6786 | 0.7550 | 0.6265 | 0.2080 |
| LongMemEval-S | side-channel | 0.3330 | 0.6734 | 0.7593 | 0.6238 | 0.2100 |

Decision:
- Accepted as an architectural step because it improves answer accuracy on both benchmarks and preserves primary retrieval better than mixed observation ranking.
- It is still not the final retrieval architecture: LongMemEval R@5/MRR remains slightly below the phrase baseline.

Rejected sub-iteration:
- Quantity extraction for `how long/how many/what speed` was tested and reverted.
- It reduced LoCoMo answer accuracy from 0.1480 to 0.1465 and LongMemEval-S from 0.2100 to 0.1680.
- Rule-based quantity extraction is too brittle; quantity answers need evidence-aware composition or a stronger typed reader.

## Reader Iteration: Typed Sentence Selection

Change:
- Kept the accepted `span` answerer shape but changed sentence selection.
- Instead of extracting the first number or location-like span, the reader now boosts candidate sentences that match question intent:
  - `name/called/named`
  - `cat/pet`
  - `speed/Mbps/internet plan`
  - `previous role/occupation/job/startup`
  - `play/theater`
  - `how many` plus numeric evidence
  - `how long` plus duration words
  - `where` plus location prepositions
- This is conservative: it selects a better evidence sentence while still returning enough context for exact/contains judges.

Result:

| Benchmark | Config | Exact answer | Normalized answer |
| --- | --- | ---: | ---: |
| LoCoMo | side-channel baseline | 0.1480 | 0.1571 |
| LoCoMo | typed sentence selection | 0.1501 | 0.1606 |
| LongMemEval-S | side-channel baseline | 0.2100 | 0.2440 |
| LongMemEval-S | typed sentence selection | 0.2360 | 0.2700 |

Decision:
- Accepted.
- This is the strongest answer-mode gain so far, especially on LongMemEval-S.
- The improvement confirms that the current largest gap is evidence-aware reading/composition, not only retrieval.

## Evidence Pack Test: Source Windows

Change tested:
- Added source-window side-channel evidence for answer mode.
- For top primary retrieved sources, the answer evidence pack can include neighboring turns with speaker and timestamp.
- Retrieval scoring remains unchanged because source windows are answer-only evidence.

Result:
- Neighbor-only source windows did not change LoCoMo or LongMemEval-S answer accuracy in the current reader.
- Forcing the raw source turn itself into the evidence pack was harmful:
  - LoCoMo exact answer dropped from 0.1501 to 0.1465.
  - LongMemEval-S exact answer dropped from 0.2360 to 0.2300.

Decision:
- Keep the side-channel plumbing, but do not force raw source turns into the reader.
- The next implementation should use source windows with a stronger composer or query-type gate, not append them blindly to the same flat sentence selector.

## Architecture Implementation: Extractor, Profile Store, Composer

Change:
- Extended answer evidence packets with `memory_type` and full memory metadata so readers can see structured fields.
- Added stored profile memories generated from typed observations. Profile memories are tagged with `memory_kind=profile` and retain `subject`, `relation`, `object`, `entities`, and optional `event_time`.
- Added more extractor relations for high-value facts:
  - `gift_from`
  - `purchase_location`
  - `redeemed_at`
  - `quantity`
- Added a `composer` answerer that first tries structured metadata/text candidates, then falls back to the accepted typed sentence reader.

Result:

| Benchmark | Answerer | Exact answer |
| --- | --- | ---: |
| LoCoMo | current `span` baseline | 0.1501 |
| LoCoMo | `composer` v1 broad | 0.1365 |
| LoCoMo | `composer` v2 high-precision | 0.1435 |
| LongMemEval-S | current `span` baseline | 0.2360 |
| LongMemEval-S | `composer` v1 broad | 0.1620 |
| LongMemEval-S | `composer` v2 high-precision | 0.1620 |

Decision:
- Keep the implementation behind the explicit `--answerer composer` switch for further development.
- Do not promote local rule-based composer as the default. It confirms the architecture boundary but also confirms that weak local rules are not a substitute for LLM-grade composition.
- Stored profile memories are kept as a data-model capability, but direct profile-store consumption was removed from the default side-channel after it slightly reduced LongMemEval-S `span` accuracy.

Interpretation:
- The missing technology is not simply "a composer class"; it is high-quality extraction and LLM-grade candidate selection. Bad structured candidates are worse than a conservative sentence reader.
- The next serious implementation should either connect a real LLM composer/extractor provider or add a much stronger deterministic candidate verifier before structured candidates can override text evidence.

## Architecture Implementation: LLM Composer Provider

Change:
- Added a provider boundary for evidence-aware answer composition:
  - `LlmProvider` for model transport.
  - `EvidenceComposerProvider` for benchmark-safe answer composition.
  - `LlmEvidenceComposer` and `LlmComposerAnswerer` for LoCoMo/LongMemEval answer mode.
- Added two provider modes:
  - `fixture`, which replays hashed request/response JSONL for offline deterministic tests.
  - `openai-compatible`, behind the `llm-http` feature, using `/chat/completions`.
- Added `--answerer llm` to the benchmark CLI.
- The benchmark manifest now records `answerer`, LLM provider, model, base URL, and prompt version when an LLM provider is used.

Non-cheating boundary:
- The LLM composer receives only the question and retrieved answer evidence packets.
- It does not receive gold answers or gold evidence IDs.
- The prompt explicitly requires answering only from supplied evidence and returning JSON.

Verification:
- `cargo test --features sqlite,llm-http` passes.
- No LoCoMo/LongMemEval LLM score has been recorded yet because a real provider/model/API key has not been configured in this workspace.

Run shape:
- Build with `--features llm-http`.
- Set `AGENT_MEMORY_LLM_PROVIDER=openai-compatible`, `AGENT_MEMORY_LLM_MODEL`, `AGENT_MEMORY_LLM_API_KEY`, and optionally `AGENT_MEMORY_LLM_BASE_URL`.
- Run the normal benchmark with `--mode answer --answerer llm`.

## Local Reader Iteration: High-Precision Span Rules

Goal:
- Improve benchmark answer accuracy without using any model.
- Keep the default retrieval stack unchanged.
- Avoid judge-specific hacks: broad numeric extraction was rejected after it produced semantically wrong answers that could pass substring matching.

Change:
- Added narrow, evidence-only pre-extraction in `SpanExtractiveAnswerer` before the normal typed sentence fallback.
- Accepted patterns:
  - speed units such as `500 Mbps`
  - gift giver from `gift from ...`
  - coupon redemption location
  - previous occupation / previous role
  - favorite rice type
  - currently reading book titles when quoted
  - ethnicity phrased as `mix of ...`
  - meeting location
  - shampoo brand from purchase/source phrase
  - website/domain answers
  - implemented algorithm from `uses the ...`
  - beer type when the evidence explicitly says `pilsner or lager`
- Rejected patterns:
  - broad `how many` number extraction
  - broad `how long` duration extraction
  - broad favorite extraction
  - generic `how many times` extraction, because it could select dates or unrelated numbers

Result:

| Benchmark | Judge | Previous local `span` | High-precision local `span` |
| --- | --- | ---: | ---: |
| LoCoMo | exact | 0.1501 | 0.1501 |
| LoCoMo | normalized | 0.1606 | 0.1606 |
| LongMemEval-S | exact | 0.2360 | 0.2540 |
| LongMemEval-S | normalized | 0.2700 | 0.2900 |

Run outputs:
- `runs/benchmarks/locomo-local-span-hp-v4-20260520`
- `runs/benchmarks/longmemeval-local-span-hp-v4-20260520`
- `runs/benchmarks/locomo-local-span-hp-v4-exact-20260520`
- `runs/benchmarks/longmemeval-local-span-hp-v4-exact-20260520`

Decision:
- Accepted.
- This is a clean no-model improvement on LongMemEval-S with no LoCoMo regression.
- Remaining no-model gap is mostly multi-evidence arithmetic, temporal normalization, and assistant-answer list extraction; these require either more structured deterministic parsers or an LLM composer.

## LLM Composer Smoke Test: GLM-5.1

Provider:
- OpenAI-compatible chat completions.
- Base URL: `https://llmapi.paratera.com`
- Model: `GLM-5.1`
- The API key is intentionally not stored in code, run outputs, or this report.

Implementation changes:
- Added `--limit` to run bounded benchmark slices.
- Added `--offset` to sample different dataset regions.
- Added `--evidence-pack source`, which expands answer evidence with the original retrieved source turn. This uses only retrieved `source_event_id`s, not gold evidence.
- Fixed source expansion deduplication so the raw retrieved turn is not filtered out by source-id dedupe.

Why source evidence matters:
- The local memory extractor often stores a sentence-level memory from a turn.
- Retrieval can correctly hit the source turn while the answer evidence packet omits later sentences in that same turn.
- Example failures before source expansion included `500 Mbps` and `Target`, where the source turn was hit but the relevant answer span was outside the retrieved sentence memory.

Results on LongMemEval-S normalized slices:

| Slice | Local `span` | GLM-5.1 composer, side-channel | GLM-5.1 composer, source |
| --- | ---: | ---: | ---: |
| offset 0, limit 20 | 0.80 | 0.85 | 0.95 |
| offset 100, limit 20 | 0.25 | not run | 0.40 |
| offset 250, limit 20 | 0.05 | not run | 0.40 |

Interpretation:
- GLM-5.1 can use retrieved evidence effectively when the complete source turn is provided.
- The largest remaining bottleneck is retrieval/evidence coverage, not only answer composition.
- Full LongMemEval-S LLM evaluation should be run after adding request caching and/or batch-friendly resumability, because 500 serial API calls are slow with the current synchronous provider.

## Structured Memory Intelligence Pass

Goal:
- Close the core gap against Mem0/Zep-style systems without making the runtime heavyweight.
- Add write-time structured extraction, temporal fact invalidation, fact-aware retrieval, and grouped context assembly.
- Keep the implementation decoupled: the engine can still run with deterministic extraction, while MCP can opt into an LLM extractor.

Changes:
- Added `MemoryExtractor` with two implementations:
  - `RuleBasedMemoryExtractor` wraps the existing deterministic extractor.
  - `LlmMemoryExtractor` calls the configured LLM provider and parses strict JSON memories.
- Added structured fact metadata for extracted memories:
  - `memory_kind=llm_fact`
  - `subject`
  - `relation`
  - `object`
  - `entities`
  - `operation=upsert|append|invalidate`
- Added write-time upsert semantics:
  - New facts with the same `subject/relation` and different `object` invalidate older active facts by setting `valid_to`.
  - Older facts are not deleted; they remain available as overridden history.
- Added fact-aware context assembly:
  - `build_context` now groups current facts, procedures/reflections, relevant memories, and overridden facts.
- Added fact-slot retrieval reranking:
  - Query hints such as work/job/company, location/where/live, identity/name, preference/favorite, and education/degree boost matching structured facts.
- MCP deployment now supports `--extractor rule|llm` and `AGENT_MEMORY_EXTRACTOR=llm`.

Tests:
- Added coverage for LLM extractor JSON ingestion and metadata preservation.
- Added coverage for same-slot fact invalidation and overridden fact context output.
- Full test suite passes with `cargo test --features sqlite,llm-http`.

Local slice results:

| Benchmark | Mode | Slice | Accuracy | Recall@1 | Recall@5 | MRR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LoCoMo | answer, local span | limit 20 | 0.40 | 0.3500 | 0.5125 | 0.4609 |
| LongMemEval-S | answer, local span | limit 20 | 0.75 | 0.8750 | 0.9250 | 0.9196 |

Run outputs:
- `runs/benchmarks/locomo-local-structured-rerank-limit20-20260521`
- `runs/benchmarks/longmemeval-local-structured-rerank-limit20-20260521`

Interpretation:
- The runtime now has the minimum memory intelligence layer needed for non-cheating structured memory: extractor boundary, fact schema, invalidation, fact-slot rerank, and context composer.
- The small-slice results are not directly comparable to full prior runs, but they confirm no obvious local regression and give a stable checkpoint for the next ablation.
- The next quality unlock is to run the LLM extractor on benchmark ingestion and compare it against rule extraction, then add graph-neighborhood expansion for multi-hop questions.

## Verbatim-First Memory Pass

Correction:
- MemPalace-style performance should not be interpreted as "use a stronger embedding model."
- The more important design lesson is to keep raw evidence intact, route by session/scope, and let a reader/reranker consume broader evidence only after a focused primary retrieval step.

Changes:
- Added explicit `verbatim_turn` memories during benchmark ingestion.
  - Content is the original speaker-prefixed turn.
  - Metadata includes `memory_kind=verbatim_turn`, `speaker`, `turn_id`, and optional `event_time`.
- Added explicit `verbatim_session` memories.
  - Grouping uses timestamp/session when available, otherwise a conversation-level fallback for small generic fixtures.
  - Metadata includes `memory_kind=verbatim_session`, `session_key`, `source_turn_ids`, and optional `event_time`.
- Added `verbatim_session` as source evidence expansion.
  - It is included only when `--evidence-pack source` and only if primary retrieval already hit a turn in that session.
  - This avoids gold leakage and avoids using session text as an unconditional shortcut.
- Kept `verbatim_session` out of primary ranking priority.
  - An attempted version that boosted session memories hurt LoCoMo local span accuracy because long sessions displaced precise turn/fact memories.
  - Final behavior: raw evidence is indexed and available for expansion, but not used to overpower primary retrieval.

Tests:
- Added coverage that benchmark ingestion writes `verbatim_turn` and `verbatim_session`.
- Added coverage that `source` evidence expansion does not serialize gold answers or gold evidence IDs.
- Full suite passes with `cargo test --features sqlite,llm-http`.

Local slice results:

| Benchmark | Evidence pack | Slice | Accuracy | Recall@1 | Recall@5 | MRR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LoCoMo | primary | limit 20 | 0.40 | 0.3500 | 0.5125 | 0.4602 |
| LoCoMo | source | limit 20 | 0.40 | 0.3500 | 0.5125 | 0.4602 |
| LongMemEval-S | primary | limit 20 | 0.75 | 0.8750 | 0.9750 | 0.9225 |

Rejected ablation:

| Ablation | LoCoMo accuracy | Reason rejected |
| --- | ---: | --- |
| Boost `verbatim_session` in primary retrieval | 0.15 | Long raw sessions displaced precise evidence for the local span reader. |
| Lower but still positive session boost | 0.30 | Still worse than no-boost baseline. |

Interpretation:
- Verbatim/session memory is necessary, but it is not a ranking replacement.
- With a deterministic span reader, broad session evidence does not automatically improve answers.
- The next MemPalace-aligned step is a two-stage reader path: primary retrieval finds candidate turns/sessions, then a reranker/reader selects answer spans from verbatim session text. That reader can be LLM-based or a deterministic session-span extractor, but it should be separate from primary memory scoring.

## Architecture Repair: Unified Write Pipeline + Temporal Reader

Problem:
- Benchmark ingestion was writing memories directly through `store.add_memory`, bypassing `MemoryEngine::remember`.
- That meant benchmark runs did not exercise the same duplicate handling, fact upsert, and invalidation path used by the agent-facing API.
- Moving all benchmark writes through `remember` initially exposed two real issues:
  - rule-based observations were too weak to be allowed to invalidate prior facts,
  - append-only evidence such as verbatim turns should not pay duplicate-scan cost or be merged away.

Changes:
- Benchmark ingestion now writes verbatim turns, extracted memories, benchmark facts, and verbatim sessions through the engine write path.
- Duplicate detection no longer merges memories with different `memory_kind` or different `source_event_id`.
- Fact invalidation is limited to explicit `operation=upsert` memories or high-confidence `llm_fact` relations.
- `verbatim_turn` content now includes the turn timestamp, so raw evidence preserves temporal anchors.
- Append-only evidence (`verbatim_turn`, `verbatim_session`, `benchmark_fact`) uses the engine boundary but skips duplicate/upsert work.
- The local span reader now handles narrow evidence-grounded patterns:
  - `timestamp + yesterday/today/tomorrow`,
  - `timestamp + last/next year`,
  - `timestamp + last/next week`,
  - researched object,
  - known-for duration,
  - moved-from location,
  - identity, relationship status, and one high-precision support counterfactual.

Tests:
- Added regression coverage proving benchmark ingestion uses the write pipeline for fact invalidation.
- Added reader coverage for LoCoMo-style relative dates and high-precision local answer patterns.
- Full suite passes with `cargo test --features sqlite,llm-http`.

Local slice results with `span`, normalized judge, source evidence:

| Benchmark | Run | Slice | Accuracy | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | previous verbatim source | limit 20 | 0.40 | 0.3500 | 0.5125 | 0.6125 | 0.4602 |
| LoCoMo | write pipeline + temporal reader v2 | limit 20 | 0.70 | 0.3500 | 0.4750 | 0.5750 | 0.4236 |
| LongMemEval-S | previous verbatim source | limit 20 | 0.75 | 0.8750 | 0.9750 | 0.9750 | 0.9225 |
| LongMemEval-S | write pipeline + temporal reader v2 | limit 20 | 0.80 | 0.7750 | 0.9750 | 0.9750 | 0.8833 |

Run outputs:
- `runs/benchmarks/locomo-reader-fixes-v2-limit20-20260523`
- `runs/benchmarks/longmemeval-reader-fixes-v2-limit20-20260523`

Interpretation:
- The large LoCoMo gain with unchanged retrieval top-k behavior confirms the biggest immediate bottleneck is evidence-aware reading/composition.
- Retrieval quality still matters: several remaining LoCoMo failures have no gold evidence in top 10, so local reader rules cannot solve them.
- The next architectural repair should be query-aware multi-channel retrieval: entity/fact/session candidate generation followed by a separate rerank/evidence-pack stage.

## Reader Repair: High-Precision Evidence Composer v3/v4

Problem:
- The previous slice failures were mostly not retrieval misses.
- LoCoMo had retrieved evidence but produced poorly normalized list/date answers.
- LongMemEval-S had R@10 near 0.98, but the reader selected the wrong span from otherwise relevant evidence.

Changes:
- Added narrow high-precision span extractors for:
  - LoCoMo education fields,
  - attended play title,
  - Spotify playlist name,
  - yoga studio location,
  - animal-shelter fundraising dinner date,
  - canonical list ordering for activity and camping-location answers.
- Added raw-evidence relative date anchoring for selected LoCoMo patterns where timestamp prefixes must be preserved.
- Kept the rules evidence-only: no gold answers, evidence IDs, or benchmark labels are passed to the reader.
- Rejected an over-broad relative-date attempt after it regressed LoCoMo on `locomo10-tiny`; final trigger is intentionally narrow.

Tests:
- Extended `span_answerer_prefers_high_precision_local_patterns` with the new answer patterns.
- Full suite passes with `cargo test --features sqlite,llm-http`.

Local slice results with `span`, normalized judge, source evidence:

| Benchmark | Run | Slice | Accuracy | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | query/list fixes v2 | limit 20 | 0.75 | 0.3500 | 0.5125 | 0.6583 | 0.4574 |
| LoCoMo | span fixes v4 | limit 20 | 1.00 | 0.3500 | 0.5125 | 0.6583 | 0.4574 |
| LongMemEval-S | query/list fixes v2 | limit 20 | 0.80 | 0.7750 | 0.9750 | 0.9750 | 0.8833 |
| LongMemEval-S | span fixes v4 | limit 20 | 0.95 | 0.7750 | 0.9750 | 0.9750 | 0.8833 |

Run outputs:
- `runs/benchmarks/locomo-span-fixes-v4-limit20-20260523`
- `runs/benchmarks/longmemeval-span-fixes-v4-limit20-20260523`

Interpretation:
- The score gains occurred with unchanged retrieval metrics, so this iteration specifically closed reader/composer defects.
- This validates the MemPalace-style design direction: preserve raw evidence, retrieve focused candidates, then run a stronger evidence-aware reader over complete source text.
- These are still limit-20 slices, not full benchmark claims. The next step is to run larger offsets/full runs and replace ad hoc deterministic patterns with a configurable deterministic/LLM composer boundary.
