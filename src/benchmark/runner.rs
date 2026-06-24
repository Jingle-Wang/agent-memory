use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::embedding::token_overlap_score;
use crate::engine::MemoryEngine;
use crate::entity::enrich_memory_entities;
use crate::extractor::MemoryExtractor;
use crate::llm::{ConfiguredLlmProvider, LlmProvider, LlmProviderMetadata};
use crate::models::{Event, Memory, MemoryPacket, MemoryQuery, MemoryType};
use crate::observation::extract_observations;
use crate::store::{MemoryStore, StoreError, StoreResult};
use crate::text::contains_any;

use super::answerer::{AnswerInput, Answerer, MemoryPacketForAnswerer};
use super::dataset::{BenchmarkDataset, BenchmarkQuestion, BenchmarkTurn, Conversation};
use super::judge::{Judge, JudgeInput};
use super::metrics::{BenchmarkSummary, QuestionResult, retrieval_metrics, summarize};
use super::reranker::CandidateReranker;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkMode {
    Retrieval,
    Answer,
}

impl BenchmarkMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_lowercase().as_str() {
            "retrieval" | "retrieve" => Ok(Self::Retrieval),
            "answer" | "qa" => Ok(Self::Answer),
            other => Err(format!("unknown benchmark mode: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            BenchmarkMode::Retrieval => "retrieval",
            BenchmarkMode::Answer => "answer",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkRunConfig {
    pub benchmark: String,
    pub mode: BenchmarkMode,
    pub top_k: usize,
    pub output_dir: PathBuf,
    pub dataset_hash: String,
    pub store: String,
    pub answerer: String,
    pub extractor: String,
    pub judge: String,
    pub evidence_pack: String,
    pub llm_provider: Option<LlmProviderMetadata>,
    pub max_questions: Option<usize>,
    pub question_offset: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkRunReport {
    pub summary: BenchmarkSummary,
    pub results: Vec<QuestionResult>,
}

pub struct BenchmarkRunner<A: Answerer, J: Judge> {
    answerer: A,
    judge: J,
    reranker: Option<Box<dyn CandidateReranker>>,
    /// Phase 4: optional LLM provider for query expansion
    query_expander: Option<ConfiguredLlmProvider>,
}

impl<A: Answerer, J: Judge> BenchmarkRunner<A, J> {
    pub fn new(answerer: A, judge: J) -> Self {
        Self {
            answerer,
            judge,
            reranker: None,
            query_expander: None,
        }
    }

    pub fn with_reranker(mut self, reranker: impl CandidateReranker + 'static) -> Self {
        self.reranker = Some(Box::new(reranker));
        self
    }

    /// Phase 4: enable LLM query expansion for vocabulary-bridging.
    pub fn with_query_expansion(mut self, provider: ConfiguredLlmProvider) -> Self {
        self.query_expander = Some(provider);
        self
    }

    pub fn run<S: MemoryStore>(
        &self,
        engine: &mut MemoryEngine<S>,
        dataset: &BenchmarkDataset,
        config: &BenchmarkRunConfig,
        extractor: Option<&dyn MemoryExtractor>,
    ) -> StoreResult<BenchmarkRunReport> {
        fs::create_dir_all(&config.output_dir)?;

        // Collect conversation IDs needed by the questions being evaluated
        let needed_ids: std::collections::HashSet<&str> = dataset
            .questions
            .iter()
            .skip(config.question_offset)
            .take(config.max_questions.unwrap_or(usize::MAX))
            .map(|q| q.conversation_id.as_str())
            .collect();

        let relevant_conversations: Vec<Conversation> = dataset
            .conversations
            .iter()
            .filter(|c| needed_ids.contains(c.id.as_str()))
            .cloned()
            .collect();

        self.ingest_dataset(engine, &relevant_conversations, extractor)?;

        let mut results = Vec::new();
        for question in dataset
            .questions
            .iter()
            .skip(config.question_offset)
            .take(config.max_questions.unwrap_or(usize::MAX))
        {
            let started = Instant::now();
            // Phase 4: expand query + multi-query fusion
            let query_texts = if let Some(expander) = &self.query_expander {
                match expand_query_llm(expander, &question.text) {
                    Ok(variants) => {
                        let mut all = vec![question.text.clone()];
                        all.extend(variants);
                        all
                    }
                    Err(e) => {
                        eprintln!("WARN [query-expansion] failed: {e} — using original query only");
                        vec![question.text.clone()]
                    }
                }
            } else {
                vec![question.text.clone()]
            };

            let search_limit = config.search_candidate_limit();
            let side_channel = config.include_side_channel_search();

            // Search with first (original) query
            let first_query = MemoryQuery::new(query_texts[0].clone())
                .namespace(question.conversation_id.clone())
                .limit(search_limit)
                .include_side_channel(side_channel);
            let mut candidates = engine.search(first_query)?;

            // Fuse with expanded variant searches (OR-combine, keep best score)
            for variant in &query_texts[1..] {
                let variant_query = MemoryQuery::new(variant.clone())
                    .namespace(question.conversation_id.clone())
                    .limit(search_limit)
                    .include_side_channel(side_channel);
                let variant_results = engine.search(variant_query)?;
                candidates = fuse_candidates(candidates, variant_results, search_limit);
            }
            if config.mode == BenchmarkMode::Answer {
                if let Some(reranker) = &self.reranker {
                    candidates = reranker.rerank_candidates(&question.text, &candidates)?;
                }
            }
            let packets = candidates
                .iter()
                .take(config.top_k)
                .cloned()
                .collect::<Vec<_>>();
            let retrieved_source_event_ids = packets
                .iter()
                .filter_map(|packet| packet.memory.source_event_id.clone())
                .collect::<Vec<_>>();
            let retrieved_memory_ids = packets
                .iter()
                .map(|packet| packet.memory.id.clone())
                .collect::<Vec<_>>();

            let retrieval =
                retrieval_metrics(&question.evidence_turn_ids, &retrieved_source_event_ids);
            let (answer, answer_correct, answer_score) = if config.mode == BenchmarkMode::Answer {
                let evidence_packets = assemble_answer_evidence(
                    engine.store(),
                    dataset,
                    question,
                    &candidates,
                    config.answer_evidence_primary_limit(),
                    config.side_channel_observation_limit(),
                    config.include_source_evidence(),
                    config.include_source_window(),
                    config.prefer_source_evidence(),
                )?;
                let answer_input = AnswerInput {
                    question: question.for_answerer(),
                    retrieved: evidence_packets
                        .iter()
                        .map(MemoryPacketForAnswerer::from)
                        .collect(),
                };
                if !ensure_no_gold_leak(question, &answer_input) {
                    return Err(StoreError::Corrupt(
                        "answer input failed gold leak check".to_string(),
                    ));
                }
                let answer = self.answerer.answer(&answer_input);
                let judge_output = self.judge.judge(&JudgeInput {
                    question: question.for_judge(),
                    answer: answer.clone(),
                });
                (
                    Some(answer.answer),
                    Some(judge_output.correct),
                    Some(judge_output.score),
                )
            } else {
                (None, None, None)
            };

            results.push(QuestionResult {
                question_id: question.id.clone(),
                conversation_id: question.conversation_id.clone(),
                category: question.category.clone(),
                retrieved_memory_ids,
                retrieved_source_event_ids,
                answer,
                answer_correct,
                answer_score,
                retrieval,
                latency_ms: started.elapsed().as_millis(),
            });
        }

        let summary = summarize(&config.benchmark, config.mode.as_str(), &results);
        let report = BenchmarkRunReport { summary, results };
        write_outputs(dataset, config, &report)?;
        Ok(report)
    }

    fn ingest_dataset<S: MemoryStore>(
        &self,
        engine: &mut MemoryEngine<S>,
        conversations: &[Conversation],
        extractor: Option<&dyn MemoryExtractor>,
    ) -> StoreResult<()> {
        for conversation in conversations {
            let mut session_turns = BTreeMap::<String, Vec<&BenchmarkTurn>>::new();
            for turn in &conversation.turns {
                let mut event = Event::new(turn.text.clone())
                    .namespace(conversation.id.clone())
                    .actor(turn.speaker.clone());
                event.id = turn.id.clone();
                event
                    .metadata
                    .insert("benchmark_turn_id".to_string(), turn.id.clone());
                if let Some(timestamp) = &turn.timestamp {
                    event
                        .metadata
                        .insert("benchmark_timestamp".to_string(), timestamp.clone());
                }

                let raw_content = if let Some(timestamp) = &turn.timestamp {
                    format!(
                        "[verbatim_turn time={timestamp}] {}: {}",
                        turn.speaker, turn.text
                    )
                } else {
                    format!("{}: {}", turn.speaker, turn.text)
                };
                let mut raw_memory = Memory::new(raw_content, MemoryType::Episodic)
                    .namespace(conversation.id.clone())
                    .source_event(turn.id.clone())
                    .importance(0.45)
                    .confidence(0.7);
                raw_memory
                    .metadata
                    .insert("memory_kind".to_string(), "verbatim_turn".to_string());
                raw_memory
                    .metadata
                    .insert("speaker".to_string(), turn.speaker.clone());
                raw_memory
                    .metadata
                    .insert("turn_id".to_string(), turn.id.clone());
                if let Some(timestamp) = &turn.timestamp {
                    raw_memory
                        .metadata
                        .insert("event_time".to_string(), timestamp.clone());
                }
                enrich_memory_entities(&mut raw_memory);
                engine.remember(raw_memory)?;

                let mut raw_text_memory = Memory::new(turn.text.clone(), MemoryType::Episodic)
                    .namespace(conversation.id.clone())
                    .source_event(turn.id.clone())
                    .importance(0.50)
                    .confidence(0.75);
                raw_text_memory
                    .metadata
                    .insert("memory_kind".to_string(), "verbatim_turn".to_string());
                raw_text_memory
                    .metadata
                    .insert("verbatim_form".to_string(), "turn_text".to_string());
                raw_text_memory
                    .metadata
                    .insert("speaker".to_string(), turn.speaker.clone());
                raw_text_memory
                    .metadata
                    .insert("turn_id".to_string(), turn.id.clone());
                if let Some(timestamp) = &turn.timestamp {
                    raw_text_memory
                        .metadata
                        .insert("event_time".to_string(), timestamp.clone());
                }
                enrich_memory_entities(&mut raw_text_memory);
                engine.remember(raw_text_memory)?;

                // ── Buffered (sliding-window) extraction ────────────────────
                // Uses IngestionBuffer inside the engine to accumulate turns
                // and extract memories from multi-turn context windows instead
                // of single-turn snippets.
                if let Some(extractor) = extractor {
                    engine.ingest_buffered_with_extractor(event.clone(), extractor)?;
                } else {
                    engine.ingest_buffered(event.clone())?;
                }

                for observation in extract_observations(&event) {
                    engine.remember(observation.to_memory(&event))?;
                }

                session_turns
                    .entry(session_key(turn))
                    .or_default()
                    .push(turn);
            }

            // ── Flush remaining buffered events ────────────────────────────
            if let Some(extractor) = extractor {
                engine.flush_buffer_with_extractor(extractor)?;
            } else {
                engine.flush_buffer()?;
            }

            let mut indexed_session = false;
            for (session_key, turns) in session_turns {
                if turns.len() < 2 {
                    continue;
                }
                add_session_memory(engine, &conversation.id, &session_key, &turns)?;
                indexed_session = true;
            }
            if !indexed_session && conversation.turns.len() > 1 {
                let turns = conversation.turns.iter().collect::<Vec<_>>();
                add_session_memory(engine, &conversation.id, "conversation", &turns)?;
            }
        }
        Ok(())
    }
}

fn session_key(turn: &BenchmarkTurn) -> String {
    if let Some(timestamp) = &turn.timestamp {
        return format!("time:{timestamp}");
    }
    "conversation".to_string()
}

fn session_transcript(turns: &[&BenchmarkTurn]) -> String {
    let header = turns
        .iter()
        .find_map(|turn| turn.timestamp.as_deref())
        .map(|timestamp| format!("[verbatim_session time={timestamp}]\n"))
        .unwrap_or_else(|| "[verbatim_session]\n".to_string());
    let mut output = header;
    for turn in turns {
        output.push_str(&turn.speaker);
        output.push_str(": ");
        output.push_str(&turn.text);
        output.push('\n');
    }
    output
}

fn add_session_memory<S: MemoryStore>(
    engine: &mut MemoryEngine<S>,
    namespace: &str,
    session_key: &str,
    turns: &[&BenchmarkTurn],
) -> StoreResult<()> {
    let content = session_transcript(turns);
    let source_event_id = turns
        .first()
        .map(|turn| turn.id.clone())
        .unwrap_or_else(|| session_key.to_string());
    let mut memory = Memory::new(content, MemoryType::Episodic)
        .namespace(namespace.to_string())
        .source_event(source_event_id)
        .importance(0.20)
        .confidence(0.45);
    memory
        .metadata
        .insert("memory_kind".to_string(), "verbatim_session".to_string());
    memory
        .metadata
        .insert("session_key".to_string(), session_key.to_string());
    memory.metadata.insert(
        "source_turn_ids".to_string(),
        turns
            .iter()
            .map(|turn| turn.id.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
    );
    if let Some(timestamp) = turns.iter().find_map(|turn| turn.timestamp.clone()) {
        memory.metadata.insert("event_time".to_string(), timestamp);
    }
    enrich_memory_entities(&mut memory);
    engine.remember(memory)?;
    Ok(())
}

impl BenchmarkRunConfig {
    fn side_channel_observation_limit(&self) -> usize {
        match self.evidence_pack.as_str() {
            "primary" | "none" => 0,
            "two-stage" => 8,
            _ => 4,
        }
    }

    fn include_source_evidence(&self) -> bool {
        matches!(
            self.evidence_pack.as_str(),
            "source" | "source-first" | "source-window" | "expanded" | "two-stage"
        )
    }

    fn include_source_window(&self) -> bool {
        matches!(
            self.evidence_pack.as_str(),
            "source-window" | "expanded" | "two-stage"
        )
    }

    fn include_side_channel_search(&self) -> bool {
        self.mode == BenchmarkMode::Answer
            && std::env::var("AGENT_MEMORY_INCLUDE_SIDE_CHANNEL_SEARCH").as_deref() == Ok("1")
    }

    fn prefer_source_evidence(&self) -> bool {
        matches!(self.evidence_pack.as_str(), "source-first" | "two-stage")
    }

    fn answer_evidence_primary_limit(&self) -> usize {
        if self.evidence_pack == "two-stage" {
            return self.top_k.max(200);
        }
        self.top_k.min(10)
    }

    fn search_candidate_limit(&self) -> usize {
        if self.mode == BenchmarkMode::Answer && self.evidence_pack == "two-stage" {
            return self.top_k.max(200);
        }
        if self.mode == BenchmarkMode::Answer {
            self.top_k.max(50) // Phase 4: larger pool for LLM reranker
        } else {
            self.top_k
        }
    }
}

// ── Phase 4: LLM query expansion ────────────────────────────────────────────

/// Expand a question into 3 alternative phrasings using an LLM.
/// Bridges the vocabulary gap between query and evidence.
fn expand_query_llm(
    provider: &ConfiguredLlmProvider,
    question: &str,
) -> Result<Vec<String>, crate::llm::LlmError> {
    use crate::llm::{LlmCompletionRequest, LlmMessage};

    let request = LlmCompletionRequest {
        model: provider.metadata().model,
        messages: vec![
            LlmMessage::system(
                "You are a query expander for a memory retrieval system. \
                 Given a question, generate 3 alternative phrasings that use different vocabulary \
                 but ask the same question. Each alternative should be a complete question. \
                 Return ONLY a JSON array of strings. Example: \
                 [\"alternative 1\", \"alternative 2\", \"alternative 3\"]",
            ),
            LlmMessage::user(format!("Question: {question}")),
        ],
        temperature: 0.3,
        max_tokens: 256,
        response_format: Some(serde_json::json!({"type": "json_object"})),
    };
    let response = provider.complete(&request)?;
    parse_query_variants(&response)
}

/// Parse LLM query expansion response into a Vec of variant strings.
fn parse_query_variants(response: &str) -> Result<Vec<String>, crate::llm::LlmError> {
    let trimmed = response.trim();
    let value: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| crate::llm::LlmError::new(format!("failed to parse query variants: {e}")))?;

    let variants = if let Some(arr) = value.get("variants").and_then(|v| v.as_array()) {
        arr
    } else if let Some(arr) = value.as_array() {
        arr
    } else {
        return Err(crate::llm::LlmError::new(
            "query expansion response is not an array",
        ));
    };

    let result: Vec<String> = variants
        .iter()
        .filter_map(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.len() > 3)
        .take(3)
        .collect();

    if result.is_empty() {
        return Err(crate::llm::LlmError::new("no valid query variants found"));
    }
    Ok(result)
}

/// Fuse two sets of search results: OR-combine by memory ID, keeping
/// the highest score per memory, then re-sort and truncate.
fn fuse_candidates(
    existing: Vec<MemoryPacket>,
    new_results: Vec<MemoryPacket>,
    limit: usize,
) -> Vec<MemoryPacket> {
    let mut score_map: std::collections::HashMap<String, MemoryPacket> = existing
        .into_iter()
        .map(|p| (p.memory.id.clone(), p))
        .collect();

    for packet in new_results {
        score_map
            .entry(packet.memory.id.clone())
            .and_modify(|existing| {
                if packet.score > existing.score {
                    *existing = packet.clone();
                }
            })
            .or_insert(packet);
    }

    let mut fused: Vec<MemoryPacket> = score_map.into_values().collect();
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(limit);
    fused
}

fn assemble_answer_evidence<S: MemoryStore>(
    store: &S,
    dataset: &BenchmarkDataset,
    question: &BenchmarkQuestion,
    primary: &[MemoryPacket],
    primary_limit: usize,
    observation_limit: usize,
    include_source: bool,
    include_source_window: bool,
    prefer_source: bool,
) -> StoreResult<Vec<MemoryPacket>> {
    if observation_limit == 0 && !include_source {
        return Ok(select_answer_primary(question, primary, primary_limit));
    }
    let primary_evidence = select_answer_primary(question, primary, primary_limit);
    let mut evidence = primary_evidence.clone();
    let mut seen_memory_ids = evidence
        .iter()
        .map(|packet| packet.memory.id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut seen_sources = evidence
        .iter()
        .filter_map(|packet| packet.memory.source_event_id.clone())
        .collect::<std::collections::BTreeSet<_>>();

    let mut source_packets = Vec::new();
    source_packets.extend(verbatim_turn_side_channel(
        store,
        question,
        &primary_evidence,
        3,
    )?);
    source_packets.extend(source_window_side_channel(
        dataset,
        question,
        &primary_evidence,
        3,
        1,
        include_source_window,
    )?);
    source_packets.extend(verbatim_session_side_channel(
        store,
        question,
        &primary_evidence,
        include_source,
    )?);

    if prefer_source {
        evidence.clear();
        seen_memory_ids.clear();
        seen_sources.clear();
    }

    for packet in source_packets
        .into_iter()
        .chain(if prefer_source {
            primary_evidence
        } else {
            Vec::new()
        })
        .into_iter()
        .chain(profile_side_channel(
            store,
            question,
            observation_limit.min(2),
        )?)
        .into_iter()
        .chain(observation_side_channel(
            store,
            question,
            observation_limit,
        )?)
    {
        let source_seen = packet
            .memory
            .source_event_id
            .as_ref()
            .is_some_and(|source| seen_sources.contains(source));
        let allow_duplicate_source = include_source
            && packet
                .memory
                .metadata
                .get("memory_kind")
                .map(String::as_str)
                .is_some_and(|kind| {
                    matches!(
                        kind,
                        "source_window" | "verbatim_turn" | "observation" | "profile"
                    )
                });
        if seen_memory_ids.insert(packet.memory.id.clone())
            && (!source_seen || allow_duplicate_source)
        {
            if let Some(source) = &packet.memory.source_event_id {
                seen_sources.insert(source.clone());
            }
            evidence.push(packet);
        }
    }

    // Phase 2 ordering fix: deduplicate by content and sort stably so the LLM
    // sees the same evidence in the same order regardless of retrieval ranking.
    // Without this, stemmer ranking changes → different evidence ordering →
    // LLM gives different answers even though the same evidence is present.
    dedup_and_sort_evidence(&mut evidence);

    Ok(evidence)
}

/// Deduplicate evidence packets by content (keep the one with the highest
/// score), then sort stably by source_event_id (None first), and content as
/// final tie-breaker.
fn dedup_and_sort_evidence(evidence: &mut Vec<MemoryPacket>) {
    // Content-based dedup — same text from different memory types
    let mut content_best: std::collections::HashMap<String, (usize, f32)> =
        std::collections::HashMap::new();
    let mut unique: Vec<(usize, MemoryPacket)> = Vec::new();
    for (i, packet) in evidence.iter().enumerate() {
        let key = packet.memory.content.clone();
        if let Some(&(existing_idx, existing_score)) = content_best.get(&key) {
            if packet.score > existing_score {
                unique.retain(|(idx, _)| *idx != existing_idx);
                content_best.insert(key, (i, packet.score));
                unique.push((i, packet.clone()));
            }
            // else skip lower-scored duplicate
        } else {
            content_best.insert(key, (i, packet.score));
            unique.push((i, packet.clone()));
        }
    }
    *evidence = unique.into_iter().map(|(_, p)| p).collect();

    // Sort stably: by source_event_id (None last), then content
    evidence.sort_by(|a, b| {
        a.memory
            .source_event_id
            .cmp(&b.memory.source_event_id)
            .then_with(|| a.memory.content.cmp(&b.memory.content))
    });
}

fn select_answer_primary(
    question: &BenchmarkQuestion,
    candidates: &[MemoryPacket],
    limit: usize,
) -> Vec<MemoryPacket> {
    let mut selected = candidates.iter().take(limit).cloned().collect::<Vec<_>>();
    let mut seen_sources = selected
        .iter()
        .filter_map(|packet| packet.memory.source_event_id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut promotions = candidates
        .iter()
        .skip(limit)
        .filter_map(|packet| {
            let bonus = structured_answer_bonus(&question.text, packet);
            (bonus >= 0.20).then_some((packet, bonus))
        })
        .collect::<Vec<_>>();
    promotions.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let promotion_limit = limit.min(2);
    for (packet, _) in promotions.into_iter().take(promotion_limit) {
        let source_is_new = packet
            .memory
            .source_event_id
            .as_ref()
            .is_none_or(|source| seen_sources.insert(source.clone()));
        if source_is_new && !selected.is_empty() {
            selected.pop();
            let mut packet = packet.clone();
            packet
                .reasons
                .push("structured_answer_promotion".to_string());
            selected.push(packet);
        }
    }
    selected
}

fn structured_answer_bonus(question: &str, packet: &MemoryPacket) -> f32 {
    let lower = question.to_lowercase();
    let content = packet.memory.content.to_lowercase();
    if contains_any(&lower, &["where", "location", "live", "from"])
        && contains_any(&content, &[" in ", " at ", " from ", " lives ", " moved "])
        && token_overlap_score(question, &packet.memory.content) > 0.0
    {
        return 0.20;
    }
    let kind = packet
        .memory
        .metadata
        .get("memory_kind")
        .map(String::as_str)
        .unwrap_or("");
    if !matches!(kind, "llm_fact" | "observation" | "profile") {
        return 0.0;
    }
    let subject_bonus = packet
        .memory
        .metadata
        .get("subject")
        .is_some_and(|subject| lower.contains(&subject.to_lowercase()))
        .then_some(0.12)
        .unwrap_or(0.0);
    let relation_bonus = packet
        .memory
        .metadata
        .get("relation")
        .is_some_and(|relation| relation_matches_question(&lower, relation))
        .then_some(0.12)
        .unwrap_or(0.0);
    let overlap_bonus = if token_overlap_score(question, &packet.memory.content) > 0.0 {
        0.08
    } else {
        0.0
    };
    subject_bonus + relation_bonus + overlap_bonus
}

fn relation_matches_question(question: &str, relation: &str) -> bool {
    let relation = relation.to_lowercase();
    question.contains(&relation)
        || (contains_any(question, &["job", "work", "career"])
            && contains_any(&relation, &["job", "work", "career", "occupation"]))
        || (contains_any(question, &["where", "location", "live", "moved"])
            && contains_any(&relation, &["location", "live", "moved"]))
        || (contains_any(question, &["prefer", "favorite", "like"])
            && contains_any(&relation, &["prefer", "favorite", "like"]))
        || (contains_any(question, &["when", "date", "year"])
            && contains_any(&relation, &["time", "date", "happened"]))
}

fn verbatim_session_side_channel<S: MemoryStore>(
    store: &S,
    question: &BenchmarkQuestion,
    primary: &[MemoryPacket],
    include_source: bool,
) -> StoreResult<Vec<MemoryPacket>> {
    if !include_source {
        return Ok(Vec::new());
    }
    let source_ids = primary
        .iter()
        .filter_map(|packet| packet.memory.source_event_id.as_deref())
        .collect::<std::collections::BTreeSet<_>>();
    if source_ids.is_empty() {
        return Ok(Vec::new());
    }
    let query = MemoryQuery::new("")
        .namespace(question.conversation_id.clone())
        .memory_types(vec![MemoryType::Episodic])
        .limit(usize::MAX);
    let mut packets = store
        .list_memories(&query)?
        .into_iter()
        .filter(|memory| {
            memory.metadata.get("memory_kind").map(String::as_str) == Some("verbatim_session")
        })
        .filter(|memory| {
            memory
                .metadata
                .get("source_turn_ids")
                .is_some_and(|ids| ids.split('\n').any(|id| source_ids.contains(id)))
        })
        .map(|memory| MemoryPacket {
            memory,
            score: 0.22,
            reasons: vec!["verbatim_session_side_channel".to_string()],
        })
        .collect::<Vec<_>>();
    packets.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    packets.truncate(2);
    Ok(packets)
}

fn verbatim_turn_side_channel<S: MemoryStore>(
    store: &S,
    question: &BenchmarkQuestion,
    primary: &[MemoryPacket],
    source_limit: usize,
) -> StoreResult<Vec<MemoryPacket>> {
    let source_ids: Vec<&str> = primary
        .iter()
        .filter_map(|p| p.memory.source_event_id.as_deref())
        .take(source_limit)
        .collect();
    if source_ids.is_empty() {
        return Ok(Vec::new());
    }

    let query = MemoryQuery::new("")
        .namespace(question.conversation_id.clone())
        .memory_types(vec![MemoryType::Episodic])
        .limit(usize::MAX);

    let mut packets: Vec<MemoryPacket> = store
        .list_memories(&query)?
        .into_iter()
        .filter(|m| m.metadata.get("memory_kind").map(String::as_str) == Some("verbatim_turn"))
        .filter(|m| {
            m.source_event_id
                .as_deref()
                .is_some_and(|id| source_ids.contains(&id))
        })
        .map(|m| MemoryPacket {
            memory: m,
            score: 0.45,
            reasons: vec!["verbatim_turn_evidence".to_string()],
        })
        .collect();

    // Prefer speaker/time-prefixed verbatim evidence over the plain text duplicate.
    packets.sort_by(|a, b| {
        let a_plain =
            a.memory.metadata.get("verbatim_form").map(String::as_str) == Some("turn_text");
        let b_plain =
            b.memory.metadata.get("verbatim_form").map(String::as_str) == Some("turn_text");
        a_plain.cmp(&b_plain).then_with(|| {
            b.memory
                .importance
                .partial_cmp(&a.memory.importance)
                .unwrap_or(Ordering::Equal)
        })
    });
    packets.truncate(source_limit);
    Ok(packets)
}

fn source_window_side_channel(
    dataset: &BenchmarkDataset,
    question: &BenchmarkQuestion,
    primary: &[MemoryPacket],
    source_limit: usize,
    radius: usize,
    include_source_window: bool,
) -> StoreResult<Vec<MemoryPacket>> {
    if !include_source_window {
        return Ok(Vec::new());
    }
    let Some(conversation) = dataset
        .conversations
        .iter()
        .find(|conversation| conversation.id == question.conversation_id)
    else {
        return Ok(Vec::new());
    };
    let source_ids = primary
        .iter()
        .filter_map(|packet| packet.memory.source_event_id.as_deref())
        .take(source_limit)
        .collect::<Vec<_>>();
    let mut packets = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for source_id in source_ids {
        let Some(index) = conversation
            .turns
            .iter()
            .position(|turn| turn.id == source_id)
        else {
            continue;
        };
        let start = index.saturating_sub(radius);
        let end = (index + radius + 1).min(conversation.turns.len());
        for turn in &conversation.turns[start..end] {
            if !seen.insert(turn.id.clone()) {
                continue;
            }
            packets.push(turn_to_window_packet(
                turn,
                &question.conversation_id,
                window_distance(index, conversation, &turn.id),
            ));
        }
    }
    Ok(packets)
}

fn window_distance(
    source_index: usize,
    conversation: &super::dataset::Conversation,
    turn_id: &str,
) -> usize {
    conversation
        .turns
        .iter()
        .position(|turn| turn.id == turn_id)
        .map(|index| source_index.abs_diff(index))
        .unwrap_or(usize::MAX)
}

fn turn_to_window_packet(turn: &BenchmarkTurn, namespace: &str, distance: usize) -> MemoryPacket {
    let content = if let Some(timestamp) = &turn.timestamp {
        format!(
            "[source_window distance={distance} time={timestamp}] {}: {}",
            turn.speaker, turn.text
        )
    } else {
        format!(
            "[source_window distance={distance}] {}: {}",
            turn.speaker, turn.text
        )
    };
    let mut memory = Memory::new(content, MemoryType::Episodic)
        .namespace(namespace.to_string())
        .source_event(turn.id.clone())
        .importance(0.52)
        .confidence(0.68);
    memory
        .metadata
        .insert("memory_kind".to_string(), "source_window".to_string());
    memory
        .metadata
        .insert("speaker".to_string(), turn.speaker.clone());
    if let Some(timestamp) = &turn.timestamp {
        memory
            .metadata
            .insert("event_time".to_string(), timestamp.clone());
    }
    MemoryPacket {
        memory,
        score: 0.30_f32 / (distance.max(1) as f32),
        reasons: vec!["source_window_side_channel".to_string()],
    }
}

fn profile_side_channel<S: MemoryStore>(
    store: &S,
    question: &BenchmarkQuestion,
    limit: usize,
) -> StoreResult<Vec<MemoryPacket>> {
    if !profile_relevant_question(&question.text) {
        return Ok(Vec::new());
    }
    let mut packets = observation_side_channel(store, question, limit * 3)?;
    packets.retain(|packet| {
        matches!(
            packet
                .memory
                .metadata
                .get("observation_kind")
                .map(String::as_str),
            Some("preference" | "attribute" | "update" | "temporal")
        )
    });

    let mut profile_packets = Vec::new();
    for packet in packets.into_iter().take(limit) {
        let subject = packet
            .memory
            .metadata
            .get("subject")
            .cloned()
            .unwrap_or_else(|| "user".to_string());
        let relation = packet
            .memory
            .metadata
            .get("relation")
            .cloned()
            .unwrap_or_else(|| "attribute".to_string());
        let object = packet
            .memory
            .metadata
            .get("object")
            .cloned()
            .unwrap_or_else(|| packet.memory.content.clone());
        let mut memory = Memory::new(
            format!("[profile] {subject} {relation}: {object}"),
            MemoryType::Semantic,
        )
        .namespace(question.conversation_id.clone())
        .importance(0.80)
        .confidence(0.70);
        memory.source_event_id = packet.memory.source_event_id.clone();
        memory
            .metadata
            .insert("memory_kind".to_string(), "profile".to_string());
        memory
            .metadata
            .insert("profile_relation".to_string(), relation);
        profile_packets.push(MemoryPacket {
            memory,
            score: packet.score + 0.10,
            reasons: vec!["profile_side_channel".to_string()],
        });
    }
    Ok(profile_packets)
}

fn profile_relevant_question(question: &str) -> bool {
    let lower = question.to_lowercase();
    contains_any(
        &lower,
        &[
            "prefer",
            "preference",
            "favorite",
            "like",
            "want",
            "current",
            "latest",
            "previous",
            "what is",
            "what was",
            "where",
            "who",
            "job",
            "work",
            "degree",
            "relationship",
        ],
    )
}

fn observation_side_channel<S: MemoryStore>(
    store: &S,
    question: &BenchmarkQuestion,
    limit: usize,
) -> StoreResult<Vec<MemoryPacket>> {
    let query = MemoryQuery::new("")
        .namespace(question.conversation_id.clone())
        .memory_types(vec![MemoryType::Semantic])
        .limit(usize::MAX);
    let mut packets = store
        .list_memories(&query)?
        .into_iter()
        .filter(|memory| {
            memory.metadata.get("memory_kind").map(String::as_str) == Some("observation")
        })
        .map(|memory| {
            let score = observation_side_score(&question.text, &memory);
            MemoryPacket {
                memory,
                score,
                reasons: vec!["observation_side_channel".to_string()],
            }
        })
        .filter(|packet| packet.score > 0.0)
        .collect::<Vec<_>>();
    packets.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    packets.truncate(limit);
    Ok(packets)
}

fn observation_side_score(question: &str, memory: &Memory) -> f32 {
    let mut score = token_overlap_score(question, &memory.content);
    let lower = question.to_lowercase();
    if let Some(kind) = memory.metadata.get("observation_kind") {
        let intent_match = match kind.as_str() {
            "preference" => contains_any(
                &lower,
                &["prefer", "preference", "favorite", "like", "want"],
            ),
            "update" => contains_any(&lower, &["current", "latest", "new", "now", "previous"]),
            "temporal" => contains_any(
                &lower,
                &[
                    "when",
                    "what year",
                    "what date",
                    "how long",
                    "before",
                    "after",
                ],
            ),
            "attribute" => contains_any(
                &lower,
                &[
                    "who", "where", "what is", "what was", "job", "degree", "name",
                ],
            ),
            _ => false,
        };
        if intent_match {
            score += 0.35;
        }
    }
    if let Some(entities) = memory.metadata.get("entities") {
        for entity in entities.split('\n') {
            if entity.len() > 2 && lower.contains(entity) {
                score += 0.20;
            }
        }
    }
    score
}

pub fn dataset_hash(bytes: &[u8]) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn write_outputs(
    dataset: &BenchmarkDataset,
    config: &BenchmarkRunConfig,
    report: &BenchmarkRunReport,
) -> StoreResult<()> {
    let manifest = BTreeMap::from([
        ("benchmark".to_string(), config.benchmark.clone()),
        ("mode".to_string(), config.mode.as_str().to_string()),
        ("dataset_name".to_string(), dataset.name.clone()),
        ("dataset_version".to_string(), dataset.version.clone()),
        ("dataset_hash".to_string(), config.dataset_hash.clone()),
        ("store".to_string(), config.store.clone()),
        ("answerer".to_string(), config.answerer.clone()),
        ("extractor".to_string(), config.extractor.clone()),
        ("judge".to_string(), config.judge.clone()),
        ("evidence_pack".to_string(), config.evidence_pack.clone()),
        ("top_k".to_string(), config.top_k.to_string()),
        (
            "answer_evidence_top_k".to_string(),
            config.answer_evidence_primary_limit().to_string(),
        ),
        (
            "search_candidate_top_k".to_string(),
            config.search_candidate_limit().to_string(),
        ),
    ]);
    let mut manifest = manifest;
    manifest.insert(
        "embedding_provider".to_string(),
        if cfg!(feature = "embed-ollama") {
            "ollama".to_string()
        } else {
            "hash".to_string()
        },
    );
    if cfg!(feature = "embed-ollama") {
        manifest.insert(
            "embedding_model".to_string(),
            std::env::var("AGENT_MEMORY_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "all-minilm".to_string()),
        );
    }
    if let Some(max_questions) = config.max_questions {
        manifest.insert("max_questions".to_string(), max_questions.to_string());
    }
    if config.question_offset > 0 {
        manifest.insert(
            "question_offset".to_string(),
            config.question_offset.to_string(),
        );
    }
    if let Some(provider) = &config.llm_provider {
        manifest.insert("llm_provider".to_string(), provider.provider.clone());
        manifest.insert("llm_model".to_string(), provider.model.clone());
        if let Some(base_url) = &provider.base_url {
            manifest.insert("llm_base_url".to_string(), base_url.clone());
        }
        if let Some(prompt_version) = &provider.prompt_version {
            manifest.insert("llm_prompt_version".to_string(), prompt_version.clone());
        }
    }

    write_json(config.output_dir.join("manifest.json"), &manifest)?;
    write_json(config.output_dir.join("summary.json"), &report.summary)?;
    write_jsonl(config.output_dir.join("scores.jsonl"), &report.results)?;
    write_report_md(config.output_dir.join("report.md"), &report.summary)?;
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> StoreResult<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, value).map_err(std::io::Error::other)?;
    Ok(())
}

fn write_jsonl(path: impl AsRef<Path>, values: &[impl Serialize]) -> StoreResult<()> {
    let mut file = File::create(path)?;
    for value in values {
        let line = serde_json::to_string(value).map_err(std::io::Error::other)?;
        writeln!(file, "{line}")?;
    }
    Ok(())
}

fn write_report_md(path: impl AsRef<Path>, summary: &BenchmarkSummary) -> StoreResult<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# Benchmark Report")?;
    writeln!(file)?;
    writeln!(file, "- benchmark: {}", summary.benchmark)?;
    writeln!(file, "- mode: {}", summary.mode)?;
    writeln!(file, "- questions: {}", summary.question_count)?;
    writeln!(file, "- accuracy: {:?}", summary.accuracy)?;
    writeln!(file, "- recall@1: {:.4}", summary.recall_at_1)?;
    writeln!(file, "- recall@3: {:.4}", summary.recall_at_3)?;
    writeln!(file, "- recall@5: {:.4}", summary.recall_at_5)?;
    writeln!(file, "- recall@10: {:.4}", summary.recall_at_10)?;
    writeln!(file, "- recall@20: {:.4}", summary.recall_at_20)?;
    writeln!(file, "- recall@50: {:.4}", summary.recall_at_50)?;
    writeln!(file, "- recall@100: {:.4}", summary.recall_at_100)?;
    writeln!(file, "- recall@200: {:.4}", summary.recall_at_200)?;
    writeln!(file, "- mrr: {:.4}", summary.mrr)?;
    writeln!(file, "- ndcg@5: {:.4}", summary.ndcg_at_5)?;
    writeln!(
        file,
        "- retrieval miss@10 rate: {:.4}",
        summary.retrieval_miss_at_10_rate
    )?;
    writeln!(
        file,
        "- hit@10 answer-wrong rate: {:.4}",
        summary.hit_at_10_answer_wrong_rate
    )?;
    writeln!(
        file,
        "- hit@1 answer-wrong rate: {:.4}",
        summary.hit_at_1_answer_wrong_rate
    )?;
    Ok(())
}

pub fn ensure_no_gold_leak(question: &BenchmarkQuestion, answer_input: &AnswerInput) -> bool {
    let serialized = serde_json::to_string(answer_input).unwrap_or_default();
    answer_input.question.id == question.id
        && answer_input.question.conversation_id == question.conversation_id
        && answer_input.question.text == question.text
        && !serialized.contains("gold_answers")
        && !serialized.contains("evidence_turn_ids")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_evidence_promotes_relevant_structured_fact_from_expanded_pool() {
        let question = BenchmarkQuestion {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Where does the user work now?".to_string(),
            gold_answers: Vec::new(),
            evidence_turn_ids: Vec::new(),
            category: None,
        };
        let mut candidates = (0..10)
            .map(|index| packet(&format!("noise-{index}"), "unrelated memory"))
            .collect::<Vec<_>>();
        let mut fact = packet("fact", "user work: NewCo");
        fact.memory
            .metadata
            .insert("memory_kind".to_string(), "llm_fact".to_string());
        fact.memory
            .metadata
            .insert("subject".to_string(), "user".to_string());
        fact.memory
            .metadata
            .insert("relation".to_string(), "work".to_string());
        candidates.push(fact);

        let selected = select_answer_primary(&question, &candidates, 10);

        assert_eq!(selected.len(), 10);
        assert!(selected.iter().any(|packet| {
            packet.memory.content == "user work: NewCo"
                && packet
                    .reasons
                    .contains(&"structured_answer_promotion".to_string())
        }));
    }

    #[test]
    fn answer_evidence_promotes_explicit_camping_location_from_expanded_pool() {
        let question = BenchmarkQuestion {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Where has Melanie camped?".to_string(),
            gold_answers: Vec::new(),
            evidence_turn_ids: Vec::new(),
            category: None,
        };
        let mut candidates = (0..10)
            .map(|index| packet(&format!("noise-{index}"), "unrelated memory"))
            .collect::<Vec<_>>();
        candidates.push(packet(
            "camping-location",
            "Melanie took her family camping in the mountains last week",
        ));

        let selected = select_answer_primary(&question, &candidates, 10);

        assert!(selected.iter().any(|packet| {
            packet.memory.content.contains("camping in the mountains")
                && packet
                    .reasons
                    .contains(&"structured_answer_promotion".to_string())
        }));
    }

    fn packet(source: &str, content: &str) -> MemoryPacket {
        MemoryPacket {
            memory: Memory::new(content, MemoryType::Semantic).source_event(source),
            score: 0.1,
            reasons: Vec::new(),
        }
    }
}
