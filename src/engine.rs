use std::collections::BTreeSet;

use crate::embedding::{EmbeddingProvider, token_overlap_score};
use crate::extractor::MemoryExtractor;
use crate::ingestion_buffer::IngestionBuffer;
use crate::models::{Event, Memory, MemoryPacket, MemoryQuery, MemoryType};
use crate::retriever::HybridMemoryRetriever;
use crate::store::{MemoryStore, StoreResult};

/// Default window size for the ingestion buffer.
/// 5 turns is the sweet spot for LoCoMo-style conversations (10-15 turns/session):
/// enough context for the extractor to catch cross-turn entity relations,
/// but not so large that it overwhelms the LLM prompt.
pub const DEFAULT_INGESTION_WINDOW: usize = 5;

pub struct MemoryEngine<S: MemoryStore> {
    store: S,
    retriever: HybridMemoryRetriever,
    buffer: Option<IngestionBuffer>,
}

impl<S: MemoryStore> MemoryEngine<S> {
    pub fn new(store: S) -> Self {
        Self {
            store,
            retriever: HybridMemoryRetriever::new(),
            buffer: None,
        }
    }

    pub fn new_with_embedding(store: S, embedder: Box<dyn EmbeddingProvider>) -> Self {
        Self {
            store,
            retriever: HybridMemoryRetriever::with_embedder(embedder),
            buffer: None,
        }
    }

    /// Create an engine with the sliding-window ingestion buffer enabled.
    ///
    /// When a buffer is present, [`ingest_buffered`] and
    /// [`ingest_buffered_with_extractor`] will accumulate events until
    /// `window_size` turns are collected, then flush them as a combined
    /// multi-turn context to the extractor for richer fact extraction.
    pub fn new_with_buffer(store: S, window_size: usize) -> Self {
        Self {
            store,
            retriever: HybridMemoryRetriever::new(),
            buffer: Some(IngestionBuffer::new(window_size)),
        }
    }

    /// Create an engine with both an external embedder and the ingestion buffer.
    pub fn new_with_buffer_and_embedding(
        store: S,
        embedder: Box<dyn EmbeddingProvider>,
        window_size: usize,
    ) -> Self {
        Self {
            store,
            retriever: HybridMemoryRetriever::with_embedder(embedder),
            buffer: Some(IngestionBuffer::new(window_size)),
        }
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    pub fn add_event(&mut self, event: Event) -> StoreResult<Event> {
        self.store.add_event(event)
    }

    pub fn remember(&mut self, memory: Memory) -> StoreResult<Memory> {
        if let Some(existing) = self.find_duplicate(&memory)? {
            let mut merged = existing;
            merged.content = merge_content(&merged.content, &memory.content);
            merged.importance = merged.importance.max(memory.importance);
            merged.confidence = merged.confidence.max(memory.confidence);
            for (key, value) in memory.metadata {
                merged.metadata.insert(key, value);
            }
            self.store.update_memory(merged)
        } else {
            self.store.add_memory(memory)
        }
    }

    pub fn ingest_event(&mut self, event: Event) -> StoreResult<Vec<Memory>> {
        let event = self.add_event(event)?;
        let candidates = extract_memories(&event);
        let mut committed = Vec::new();
        for memory in candidates {
            committed.push(self.remember(memory)?);
        }
        Ok(committed)
    }

    /// Ingest an event using an external `MemoryExtractor` instead of the
    /// built-in rule-based extractor. Useful for LLM-backed extraction.
    pub fn ingest_event_with_extractor(
        &mut self,
        event: Event,
        extractor: &dyn MemoryExtractor,
    ) -> StoreResult<Vec<Memory>> {
        let event = self.add_event(event)?;
        let candidates = extractor
            .extract(&event, None)
            .map_err(|msg| crate::store::StoreError::Corrupt(msg))?;
        let mut committed = Vec::new();
        for memory in candidates {
            committed.push(self.remember(memory)?);
        }
        Ok(committed)
    }

    // ── Buffered (sliding-window) ingestion methods ──────────────────────
    //
    // These methods accumulate conversation turns in an internal
    // `IngestionBuffer`. When the buffer reaches its window size (default
    // 5 turns), all buffered events are combined into a single multi-turn
    // context and sent to the extractor in one call. This gives the LLM
    // enough surrounding dialog to extract rich facts — entity relationships,
    // implied preferences, emotional arcs — instead of thin single-turn
    // snippets.
    //
    // For conversations shorter than the window size, call `flush_buffer()`
    // at the end to extract from the remaining turns.

    /// Ingest an event with buffered rule-based extraction.
    ///
    /// Events accumulate in the internal buffer. When the window is full,
    /// the combined multi-turn context is passed to the built-in rule-based
    /// extractor. Returns an empty vec for intermediate turns (buffer not
    /// yet full).
    ///
    /// Requires the engine to have been created with
    /// [`new_with_buffer`](Self::new_with_buffer).
    pub fn ingest_buffered(&mut self, event: Event) -> StoreResult<Vec<Memory>> {
        let event = self.add_event(event)?;
        let Some(buffer) = self.buffer.as_mut() else {
            // No buffer configured — fall back to single-event extraction
            let candidates = extract_memories(&event);
            let mut committed = Vec::new();
            for memory in candidates {
                committed.push(self.remember(memory)?);
            }
            return Ok(committed);
        };
        if let Some(combined) = buffer.push(event) {
            let candidates = extract_memories(&combined);
            let mut committed = Vec::new();
            for memory in candidates {
                committed.push(self.remember(memory)?);
            }
            Ok(committed)
        } else {
            Ok(vec![])
        }
    }

    /// Ingest an event with buffered LLM extraction.
    ///
    /// Like [`ingest_buffered`](Self::ingest_buffered) but uses the
    /// provided external extractor (typically `LlmMemoryExtractor`) on
    /// the combined multi-turn context.
    ///
    /// Requires the engine to have been created with
    /// [`new_with_buffer`](Self::new_with_buffer).
    pub fn ingest_buffered_with_extractor(
        &mut self,
        event: Event,
        extractor: &dyn MemoryExtractor,
    ) -> StoreResult<Vec<Memory>> {
        let event = self.add_event(event)?;
        let Some(buffer) = self.buffer.as_mut() else {
            // No buffer configured — fall back to single-event extraction
            let candidates = extractor
                .extract(&event, None)
                .map_err(|msg| crate::store::StoreError::Corrupt(msg))?;
            let mut committed = Vec::new();
            for memory in candidates {
                committed.push(self.remember(memory)?);
            }
            return Ok(committed);
        };
        if let Some(combined) = buffer.push(event) {
            let candidates = extractor
                .extract(&combined, None)
                .map_err(|msg| crate::store::StoreError::Corrupt(msg))?;
            let mut committed = Vec::new();
            for memory in candidates {
                committed.push(self.remember(memory)?);
            }
            Ok(committed)
        } else {
            Ok(vec![])
        }
    }

    /// Flush any remaining events in the ingestion buffer.
    ///
    /// Call this at the end of a conversation to ensure the last
    /// (possibly partial) window is not lost. Uses the rule-based extractor.
    pub fn flush_buffer(&mut self) -> StoreResult<Vec<Memory>> {
        let Some(buffer) = self.buffer.as_mut() else {
            return Ok(vec![]);
        };
        let Some(combined) = buffer.flush() else {
            return Ok(vec![]);
        };
        let candidates = extract_memories(&combined);
        let mut committed = Vec::new();
        for memory in candidates {
            committed.push(self.remember(memory)?);
        }
        Ok(committed)
    }

    /// Flush any remaining events in the ingestion buffer using an external extractor.
    ///
    /// Call this at the end of a conversation. Uses the provided extractor
    /// (typically `LlmMemoryExtractor`).
    pub fn flush_buffer_with_extractor(
        &mut self,
        extractor: &dyn MemoryExtractor,
    ) -> StoreResult<Vec<Memory>> {
        let Some(buffer) = self.buffer.as_mut() else {
            return Ok(vec![]);
        };
        let Some(combined) = buffer.flush() else {
            return Ok(vec![]);
        };
        let candidates = extractor
            .extract(&combined, None)
            .map_err(|msg| crate::store::StoreError::Corrupt(msg))?;
        let mut committed = Vec::new();
        for memory in candidates {
            committed.push(self.remember(memory)?);
        }
        Ok(committed)
    }

    /// Check whether the engine has a buffer configured.
    pub fn has_buffer(&self) -> bool {
        self.buffer.is_some()
    }

    pub fn search(&self, query: MemoryQuery) -> StoreResult<Vec<MemoryPacket>> {
        self.retriever.search(&self.store, &query)
    }

    pub fn build_context(&self, query: MemoryQuery) -> StoreResult<String> {
        let packets = self.search(query)?;
        if packets.is_empty() {
            return Ok(String::new());
        }

        let mut output = String::from("# Agent Memory\n");
        for packet in packets {
            output.push_str(&format!(
                "- [{} score={:.3} confidence={:.2}] {}\n",
                packet.memory.memory_type,
                packet.score,
                packet.memory.confidence,
                packet.memory.content
            ));
        }
        Ok(output)
    }

    pub fn delete_memory(&mut self, memory_id: &str) -> StoreResult<()> {
        self.store.delete_memory(memory_id)
    }

    fn find_duplicate(&self, candidate: &Memory) -> StoreResult<Option<Memory>> {
        let query = MemoryQuery::new(candidate.content.clone())
            .namespace(candidate.namespace.clone())
            .memory_types(vec![candidate.memory_type.clone()])
            .limit(16);
        let existing = self.store.list_memories(&query)?;
        Ok(existing.into_iter().find(|memory| {
            memory.content.eq_ignore_ascii_case(&candidate.content)
                || token_overlap_score(&memory.content, &candidate.content) >= 0.72
        }))
    }
}

pub fn extract_memories(event: &Event) -> Vec<Memory> {
    let mut memories = Vec::new();
    let sentences = split_sentences(&event.text);

    for sentence in sentences {
        let lower = sentence.to_lowercase();
        let (memory_type, importance, confidence) = classify_sentence(&lower);
        if should_keep(&lower, &memory_type) {
            memories.push(
                Memory::new(sentence.trim(), memory_type)
                    .namespace(event.namespace.clone())
                    .source_event(event.id.clone())
                    .importance(importance)
                    .confidence(confidence),
            );
        }
    }

    if memories.is_empty() && event.text.split_whitespace().count() >= 8 {
        memories.push(
            Memory::new(summarize_episode(event), MemoryType::Episodic)
                .namespace(event.namespace.clone())
                .source_event(event.id.clone())
                .importance(0.35)
                .confidence(0.55),
        );
    }

    dedupe_candidates(memories)
}

fn classify_sentence(lower: &str) -> (MemoryType, f32, f32) {
    if contains_any(
        lower,
        &["i prefer", "i like", "i dislike", "my name is", "call me"],
    ) {
        (MemoryType::Semantic, 0.85, 0.82)
    } else if contains_any(lower, &["remember that", "important:", "do not", "always"]) {
        (MemoryType::Semantic, 0.9, 0.78)
    } else if contains_any(
        lower,
        &["workflow", "when you", "steps:", "procedure", "use the"],
    ) {
        (MemoryType::Procedural, 0.75, 0.72)
    } else if contains_any(lower, &["we learned", "next time", "mistake", "root cause"]) {
        (MemoryType::Reflection, 0.7, 0.68)
    } else {
        (MemoryType::Episodic, 0.45, 0.58)
    }
}

fn should_keep(lower: &str, memory_type: &MemoryType) -> bool {
    if lower.len() < 12 {
        return false;
    }
    match memory_type {
        MemoryType::Semantic | MemoryType::Procedural | MemoryType::Reflection => true,
        MemoryType::Episodic => {
            // Relaxed filter: keep all non-chitchat sentences.
            // Filter out only pure social formulas (≤3 words, greeting-like).
            let word_count = lower.split_whitespace().count();
            if word_count <= 3 {
                let chitchat = [
                    "hi",
                    "hey",
                    "hello",
                    "ok",
                    "okay",
                    "thanks",
                    "thank you",
                    "bye",
                    "goodbye",
                    "yes",
                    "no",
                    "sure",
                    "cool",
                    "nice",
                    "great",
                    "fine",
                    "alright",
                    "hmm",
                    "lol",
                ];
                if chitchat.iter().any(|&c| lower.trim() == c) {
                    return false;
                }
            }
            true
        }
        MemoryType::Working => false,
    }
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn summarize_episode(event: &Event) -> String {
    let mut text = event.text.trim().replace('\n', " ");
    if text.len() > 240 {
        text.truncate(240);
        text.push_str("...");
    }
    format!("{} said: {text}", event.actor)
}

fn dedupe_candidates(memories: Vec<Memory>) -> Vec<Memory> {
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for memory in memories {
        let key = memory.content.to_lowercase();
        if seen.insert(key) {
            result.push(memory);
        }
    }
    result
}

fn merge_content(left: &str, right: &str) -> String {
    if left.contains(right) {
        left.to_string()
    } else if right.contains(left) {
        right.to_string()
    } else {
        format!("{left}; {right}")
    }
}
