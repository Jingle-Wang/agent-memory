use std::collections::BTreeSet;

use crate::embedding::token_overlap_score;
use crate::models::{Event, Memory, MemoryPacket, MemoryQuery, MemoryType};
use crate::retriever::HybridMemoryRetriever;
use crate::store::{MemoryStore, StoreResult};

pub struct MemoryEngine<S: MemoryStore> {
    store: S,
    retriever: HybridMemoryRetriever,
}

impl<S: MemoryStore> MemoryEngine<S> {
    pub fn new(store: S) -> Self {
        Self {
            store,
            retriever: HybridMemoryRetriever::new(),
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
        MemoryType::Episodic => contains_any(
            lower,
            &[
                "decided",
                "finished",
                "failed",
                "fixed",
                "blocked",
                "created",
                "implemented",
            ],
        ),
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
