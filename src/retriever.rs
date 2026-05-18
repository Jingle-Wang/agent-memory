use std::cmp::Ordering;
use std::time::SystemTime;

use crate::embedding::{HashEmbedding, cosine_similarity, token_overlap_score};
use crate::models::{MemoryPacket, MemoryQuery, MemoryType, epoch_seconds};
use crate::store::{MemoryStore, StoreResult};

#[derive(Clone, Debug)]
pub struct HybridMemoryRetriever {
    embedder: HashEmbedding,
}

impl Default for HybridMemoryRetriever {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridMemoryRetriever {
    pub fn new() -> Self {
        Self {
            embedder: HashEmbedding::new(128),
        }
    }

    pub fn search<S: MemoryStore>(
        &self,
        store: &S,
        query: &MemoryQuery,
    ) -> StoreResult<Vec<MemoryPacket>> {
        let query_embedding = self.embedder.embed(&query.text);
        let now = SystemTime::now();
        let mut packets = Vec::new();

        for memory in store.list_memories(query)? {
            let memory_embedding = self.embedder.embed(&memory.content);
            let vector = cosine_similarity(&query_embedding, &memory_embedding).max(0.0);
            let lexical = token_overlap_score(&query.text, &memory.content);
            let recency = recency_score(memory.updated_at, now);
            let type_weight = type_weight(&memory.memory_type);

            let score = 0.42 * vector
                + 0.24 * lexical
                + 0.14 * memory.importance
                + 0.10 * memory.confidence
                + 0.06 * recency
                + 0.04 * type_weight;

            packets.push(MemoryPacket {
                memory,
                score,
                reasons: score_reasons(vector, lexical, recency),
            });
        }

        packets.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        packets.truncate(query.limit);
        Ok(packets)
    }
}

fn type_weight(memory_type: &MemoryType) -> f32 {
    match memory_type {
        MemoryType::Working => 0.75,
        MemoryType::Episodic => 0.65,
        MemoryType::Semantic => 1.0,
        MemoryType::Procedural => 0.9,
        MemoryType::Reflection => 0.85,
    }
}

fn recency_score(updated_at: SystemTime, now: SystemTime) -> f32 {
    let updated = epoch_seconds(updated_at);
    let current = epoch_seconds(now);
    let age_days = current.saturating_sub(updated) as f32 / 86_400.0;
    1.0 / (1.0 + age_days / 30.0)
}

fn score_reasons(vector: f32, lexical: f32, recency: f32) -> Vec<String> {
    let mut reasons = Vec::new();
    if vector > 0.2 {
        reasons.push("vector_similarity".to_string());
    }
    if lexical > 0.0 {
        reasons.push("keyword_overlap".to_string());
    }
    if recency > 0.8 {
        reasons.push("recent".to_string());
    }
    if reasons.is_empty() {
        reasons.push("prioritized_by_importance".to_string());
    }
    reasons
}
