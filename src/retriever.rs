use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::embedding::{
    EmbeddingProvider, HashEmbedding, cosine_similarity, token_overlap_score, tokenize,
};
use crate::entity::{compute_entity_boost, extract_query_entities};
use crate::models::{Memory, MemoryPacket, MemoryQuery};
use crate::store::{MemoryStore, StoreResult};

// ── BM25 constants ──────────────────────────────────────────────────────────
const BM25_K1: f32 = 1.5;
const BM25_B: f32 = 0.75;
const MAX_LEN_NORM: f32 = 200.0; // cap memory word count for BM25 normalization

// ── Scoring weights ─────────────────────────────────────────────────────────
// HashEmbedding (128d pseudo-random) — lean into sparse/lexical signals.
struct ScoringWeights {
    term_density: f32,  // Jaccard: fraction of query tokens found in memory
    term_freq: f32,     // BM25-saturated TF, normalized by memory length
    lexical: f32,       // full token overlap Jaccard
    entity_bonus: f32,  // metadata entity match
    speaker_match: f32, // query mentions name matching memory speaker
    vector: f32,        // cosine similarity (near-noise for HashEmbedding)
    importance: f32,    // fixed ingestion importance value
    type_bonus: f32,    // verbatim_turn boost factor
}

const WEIGHTS_HASH: ScoringWeights = ScoringWeights {
    term_density: 0.32, // reduced 0.45→0.32 to offset prefix_bonus increase
    term_freq: 0.13,    // reduced 0.25→0.13 to offset prefix_bonus increase
    lexical: 0.00,      // was Jaccard duplicate of term_density — zeroed out
    entity_bonus: 0.15,
    speaker_match: 0.05,
    vector: 0.02,
    importance: 0.03,
    type_bonus: 0.05,
};

const WEIGHTS_OLLAMA: ScoringWeights = ScoringWeights {
    term_density: 0.22,  // proportionally scaled 0.18→0.22 (sum→1.0)
    term_freq: 0.09,     // proportionally scaled 0.07→0.09
    lexical: 0.00,       // was Jaccard duplicate of term_density — zeroed out
    entity_bonus: 0.22,  // proportionally scaled 0.18→0.22
    speaker_match: 0.06, // proportionally scaled 0.05→0.06
    vector: 0.25,        // proportionally scaled 0.20→0.25
    importance: 0.04,    // proportionally scaled 0.03→0.04
    type_bonus: 0.12,    // proportionally scaled 0.10→0.12
};

// ── Retriever ───────────────────────────────────────────────────────────────

pub struct HybridMemoryRetriever {
    embedder: HashEmbedding,
    /// Optional external embedding provider (e.g., Ollama via `embed-ollama`).
    /// Clones of the retriever will drop the external provider and fall back
    /// to HashEmbedding, since `Box<dyn EmbeddingProvider>` is not `Clone`.
    external: Option<Box<dyn EmbeddingProvider>>,
}

impl Clone for HybridMemoryRetriever {
    fn clone(&self) -> Self {
        Self {
            embedder: self.embedder.clone(),
            external: None,
        }
    }
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
            external: None,
        }
    }

    pub fn with_embedder(provider: Box<dyn EmbeddingProvider>) -> Self {
        let dims = provider.dimensions();
        Self {
            embedder: HashEmbedding::new(dims),
            external: Some(provider),
        }
    }

    /// Embed query text — prefers the external provider when available,
    /// falling back to the deterministic `HashEmbedding`.
    fn embed_query(&self, text: &str) -> Vec<f32> {
        if let Some(ref provider) = self.external {
            provider.embed(text)
        } else {
            self.embedder.embed(text)
        }
    }

    /// Embed memory content — same selection as `embed_query`.
    fn embed_memory(&self, content: &str) -> Vec<f32> {
        if let Some(ref provider) = self.external {
            provider.embed(content)
        } else {
            self.embedder.embed(content)
        }
    }

    // ── Main search ─────────────────────────────────────────────────────────

    pub fn search<S: MemoryStore>(
        &self,
        store: &S,
        query: &MemoryQuery,
    ) -> StoreResult<Vec<MemoryPacket>> {
        // Expand candidate pool to cover all memories in namespace.
        let mut list_query = query.clone();
        list_query.limit = list_query.limit.max(5_000);
        let memories = store.list_memories(&list_query)?;
        if memories.is_empty() {
            return Ok(Vec::new());
        }

        let weights = if self.external.is_some() {
            &WEIGHTS_OLLAMA
        } else {
            &WEIGHTS_HASH
        };

        // Pre-compute embeddings
        let query_embedding = self.embed_query(&query.text);
        let memory_embeddings: HashMap<String, Vec<f32>> = memories
            .iter()
            .map(|m| (m.id.clone(), self.embed_memory(&m.content)))
            .collect();

        // Build BM25 corpus stats
        let corpus_stats = build_bm25_stats(&memories);

        // Extract query entities
        let query_entities = extract_query_entities(&query.text);
        let query_lower = query.text.to_lowercase();
        let query_tokens: BTreeSet<String> = tokenize(&query.text).into_iter().collect();
        let query_token_count = query_tokens.len().max(1) as f32;

        let mut packets: Vec<MemoryPacket> = memories
            .into_iter()
            .map(|memory| {
                let mem_emb = memory_embeddings
                    .get(&memory.id)
                    .cloned()
                    .unwrap_or_default();
                let mem_tokens: BTreeSet<String> = tokenize(&memory.content).into_iter().collect();
                let mem_word_count = memory.content.split_whitespace().count().max(1) as f32;

                // ── 1. cosine similarity ──────────────────────────────
                let cosine = cosine_similarity(&query_embedding, &mem_emb).max(0.0);

                // ── 2. term_density (Jaccard) ─────────────────────────
                // What fraction of query tokens appear in this memory?
                let intersection = query_tokens.intersection(&mem_tokens).count() as f32;
                let union = query_tokens.union(&mem_tokens).count() as f32;
                let term_density = if union > 0.0 {
                    intersection / union
                } else {
                    0.0
                };

                // ── 3. BM25 term_freq (length-normalized) ─────────────
                let bm25_raw = bm25_score(&query.text, &memory.content, &corpus_stats);
                // Normalize by memory word count (cap at MAX_LEN_NORM) to avoid
                // long session transcripts dominating.
                let _len_norm = mem_word_count.min(MAX_LEN_NORM);
                let term_freq = bm25_raw;

                // ── 4. lexical (full token overlap Jaccard) ───────────
                let lexical = token_overlap_score(&query.text, &memory.content);

                // ── 5. entity boost ────────────────────────────────────
                let mem_entities = memory
                    .metadata
                    .get("entities")
                    .map(|s| {
                        s.split('\n')
                            .map(str::trim)
                            .filter(|e| !e.is_empty())
                            .map(str::to_lowercase)
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default();
                let entity_boost = compute_entity_boost(&query_entities, &mem_entities);

                // ── 5b. temporal boost (Phase 4) ────────────────────────────
                // Prefer earlier memories when entity match exists
                let temporal_boost = if entity_boost > 0.05 {
                    turn_index_from_source(memory.source_event_id.as_deref())
                        .map(|idx| 1.0 / (1.0 + idx as f32))
                        .unwrap_or(0.0)
                } else {
                    0.0
                };

                // ── 6. speaker match ──────────────────────────────────
                let speaker = memory
                    .metadata
                    .get("speaker")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                let speaker_bonus = if !speaker.is_empty() && query_lower.contains(&speaker) {
                    1.0
                } else {
                    0.0
                };

                // ── 7. type bonus ─────────────────────────────────────
                let kind = memory
                    .metadata
                    .get("memory_kind")
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let type_factor: f32 = match kind {
                    "verbatim_turn" => 1.2,
                    "verbatim_session" => 0.3,
                    "llm_fact" => 1.1,
                    "observation" => 1.0,
                    _ => 0.8,
                };

                // ── 8. importance ─────────────────────────────────────
                let importance = memory.importance as f32;

                // ── 9. prefix/stem bonus — close the "researching"≠"research" gap
                let prefix_bonus: f32 = query_tokens
                    .iter()
                    .map(|qt| {
                        mem_tokens
                            .iter()
                            .map(|mt| {
                                if qt == mt {
                                    1.0
                                } else if qt.len() >= 4
                                    && mt.len() >= 4
                                    && (mt.starts_with(qt.as_str()) || qt.starts_with(mt.as_str()))
                                {
                                    0.5
                                } else {
                                    0.0
                                }
                            })
                            .fold(0.0f32, |a, b| a.max(b))
                    })
                    .sum::<f32>()
                    / query_token_count;

                // ── final weighted score ──────────────────────────────
                let score = weights.vector * cosine
                    + weights.term_density * term_density
                    + weights.term_freq * term_freq
                    + weights.lexical * lexical
                    + weights.entity_bonus * entity_boost
                    + weights.speaker_match * speaker_bonus
                    + weights.importance * importance
                    + weights.type_bonus * type_factor
                    + 0.25 * prefix_bonus
                    + 0.1 * temporal_boost; // Phase 4: recency signal

                let mut reasons = Vec::new();
                if cosine > 0.3 {
                    reasons.push("semantic_match".to_string());
                } else if cosine > 0.1 {
                    reasons.push("weak_semantic_match".to_string());
                }
                if term_freq > 0.1 {
                    reasons.push("keyword_match".to_string());
                } else if term_density > 0.1 {
                    reasons.push("weak_keyword_match".to_string());
                }
                if entity_boost > 0.05 {
                    reasons.push("entity_match".to_string());
                }
                if speaker_bonus > 0.0 {
                    reasons.push("speaker_match".to_string());
                }
                if reasons.is_empty() {
                    reasons.push("low_relevance".to_string());
                }

                MemoryPacket {
                    memory,
                    score,
                    reasons,
                }
            })
            .collect();

        // Sort descending by score and apply source diversity dedup
        packets.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });

        // Dedup: keep at most 2 results per source_event_id so a single
        // turn doesn't monopolize top-K (pattern from HeuristicReranker).
        let mut source_counts: HashMap<String, usize> = HashMap::new();
        packets.retain(|packet| {
            if let Some(ref source) = packet.memory.source_event_id {
                let count = source_counts.entry(source.clone()).or_insert(0);
                if *count >= 2 {
                    return false;
                }
                *count += 1;
            }
            true
        });

        packets.truncate(query.limit);
        Ok(packets)
    }
}

// ── BM25 implementation ─────────────────────────────────────────────────────

/// Per-corpus statistics needed for BM25 IDF computation.
#[derive(Clone, Debug)]
struct Bm25Stats {
    doc_count: usize,
    avg_doc_len: f32,
    doc_freq: BTreeMap<String, usize>,
}

fn build_bm25_stats(memories: &[Memory]) -> Bm25Stats {
    let doc_count = memories.len();
    if doc_count == 0 {
        return Bm25Stats {
            doc_count: 0,
            avg_doc_len: 0.0,
            doc_freq: BTreeMap::new(),
        };
    }

    let docs: Vec<BTreeMap<String, u32>> = memories
        .iter()
        .map(|m| {
            let tokens = tokenize(&m.content);
            let mut tf = BTreeMap::new();
            for t in tokens {
                *tf.entry(t).or_insert(0) += 1;
            }
            tf
        })
        .collect();

    let total_len: usize = docs.iter().map(|d| d.values().sum::<u32>() as usize).sum();
    let avg_doc_len = total_len as f32 / doc_count as f32;

    let mut doc_freq: BTreeMap<String, usize> = BTreeMap::new();
    for doc in &docs {
        let mut seen = BTreeSet::new();
        for term in doc.keys() {
            if seen.insert(term.clone()) {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }
    }

    Bm25Stats {
        doc_count,
        avg_doc_len,
        doc_freq,
    }
}

fn bm25_score(query: &str, doc: &str, stats: &Bm25Stats) -> f32 {
    if stats.doc_count == 0 {
        return 0.0;
    }

    let doc_tokens = tokenize(doc);
    let doc_len = doc_tokens.len() as f32;
    let mut doc_tf: BTreeMap<String, u32> = BTreeMap::new();
    for t in &doc_tokens {
        *doc_tf.entry(t.clone()).or_insert(0) += 1;
    }

    let query_terms: BTreeSet<String> = tokenize(query).into_iter().collect();

    let mut score = 0.0f32;

    for term in &query_terms {
        let df = *stats.doc_freq.get(term).unwrap_or(&0) as f32;
        if df == 0.0 {
            continue;
        }

        let idf = ((stats.doc_count as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

        let tf = *doc_tf.get(term).unwrap_or(&0) as f32;
        if tf == 0.0 {
            continue;
        }

        let numerator = tf * (BM25_K1 + 1.0);
        let denominator =
            tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / stats.avg_doc_len.max(1.0)));
        score += idf * (numerator / denominator);
    }

    score
}

// ── Phase 4: temporal boost helper ──────────────────────────────────────────

/// Extract the turn index from a source_event_id string.
/// Handles formats like "locomo_conv_1_turn_5", "turn_5", "5", etc.
fn turn_index_from_source(source: Option<&str>) -> Option<usize> {
    let source = source?;
    // Try the last underscore-separated component
    for part in source.rsplit('_') {
        if let Ok(n) = part.parse::<usize>() {
            return Some(n);
        }
    }
    // Try the whole string
    source.parse::<usize>().ok()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let memories = vec![
            Memory::new(
                "Rust is a systems programming language",
                crate::models::MemoryType::Semantic,
            ),
            Memory::new(
                "Python is popular for data science",
                crate::models::MemoryType::Semantic,
            ),
            Memory::new(
                "Rust guarantees memory safety without garbage collection",
                crate::models::MemoryType::Semantic,
            ),
        ];
        let stats = build_bm25_stats(&memories);
        let s0 = bm25_score("Rust systems", &memories[0].content, &stats);
        let s1 = bm25_score("Rust systems", &memories[1].content, &stats);
        assert!(
            s0 > s1,
            "doc about Rust should score higher for 'Rust systems' query"
        );
    }

    #[test]
    fn test_bm25_exact_match() {
        let doc = "The quick brown fox jumps over the lazy dog";
        let stats = Bm25Stats {
            doc_count: 1,
            avg_doc_len: tokenize(doc).len() as f32,
            doc_freq: tokenize(doc).into_iter().fold(BTreeMap::new(), |mut m, t| {
                m.entry(t).or_insert(1);
                m
            }),
        };
        let score = bm25_score("quick fox", doc, &stats);
        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_no_match() {
        let doc = "hello world";
        let stats = build_bm25_stats(&[Memory::new(doc, crate::models::MemoryType::Semantic)]);
        let score = bm25_score("foobar", doc, &stats);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_corpus() {
        let stats = build_bm25_stats(&[]);
        assert_eq!(stats.doc_count, 0);
        let score = bm25_score("anything", "whatever", &stats);
        assert!((score - 0.0).abs() < 0.001);
    }
}
