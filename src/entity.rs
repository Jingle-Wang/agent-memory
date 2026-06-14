//! Entity extraction and cross-memory entity linking.
//!
//! This module replaces the naive uppercase-first-letter entity detection
//! previously in retriever.rs with regex-based heuristics that extract:
//! - Proper nouns (consecutive uppercase-starting words: "New York", "Alice")
//! - Quoted text ("the red door")
//! - Title-case compounds ("Machine Learning")
//! - Individual proper nouns as fallback
//!
//! The approach mirrors mem0's spaCy-based entity extraction but uses
//! pure Rust heuristics to avoid Python subprocess overhead.

use std::collections::{BTreeMap, BTreeSet};

use crate::models::Memory;

/// Known question/determiner words that start uppercase but aren't entities.
const NON_ENTITY_WORDS: &[&str] = &[
    "What", "When", "Where", "Who", "How", "Did", "Does", "Is", "Are", "Was", "Were", "Which",
    "Whose", "Whom", "The", "A", "An", "This", "That", "These", "Those", "My", "Your", "His",
    "Her", "Our", "Their", "It", "Its", "I", "You", "He", "She", "We", "They", "And", "But", "Or",
    "For", "Nor", "So", "Yet", "In", "On", "At", "To", "From", "By", "With", "About", "Into",
    "Through", "During", "Before", "After", "Above", "Below", "Between", "Under", "Over", "Of",
    "Can", "Will", "Would", "Could", "Should", "May", "Might", "Must", "Not", "No", "Yes", "If",
    "Then", "Else", "When", "While", "Since", "Just", "Only", "Also", "Still", "Even", "Very",
    "Too", "All", "Some", "Any", "Each", "Every", "Both", "Few", "Many", "More", "Most",
];

/// Check if a word is a non-entity uppercase word (question words, determiners, etc).
fn is_non_entity_word(word: &str) -> bool {
    NON_ENTITY_WORDS.contains(&word)
}

/// Check if a token looks like a proper noun (starts with uppercase, not a known non-entity).
fn is_proper_candidate(token: &str) -> bool {
    let clean = token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric());
    if clean.is_empty() || clean.len() == 1 {
        return false;
    }
    // Must start with uppercase ASCII letter
    if !clean
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
    {
        return false;
    }
    // Exclude known non-entity words
    if is_non_entity_word(clean) {
        return false;
    }
    // Exclude all-uppercase acronyms shorter than 2 chars (single letters)
    // But keep acronyms like "AI", "ML", "NASA"
    true
}

/// Extract entity strings from text.
///
/// This implements heuristic entity extraction that captures:
/// 1. **Quoted text** — anything in double quotes (e.g., `"the red door"`)
/// 2. **Proper noun sequences** — consecutive uppercase words (e.g., "New York City")
/// 3. **Individual proper nouns** — single uppercase words (e.g., "Rust", "Alice")
///
/// Returns a set of lowercased entity strings.
pub fn extract_entities(text: &str) -> BTreeSet<String> {
    let mut entities = BTreeSet::new();

    // === 1. Extract quoted text ===
    let mut in_quote = false;
    let mut quote_start = 0usize;
    for (i, ch) in text.char_indices() {
        if ch == '"' || ch == '\u{201c}' || ch == '\u{201d}' {
            if in_quote {
                let quoted = text[quote_start..i].trim().to_string();
                if quoted.len() > 1
                    && !quoted.chars().all(|c| c.is_whitespace())
                    && !quoted.chars().all(|c| c.is_ascii_punctuation())
                {
                    entities.insert(quoted.to_lowercase());
                }
                in_quote = false;
            } else {
                quote_start = i + ch.len_utf8();
                in_quote = true;
            }
        }
    }

    // === 2. Extract proper noun sequences and individual proper nouns ===
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        let token = words[i];
        if is_proper_candidate(token) {
            let clean = token
                .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
                .to_string();

            // Collect consecutive proper nouns
            let mut compound = vec![clean.clone()];
            let mut j = i + 1;
            while j < words.len() {
                if is_proper_candidate(words[j]) {
                    let next_clean = words[j]
                        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
                        .to_string();
                    compound.push(next_clean);
                    j += 1;
                } else {
                    break;
                }
            }

            if compound.len() > 1 {
                // Add full compound
                entities.insert(compound.join(" ").to_lowercase());
            }
            // Add individual parts
            for part in &compound {
                entities.insert(part.to_lowercase());
            }

            i = j;
        } else {
            i += 1;
        }
    }

    // === 3. Extract title-case compounds from text ===
    // Walk through the text and find sequences where each word is title-case
    // (first letter uppercase, rest lowercase or all uppercase for acronyms)
    let raw_words: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    while i < raw_words.len() {
        let token = raw_words[i];
        let clean = token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric());
        let is_title = clean
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())
            && clean.chars().skip(1).all(|ch| ch.is_ascii_lowercase());

        let is_acronym = clean.len() >= 2 && clean.chars().all(|ch| ch.is_ascii_uppercase());

        if (is_title || is_acronym) && !is_non_entity_word(clean) {
            let mut compound = vec![clean.to_string()];
            let mut j = i + 1;
            while j < raw_words.len() {
                let next = raw_words[j];
                let next_clean = next.trim_matches(|ch: char| !ch.is_ascii_alphanumeric());
                let next_title = next_clean
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_uppercase())
                    && next_clean.chars().skip(1).all(|ch| ch.is_ascii_lowercase());
                let next_acronym =
                    next_clean.len() >= 2 && next_clean.chars().all(|ch| ch.is_ascii_uppercase());
                if (next_title || next_acronym) && !is_non_entity_word(next_clean) {
                    compound.push(next_clean.to_string());
                    j += 1;
                } else {
                    break;
                }
            }
            if compound.len() > 1 {
                let full = compound.join(" ");
                entities.insert(full.to_lowercase());
            }
            i = j;
        } else {
            i += 1;
        }
    }

    entities
}

/// Extract entity strings from query text for use in QueryAnalysis.
///
/// This replaces the old `extract_query_entities` in retriever.rs.
/// Returns a BTreeSet of lowercased entity strings.
pub fn extract_query_entities(text: &str) -> BTreeSet<String> {
    let mut entities = BTreeSet::new();

    // Process whole text and also sentence-by-sentence for better coverage
    entities.extend(extract_entities(text));

    for sentence in text.split(|ch| ch == '.' || ch == '!' || ch == '?') {
        if !sentence.trim().is_empty() {
            entities.extend(extract_entities(sentence));
        }
    }

    entities
}

/// Extract entities from memory content for storage in metadata.
///
/// Returns a newline-joined string suitable for storing in `metadata["entities"]`.
pub fn extract_memory_entities(content: &str) -> String {
    let entities = extract_query_entities(content);
    if entities.is_empty() {
        return String::new();
    }
    let mut sorted: Vec<String> = entities.into_iter().collect();
    sorted.sort();
    sorted.join("\n")
}

/// Persist entity links on each memory so stores keep a durable entity index
/// without requiring a second storage backend.
pub fn enrich_memory_entities(memory: &mut Memory) {
    let mut entities = extract_query_entities(&memory.content);
    if let Some(existing) = memory.metadata.get("entities") {
        entities.extend(
            existing
                .split('\n')
                .map(str::trim)
                .filter(|entity| !entity.is_empty())
                .map(str::to_lowercase),
        );
    }
    for key in ["subject", "object"] {
        if let Some(value) = memory.metadata.get(key) {
            entities.extend(extract_query_entities(value));
            if value.len() > 1 {
                entities.insert(value.to_lowercase());
            }
        }
    }
    if !entities.is_empty() {
        memory.metadata.insert(
            "entities".to_string(),
            entities.into_iter().collect::<Vec<_>>().join("\n"),
        );
    }
}

/// Inverted entity links reconstructed from metadata persisted by the store.
#[derive(Clone, Debug, Default)]
pub struct EntityLinkIndex {
    memory_ids_by_entity: BTreeMap<String, BTreeSet<String>>,
}

impl EntityLinkIndex {
    pub fn from_memories(memories: &[Memory]) -> Self {
        let mut index = Self::default();
        for memory in memories {
            let Some(entities) = memory.metadata.get("entities") else {
                continue;
            };
            for entity in entities
                .split('\n')
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                index
                    .memory_ids_by_entity
                    .entry(entity.to_lowercase())
                    .or_default()
                    .insert(memory.id.clone());
            }
        }
        index
    }

    pub fn linked_memory_ids(&self, entities: &BTreeSet<String>) -> BTreeSet<String> {
        let mut memory_ids = BTreeSet::new();
        for entity in entities {
            if let Some(linked) = self.memory_ids_by_entity.get(entity) {
                memory_ids.extend(linked.iter().cloned());
            }
        }
        memory_ids
    }
}

/// Count how many entities are shared between query entities and memory entities.
///
/// This is used for the entity_boost computation in the scoring model.
pub fn count_shared_entities(
    query_entities: &BTreeSet<String>,
    memory_entities: &BTreeSet<String>,
) -> usize {
    query_entities.intersection(memory_entities).count()
}

/// Compute entity boost for cross-memory entity linking.
///
/// For each memory, count shared entities with query entities.
/// Returns a value between 0.0 and 0.5 (ENTITY_BOOST_WEIGHT).
pub fn compute_entity_boost(
    query_entities: &BTreeSet<String>,
    memory_entities: &BTreeSet<String>,
) -> f32 {
    let shared = count_shared_entities(query_entities, memory_entities);
    if shared > 0 {
        (shared as f32 * 0.50).min(1.2)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_proper_nouns() {
        let entities = extract_query_entities("Alice visited New York with Bob");
        assert!(entities.contains("alice"));
        assert!(entities.contains("new york"));
        assert!(entities.contains("bob"));
        assert!(entities.contains("new"));
        assert!(entities.contains("york"));
    }

    #[test]
    fn test_extract_quoted_text() {
        let entities = extract_query_entities("She called it \"the red door\" and left");
        assert!(entities.contains("the red door"));
    }

    #[test]
    fn test_extract_title_case_compound() {
        let entities = extract_query_entities("I work on Machine Learning at Google");
        assert!(entities.contains("machine learning"));
        assert!(entities.contains("google"));
    }

    #[test]
    fn test_ignore_question_words() {
        let entities = extract_query_entities("What is the capital of France");
        assert!(!entities.contains("what"));
        assert!(entities.contains("france"));
    }

    #[test]
    fn test_extract_acronyms() {
        let entities = extract_query_entities("The AI and ML teams at NASA");
        assert!(entities.contains("ai"));
        assert!(entities.contains("ml"));
        assert!(entities.contains("nasa"));
    }

    #[test]
    fn test_count_shared_entities() {
        let query: BTreeSet<String> = ["rust", "python", "ai"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let memory: BTreeSet<String> = ["rust", "golang", "ai"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(count_shared_entities(&query, &memory), 2);
    }

    #[test]
    fn test_compute_entity_boost() {
        let query: BTreeSet<String> = ["rust", "python", "ai", "ml", "nasa"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let memory: BTreeSet<String> = ["rust", "ai", "ml"].iter().map(|s| s.to_string()).collect();
        // 3 shared * 0.50 = 1.5, capped at 1.2
        let boost = compute_entity_boost(&query, &memory);
        assert!((boost - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_memory_entities() {
        let result = extract_memory_entities("Alice and Bob work at Google on AI");
        assert!(result.contains("alice"));
        assert!(result.contains("bob"));
        assert!(result.contains("google"));
        assert!(result.contains("ai"));
    }

    #[test]
    fn test_persisted_entity_link_index() {
        let mut memory = Memory::new(
            "Alice visited New York",
            crate::models::MemoryType::Episodic,
        );
        enrich_memory_entities(&mut memory);
        let index = EntityLinkIndex::from_memories(&[memory.clone()]);
        let query = extract_query_entities("What did Alice do?");

        assert!(memory.metadata["entities"].contains("alice"));
        assert!(index.linked_memory_ids(&query).contains(&memory.id));
    }
}
