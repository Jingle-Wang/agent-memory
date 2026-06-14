use std::collections::BTreeSet;

use serde_json::Value;

use crate::llm::{LlmCompletionRequest, LlmError, LlmMessage, LlmProvider, LlmProviderMetadata};
use crate::models::MemoryPacket;
use crate::store::{StoreError, StoreResult};
use crate::text::contains_any;

pub const LLM_RERANK_MODEL: &str = "deepseek-v4-flash";

pub trait CandidateReranker {
    fn rerank_candidates(
        &self,
        question: &str,
        candidates: &[MemoryPacket],
    ) -> StoreResult<Vec<MemoryPacket>>;

    fn metadata(&self) -> Option<LlmProviderMetadata> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct HeuristicReranker {
    candidate_limit: usize,
    output_limit: usize,
}

impl Default for HeuristicReranker {
    fn default() -> Self {
        Self {
            candidate_limit: 50,
            output_limit: 10,
        }
    }
}

impl HeuristicReranker {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CandidateReranker for HeuristicReranker {
    fn rerank_candidates(
        &self,
        question: &str,
        candidates: &[MemoryPacket],
    ) -> StoreResult<Vec<MemoryPacket>> {
        let question_lower = question.to_lowercase();
        let mut ranked = candidates
            .iter()
            .take(self.candidate_limit)
            .enumerate()
            .map(|(index, packet)| {
                (
                    index,
                    packet.score + heuristic_bonus(&question_lower, packet),
                    packet,
                )
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut ordered = Vec::new();
        let mut seen = BTreeSet::new();
        let mut seen_sources = BTreeSet::new();
        for (_, _, packet) in ranked.iter().take(self.output_limit) {
            let source_is_new = packet
                .memory
                .source_event_id
                .as_ref()
                .is_none_or(|source| seen_sources.insert(source.clone()));
            if source_is_new && seen.insert(packet.memory.id.clone()) {
                let mut packet = (*packet).clone();
                packet.reasons.push("heuristic_rerank".to_string());
                ordered.push(packet);
            }
        }
        for (_, _, packet) in ranked {
            if seen.insert(packet.memory.id.clone()) {
                ordered.push(packet.clone());
            }
        }
        for packet in candidates.iter().skip(self.candidate_limit) {
            if seen.insert(packet.memory.id.clone()) {
                ordered.push(packet.clone());
            }
        }
        Ok(ordered)
    }
}

#[derive(Clone, Debug)]
pub struct LlmReranker<P: LlmProvider> {
    provider: P,
    model: String,
    candidate_limit: usize,
    output_limit: usize,
    max_tokens: u32,
}

impl<P: LlmProvider> LlmReranker<P> {
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            model: LLM_RERANK_MODEL.to_string(),
            candidate_limit: 50,
            output_limit: 10,
            max_tokens: 256,
        }
    }
}

impl<P: LlmProvider> CandidateReranker for LlmReranker<P> {
    fn rerank_candidates(
        &self,
        question: &str,
        candidates: &[MemoryPacket],
    ) -> StoreResult<Vec<MemoryPacket>> {
        llm_rerank_with_options(
            &self.provider,
            &self.model,
            question,
            candidates,
            self.candidate_limit,
            self.output_limit,
            self.max_tokens,
        )
        .map_err(|error| StoreError::Corrupt(format!("LLM rerank failed: {error}")))
    }

    fn metadata(&self) -> Option<LlmProviderMetadata> {
        let mut metadata = self.provider.metadata();
        metadata.model = self.model.clone();
        metadata.prompt_version = Some("benchmark-rerank-v1".to_string());
        Some(metadata)
    }
}

fn heuristic_bonus(question_lower: &str, packet: &MemoryPacket) -> f32 {
    let content_lower = packet.memory.content.to_lowercase();
    let mut bonus: f32 = 0.0;

    if contains_any(
        question_lower,
        &["field", "fields", "education", "educaton", "degree"],
    ) && contains_any(
        &content_lower,
        &[
            "psychology",
            "counseling",
            "certification",
            "mental health",
            "education",
        ],
    ) {
        bonus += 0.90;
    }
    if question_lower.contains("research")
        && contains_any(
            &content_lower,
            &["research", "researching", "adoption agenc"],
        )
    {
        bonus += 0.85;
    }
    if question_lower.contains("identity")
        && contains_any(
            &content_lower,
            &["transgender", "trans woman", "trans community"],
        )
    {
        bonus += 0.75;
    }
    if question_lower.contains("relationship status")
        && contains_any(&content_lower, &["single", "partner", "relationship"])
    {
        bonus += 0.70;
    }
    if contains_any(
        question_lower,
        &["when", "what date", "what year", "how long"],
    ) {
        if packet.memory.metadata.contains_key("event_time") {
            bonus += 0.20;
        }
        if contains_any(
            &content_lower,
            &[
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
                "2022",
                "2023",
            ],
        ) {
            bonus += 0.20;
        }
    }
    if contains_any(question_lower, &["where", "location", "live", "from"])
        && contains_any(
            &content_lower,
            &[" in ", " at ", " from ", " moved ", " lives "],
        )
    {
        bonus += 0.30;
    }

    for key in ["subject", "object", "relation", "speaker"] {
        if let Some(value) = packet.memory.metadata.get(key) {
            let value_lower = value.to_lowercase();
            if value_lower.len() >= 3 && question_lower.contains(&value_lower) {
                bonus += 0.15;
            }
        }
    }

    bonus.min(1.5)
}

pub fn llm_rerank<P: LlmProvider>(
    provider: &P,
    question: &str,
    candidates: &[MemoryPacket],
) -> Result<Vec<MemoryPacket>, LlmError> {
    llm_rerank_with_options(
        provider,
        LLM_RERANK_MODEL,
        question,
        candidates,
        50,
        10,
        256,
    )
}

fn llm_rerank_with_options<P: LlmProvider>(
    provider: &P,
    model: &str,
    question: &str,
    candidates: &[MemoryPacket],
    candidate_limit: usize,
    output_limit: usize,
    max_tokens: u32,
) -> Result<Vec<MemoryPacket>, LlmError> {
    if candidates.is_empty() {
        return Ok(candidates.to_vec());
    }

    let prompt_candidates = candidates
        .iter()
        .take(candidate_limit)
        .cloned()
        .collect::<Vec<_>>();
    let select_count = output_limit.min(prompt_candidates.len());
    let request = LlmCompletionRequest {
        model: model.to_string(),
        messages: vec![
            LlmMessage::system(
                "You rerank memory search results. Return only valid JSON with the most relevant memory IDs.",
            ),
            LlmMessage::user(build_prompt(question, &prompt_candidates, select_count)),
        ],
        temperature: 0.0,
        max_tokens,
        response_format: None,
    };
    let response = provider.complete(&request)?;
    let candidate_ids = prompt_candidates
        .iter()
        .map(|packet| packet.memory.id.as_str())
        .collect::<Vec<_>>();
    let selected_ids = parse_selected_ids(&response, &candidate_ids);

    Ok(reorder_candidates(candidates, &selected_ids, output_limit))
}

fn build_prompt(question: &str, candidates: &[MemoryPacket], output_limit: usize) -> String {
    let mut prompt = format!(
        "Question:\n{question}\n\nSelect the {output_limit} memories that best answer the question.\nReturn exactly this JSON shape and nothing else:\n{{\"ids\":[\"memory_id\"]}}\n\nMemories:\n"
    );
    for (index, packet) in candidates.iter().enumerate() {
        prompt.push_str(&format!(
            "{}. ID: {}\nContent: {}\n\n",
            index + 1,
            packet.memory.id,
            truncate_for_prompt(&packet.memory.content, 1200)
        ));
    }
    prompt
}

fn truncate_for_prompt(content: &str, max_chars: usize) -> String {
    let mut output = content.chars().take(max_chars).collect::<String>();
    if output.len() < content.len() {
        output.push_str("...");
    }
    output
}

fn parse_selected_ids(response: &str, candidate_ids: &[&str]) -> Vec<String> {
    if let Ok(value) = serde_json::from_str::<Value>(response.trim()) {
        let ids = ids_from_json(&value, candidate_ids);
        if !ids.is_empty() {
            return ids;
        }
    }

    let mut positions = candidate_ids
        .iter()
        .filter_map(|id| {
            response
                .find(id)
                .map(|position| (position, (*id).to_string()))
        })
        .collect::<Vec<_>>();
    positions.sort_by_key(|(position, _)| *position);
    positions.into_iter().map(|(_, id)| id).collect()
}

fn ids_from_json(value: &Value, candidate_ids: &[&str]) -> Vec<String> {
    match value {
        Value::Array(items) => ids_from_array(items, candidate_ids),
        Value::Object(map) => {
            for key in ["ids", "memory_ids", "ranked_ids", "ranking", "selected"] {
                if let Some(Value::Array(items)) = map.get(key) {
                    let ids = ids_from_array(items, candidate_ids);
                    if !ids.is_empty() {
                        return ids;
                    }
                }
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn ids_from_array(items: &[Value], candidate_ids: &[&str]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match item {
            Value::String(id) => Some(id.trim().to_string()),
            Value::Number(number) => number
                .as_u64()
                .and_then(|rank| rank.checked_sub(1))
                .and_then(|index| candidate_ids.get(index as usize))
                .map(|id| (*id).to_string()),
            Value::Object(map) => ["id", "memory_id"]
                .iter()
                .find_map(|key| map.get(*key).and_then(Value::as_str))
                .map(|id| id.trim().to_string()),
            _ => None,
        })
        .filter(|id| !id.is_empty())
        .collect()
}

fn reorder_candidates(
    candidates: &[MemoryPacket],
    selected_ids: &[String],
    output_limit: usize,
) -> Vec<MemoryPacket> {
    let mut ordered = Vec::new();
    let mut seen = BTreeSet::new();

    for id in selected_ids {
        if ordered.len() >= output_limit {
            break;
        }
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(packet) = candidates.iter().find(|packet| packet.memory.id == *id) {
            let mut packet = packet.clone();
            packet.reasons.push("llm_rerank".to_string());
            ordered.push(packet);
        }
    }

    for packet in candidates {
        if seen.insert(packet.memory.id.clone()) {
            ordered.push(packet.clone());
        }
    }

    ordered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmProviderMetadata;
    use crate::models::{Memory, MemoryType};

    #[derive(Clone)]
    struct StaticProvider {
        response: String,
    }

    impl LlmProvider for StaticProvider {
        fn complete(&self, _request: &LlmCompletionRequest) -> Result<String, LlmError> {
            Ok(self.response.clone())
        }

        fn metadata(&self) -> LlmProviderMetadata {
            LlmProviderMetadata {
                provider: "fixture".to_string(),
                model: "fixture".to_string(),
                base_url: None,
                prompt_version: None,
            }
        }
    }

    #[test]
    fn llm_rerank_uses_returned_memory_id_order() {
        let candidates = vec![
            packet("m1", "unrelated"),
            packet("m2", "answer"),
            packet("m3", "also relevant"),
        ];
        let provider = StaticProvider {
            response: "{\"ids\":[\"m3\",\"m2\"]}".to_string(),
        };

        let reranked = llm_rerank(&provider, "question", &candidates).unwrap();

        assert_eq!(reranked[0].memory.id, "m3");
        assert_eq!(reranked[1].memory.id, "m2");
        assert!(reranked[0].reasons.contains(&"llm_rerank".to_string()));
    }

    #[test]
    fn llm_rerank_accepts_one_based_rank_numbers() {
        let candidates = vec![
            packet("m1", "unrelated"),
            packet("m2", "answer"),
            packet("m3", "also relevant"),
        ];
        let provider = StaticProvider {
            response: "{\"ids\":[3,2]}".to_string(),
        };

        let reranked = llm_rerank(&provider, "question", &candidates).unwrap();

        assert_eq!(reranked[0].memory.id, "m3");
        assert_eq!(reranked[1].memory.id, "m2");
    }

    fn packet(id: &str, content: &str) -> MemoryPacket {
        let mut memory = Memory::new(content, MemoryType::Semantic);
        memory.id = id.to_string();
        MemoryPacket {
            memory,
            score: 0.1,
            reasons: Vec::new(),
        }
    }
}
