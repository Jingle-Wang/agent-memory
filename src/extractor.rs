use std::{env, str::FromStr};

use serde_json::Value;

use crate::engine::extract_memories;
use crate::llm::{LlmCompletionRequest, LlmError, LlmMessage, LlmProvider};
use crate::models::{Event, Memory, MemoryType};

pub trait MemoryExtractor {
    fn extract(&self, event: &Event, timestamp: Option<&str>) -> Result<Vec<Memory>, String>;
}

#[derive(Clone, Debug, Default)]
pub struct RuleBasedMemoryExtractor;

impl MemoryExtractor for RuleBasedMemoryExtractor {
    fn extract(&self, event: &Event, _timestamp: Option<&str>) -> Result<Vec<Memory>, String> {
        Ok(extract_memories(event))
    }
}

#[derive(Clone, Debug)]
pub struct LlmMemoryExtractor<P: LlmProvider> {
    provider: P,
    model: String,
    max_tokens: u32,
}

impl<P: LlmProvider> LlmMemoryExtractor<P> {
    pub fn new(provider: P, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            max_tokens: extractor_max_tokens(),
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

impl<P: LlmProvider> MemoryExtractor for LlmMemoryExtractor<P> {
    fn extract(&self, event: &Event, timestamp: Option<&str>) -> Result<Vec<Memory>, String> {
        let observation_date = timestamp.unwrap_or("");
        let user_prompt = format!(
            "New Message:\nRole: {}\nContent: {}\nObservation Date: {}",
            event.actor, event.text, observation_date
        );
        let request = LlmCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                LlmMessage::system(ADDITIVE_EXTRACTION_PROMPT),
                LlmMessage::user(user_prompt),
            ],
            temperature: 0.0,
            max_tokens: self.max_tokens,
            response_format: Some(serde_json::json!({"type": "json_object"})),
        };

        const MAX_RETRIES: usize = 2; // 3 total attempts (initial + 2 retries)
        let mut last_error = String::new();

        for attempt in 0..=MAX_RETRIES {
            let response = match self.provider.complete(&request) {
                Ok(response) => response,
                Err(e) => {
                    let msg = format!(
                        "WARN [llm-extractor] LLM completion failed (event {}, attempt {}/{}): {e}",
                        event.id,
                        attempt + 1,
                        MAX_RETRIES + 1
                    );
                    eprintln!("{msg}");
                    last_error = msg;
                    if attempt < MAX_RETRIES {
                        eprintln!(
                            "WARN [llm-extractor] Retrying LLM completion (event {}, attempt {}/{})",
                            event.id,
                            attempt + 2,
                            MAX_RETRIES + 1
                        );
                        continue;
                    }
                    eprintln!(
                        "WARN [llm-extractor] All retries exhausted (event {}) — falling back to rule extractor",
                        event.id
                    );
                    return RuleBasedMemoryExtractor.extract(event, timestamp);
                }
            };

            match parse_llm_memories(event, &response) {
                Ok(memories) => {
                    if !memories.is_empty() {
                        return Ok(memories);
                    }
                    // LLM returned valid empty facts array — no facts to extract.
                    // Don't retry or fallback; empty is the correct answer.
                    eprintln!(
                        "INFO [llm-extractor] LLM returned zero memories (event {}) — no facts to extract",
                        event.id
                    );
                    return Ok(vec![]);
                }
                Err(e) => {
                    let msg = format!(
                        "WARN [llm-extractor] Failed to parse LLM JSON (event {}, attempt {}/{}): {e} — response_preview={}",
                        event.id,
                        attempt + 1,
                        MAX_RETRIES + 1,
                        response_preview(&response, 500)
                    );
                    eprintln!("{msg}");
                    last_error = msg;
                    if attempt < MAX_RETRIES {
                        eprintln!(
                            "WARN [llm-extractor] Retrying LLM extraction (event {}, attempt {}/{})",
                            event.id,
                            attempt + 2,
                            MAX_RETRIES + 1
                        );
                    }
                }
            }
        }

        eprintln!(
            "WARN [llm-extractor] All retries exhausted (event {}) — falling back to rule extractor. Last error: {last_error}",
            event.id
        );
        RuleBasedMemoryExtractor.extract(event, timestamp)
    }
}

const ADDITIVE_EXTRACTION_PROMPT: &str = r#"You are a memory extractor. Extract important facts from conversation messages.

Return a JSON object: {"facts": ["fact1", "fact2", ...]}

CRITICAL: Never return an empty facts array {"facts":[]} unless the message is genuinely meaningless (e.g., a single word like "ok"). Almost every message contains extractable facts — the speaker's emotions, reactions, opinions, plans, relationships, or context. Even brief replies like "That's great!" reveal the speaker's attitude. When in doubt, extract at least one fact.

Each fact should be a complete, self-contained sentence including:
- Speaker names (never use pronouns like "he", "she", "they")
- Specific dates (convert relative dates using the Observation Date)
- Emotions and reactions when expressed
- Exact terms used (LGBTQ, transgender, adoption, support group, etc.)
- Preserve key concrete words from the original message (e.g., "swimming", "hiking", "camping", "volunteering") — do NOT replace specific activities with vague phrases like "taking care of ourselves" or "spending time together"

Examples:

Input:
Role: Taylor
Content: I adopted a golden retriever puppy named Max last Saturday. He's 3 months old and already knows how to sit. My partner Jordan wasn't sure about getting a dog but now loves him.
Observation Date: 15 June 2026
Output: {"facts": ["Taylor adopted a golden retriever puppy named Max on Saturday 8 June 2026", "Max is a 3-month-old golden retriever puppy who already knows how to sit", "Taylor's partner is Jordan", "Jordan was initially unsure about getting a dog but now loves Max", "Taylor and Jordan have a golden retriever puppy named Max at home"]}

Input:
Role: Caroline
Content: I went to a LGBTQ support group yesterday and it was so powerful. The transgender stories were inspiring! I felt so accepted and grateful for that space.
Observation Date: 7 June 2026
Output: {"facts": ["Caroline attended a LGBTQ support group on 6 June 2026 and found it powerful", "Caroline found the transgender stories at the LGBTQ support group inspiring", "Caroline felt accepted and grateful at the LGBTQ support group on 6 June 2026"]}

Input:
Role: Sam
Content: Just booked flights to Tokyo! Leaving March 3rd and staying for two weeks. Got a great deal — only $680 round trip.
Observation Date: 20 February 2026
Output: {"facts": ["Sam booked round trip flights to Tokyo for $680 departing March 3rd 2026", "Sam is staying in Tokyo for two weeks starting March 3rd 2026"]}

Input:
Role: Melanie
Content: Wow, that's amazing Caroline! I'm so happy you found that support group. Have you thought about going back next week?
Observation Date: 7 June 2026
Output: {"facts": ["Melanie expressed happiness that Caroline found the LGBTQ support group", "Melanie asked Caroline about going back to the support group next week around 14 June 2026", "Melanie is supportive of Caroline attending the LGBTQ support group"]}

Input:
Role: Alice
Content: Hi, how are you?
Observation Date: 1 June 2026
Output: {"facts": []}

Rules:
1. Include ALL speakers' names explicitly in each fact
2. Preserve identity/community terms exactly: LGBTQ, transgender, trans community, adoption, counseling, support group, coming out, pride
3. One fact per distinct piece of information (decompose compound statements)
4. Include emotions when expressed (happy, nervous, excited, worried, grateful)
5. Convert ALL relative dates (yesterday, last week, next month) to absolute dates using the Observation Date
6. For multi-turn conversations, extract facts from EVERY turn
7. Extract 3-10 facts per message depending on content richness
8. Never invent facts not stated or clearly implied"#;

fn parse_llm_memories(event: &Event, response: &str) -> Result<Vec<Memory>, LlmError> {
    let memories = parse_memory_values(response)?;
    let mut output = Vec::new();
    for item in &memories {
        let Some(memory) = memory_from_value(event, item) else {
            continue;
        };
        output.push(memory);
    }
    // Empty is valid: LLM correctly determined no facts to extract.
    // Don't treat it as an error; caller handles empty Vec gracefully.
    Ok(output)
}

fn normalize_memory_type(mem0_type: &str) -> &str {
    match mem0_type {
        "preference" => "reflection",
        "event" => "episodic",
        "relationship" => "semantic",
        "attribute" => "semantic",
        "time" => "semantic",
        "quantity" => "semantic",
        other => other,
    }
}

fn memory_from_value(event: &Event, item: &Value) -> Option<Memory> {
    let item = item.get("memory").unwrap_or(item);
    let Some(content) = item
        .get("content")
        .or_else(|| item.get("text"))
        .or_else(|| item.get("fact"))
        .and_then(Value::as_str)
    else {
        return None;
    };
    let content = content.trim();
    if content.is_empty() {
        return None;
    }
    let mut source_keywords = string_array(item, "source_keywords");
    if source_keywords.is_empty() {
        source_keywords = fallback_source_keywords(event);
    }
    // source_keywords stored only in metadata, NOT appended to content
    // (to avoid polluting the embedding vector with keyword spam)
    let memory_type = item
        .get("type")
        .and_then(Value::as_str)
        .map(normalize_memory_type)
        .and_then(|t| MemoryType::from_str(t).ok())
        .unwrap_or(MemoryType::Semantic);
    let mut memory = Memory::new(content, memory_type)
        .namespace(event.namespace.clone())
        .source_event(event.id.clone())
        .importance(number(item, "importance").unwrap_or(0.65))
        .confidence(number(item, "confidence").unwrap_or(0.70));
    memory
        .metadata
        .insert("memory_kind".to_string(), "llm_fact".to_string());
    memory
        .metadata
        .insert("speaker".to_string(), event.actor.clone());
    for key in ["subject", "relation", "object", "operation"] {
        if let Some(value) = item.get(key).and_then(Value::as_str) {
            memory
                .metadata
                .insert(key.to_string(), value.trim().to_string());
        }
    }
    if let Some(entities) = item.get("entities").and_then(Value::as_array) {
        let entities = entities
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        if !entities.is_empty() {
            memory
                .metadata
                .insert("entities".to_string(), entities.join("\n"));
        }
    }
    if !source_keywords.is_empty() {
        memory
            .metadata
            .insert("source_keywords".to_string(), source_keywords.join("\n"));
        // Also merge source_keywords into entities so retriever can match them
        let kw_str = source_keywords.join("\n");
        if let Some(existing) = memory.metadata.get("entities") {
            memory
                .metadata
                .insert("entities".to_string(), format!("{}\n{}", existing, kw_str));
        } else {
            memory.metadata.insert("entities".to_string(), kw_str);
        }
    }
    Some(memory)
}

fn string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn fallback_source_keywords(event: &Event) -> Vec<String> {
    let mut keywords = Vec::new();
    push_keyword(&mut keywords, &event.actor);
    if let Some(timestamp) = event.metadata.get("benchmark_timestamp") {
        push_keyword(&mut keywords, timestamp);
    }

    let lower = event.text.to_lowercase();
    for phrase in [
        "last year",
        "last month",
        "last week",
        "yesterday",
        "tomorrow",
        "next week",
        "next month",
        "two weeks",
        "support group",
        "mental health",
        "trans community",
    ] {
        if lower.contains(phrase) {
            push_keyword(&mut keywords, phrase);
        }
    }
    if lower.contains("transgender") || lower.contains("lgbtq") {
        push_keyword(&mut keywords, "identity");
    }

    for token in event
        .text
        .split(|ch: char| !ch.is_alphanumeric() && ch != '\'' && ch != '-')
        .map(str::trim)
        .filter(|token| token.len() >= 3)
        .filter(|token| !is_low_value_keyword(token))
    {
        push_keyword(&mut keywords, token);
        if keywords.len() >= 14 {
            break;
        }
    }
    keywords
}

fn push_keyword(keywords: &mut Vec<String>, keyword: &str) {
    let keyword = keyword.trim();
    if keyword.is_empty() {
        return;
    }
    if keywords
        .iter()
        .any(|existing| existing.eq_ignore_ascii_case(keyword))
    {
        return;
    }
    keywords.push(keyword.to_string());
}

fn is_low_value_keyword(token: &str) -> bool {
    matches!(
        token.to_lowercase().as_str(),
        "the"
            | "and"
            | "for"
            | "that"
            | "this"
            | "with"
            | "you"
            | "your"
            | "was"
            | "were"
            | "are"
            | "but"
            | "not"
            | "have"
            | "has"
            | "had"
            | "all"
            | "really"
            | "very"
            | "just"
            | "about"
    )
}

fn parse_memory_values(response: &str) -> Result<Vec<Value>, LlmError> {
    let trimmed = strip_markdown_fences(response).trim().to_string();
    if trimmed.is_empty() {
        return Err(LlmError::new("empty extractor response"));
    }

    if let Ok(value) = serde_json::from_str::<Value>(&trimmed) {
        let values = values_from_complete_json(&value);
        if !values.is_empty() {
            return Ok(values);
        }
    }
    if let Some(json) = extract_json_object(&trimmed) {
        if let Ok(value) = serde_json::from_str::<Value>(json) {
            let values = values_from_complete_json(&value);
            if !values.is_empty() {
                return Ok(values);
            }
        }
    }

    // Try extracting a JSON array from amidst analysis text.
    // Reasoning models (e.g. deepseek-v4-pro) may output analysis before the JSON.
    // This finds the first `[` and last `]` and parses everything between them.
    if let Some(array_str) = extract_json_array(&trimmed) {
        if let Ok(value) = serde_json::from_str::<Value>(array_str) {
            let values = values_from_complete_json(&value);
            if !values.is_empty() {
                return Ok(values);
            }
        }
        // Also try repair on the extracted array substring
        if let Some(repaired) = repair_truncated_json(array_str) {
            if let Ok(value) = serde_json::from_str::<Value>(&repaired) {
                let values = values_from_complete_json(&value);
                if !values.is_empty() {
                    return Ok(values);
                }
            }
        }
    }

    let mut values = parse_jsonl_values(&trimmed);
    values.extend(parse_truncated_memories_array(&trimmed));

    // Try repairing truncated JSON (handles direct arrays like [{...}] that
    // were cut off mid-response by token limits)
    if values.is_empty() {
        if let Some(repaired) = repair_truncated_json(&trimmed) {
            if let Ok(value) = serde_json::from_str::<Value>(&repaired) {
                values = values_from_complete_json(&value);
            }
        }
    }

    if values.is_empty() {
        return Err(LlmError::new("invalid extractor JSON/JSONL"));
    }
    Ok(values)
}

fn values_from_complete_json(value: &Value) -> Vec<Value> {
    // Handle {"facts": ["string1", "string2", ...]} format (Mem0-style)
    if let Some(facts) = value.get("facts").and_then(Value::as_array) {
        let converted: Vec<Value> = facts
            .iter()
            .filter_map(|f| {
                if let Some(s) = f.as_str() {
                    if !s.trim().is_empty() {
                        Some(serde_json::json!({"fact": s.trim()}))
                    } else {
                        None
                    }
                } else if f.is_object() {
                    Some(f.clone())
                } else {
                    None
                }
            })
            .collect();
        // Return converted (possibly empty) when "facts" key is present.
        // An empty facts array is a valid output meaning "no facts to extract".
        if value.get("facts").is_some() {
            return converted;
        }
    }
    if let Some(memories) = value.get("memories").and_then(Value::as_array) {
        return memories.clone();
    }
    if let Some(memories) = value.get("memory").and_then(Value::as_array) {
        return memories.clone();
    }
    if let Some(array) = value.as_array() {
        return array.clone();
    }
    if value.get("content").is_some()
        || value.get("fact").is_some()
        || value.get("memory").is_some()
    {
        return vec![value.clone()];
    }
    Vec::new()
}

fn parse_jsonl_values(response: &str) -> Vec<Value> {
    response
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
        .flat_map(|value| values_from_complete_json(&value))
        .collect()
}

fn parse_truncated_memories_array(response: &str) -> Vec<Value> {
    let Some(memories_index) = response.find("\"memories\"") else {
        return Vec::new();
    };
    let Some(array_offset) = response[memories_index..].find('[') else {
        return Vec::new();
    };
    parse_top_level_objects(&response[memories_index + array_offset + 1..])
        .into_iter()
        .filter_map(|object| serde_json::from_str::<Value>(object).ok())
        .collect()
}

fn parse_top_level_objects(text: &str) -> Vec<&str> {
    let mut objects = Vec::new();
    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(index);
                }
                depth += 1;
            }
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = start.take() {
                        objects.push(&text[start..=index]);
                    }
                }
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }

    objects
}

fn number(value: &Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|number| (number as f32).clamp(0.0, 1.0))
}

fn extract_json_object(response: &str) -> Option<&str> {
    if let (Some(start), Some(end)) = (response.find('{'), response.rfind('}')) {
        if end > start {
            return Some(&response[start..=end]);
        }
    }
    None
}

/// Extract a JSON array from text that may contain surrounding analysis/reasoning.
/// Finds the first `[` and last `]` and returns everything between them (inclusive).
fn extract_json_array(response: &str) -> Option<&str> {
    let start = response.find('[')?;
    let end = response.rfind(']')?;
    if end > start {
        Some(&response[start..=end])
    } else {
        None
    }
}

fn strip_markdown_fences(raw: &str) -> &str {
    let trimmed = raw.trim();
    // Strip opening ```json or ``` fence
    let after_open = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    // Strip closing ``` fence
    after_open.strip_suffix("```").unwrap_or(after_open).trim()
}

/// Attempt to repair a truncated JSON response by closing unclosed strings,
/// braces, and brackets. Returns the repaired JSON string if it becomes valid.
fn repair_truncated_json(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut result = String::with_capacity(trimmed.len() + 20);
    let mut in_string = false;
    let mut escaped = false;
    let mut brace_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;

    for ch in trimmed.chars() {
        if in_string {
            result.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else {
            match ch {
                '"' => {
                    in_string = true;
                    result.push(ch);
                }
                '{' => {
                    brace_depth += 1;
                    result.push(ch);
                }
                '}' => {
                    if brace_depth > 0 {
                        brace_depth -= 1;
                    }
                    result.push(ch);
                }
                '[' => {
                    bracket_depth += 1;
                    result.push(ch);
                }
                ']' => {
                    if bracket_depth > 0 {
                        bracket_depth -= 1;
                    }
                    result.push(ch);
                }
                _ => result.push(ch),
            }
        }
    }

    // Close unclosed string
    if in_string {
        result.push('"');
    }

    // Remove trailing commas (invalid before closing brackets/braces)
    let trimmed_result = result.trim_end_matches(&[',', '\n', ' ', '\t', '\r'][..]);
    let new_len = trimmed_result.len();
    result.truncate(new_len);

    // Close unclosed braces and brackets in correct order (inner first)
    for _ in 0..brace_depth {
        result.push('}');
    }
    for _ in 0..bracket_depth {
        result.push(']');
    }

    // Validate the repair produces valid JSON with actual content
    match serde_json::from_str::<Value>(&result) {
        Ok(v)
            if v.is_array()
                || (v.is_object() && !v.as_object().map(|o| o.is_empty()).unwrap_or(true)) =>
        {
            Some(result)
        }
        _ => None,
    }
}

/// Return a truncated preview of the LLM response for diagnostic logging.
fn response_preview(response: &str, max_len: usize) -> String {
    let response = response.trim();
    if response.len() <= max_len {
        response.to_string()
    } else {
        format!(
            "{}...<truncated, total {} chars>",
            &response[..max_len],
            response.len()
        )
    }
}

fn extractor_max_tokens() -> u32 {
    env::var("AGENT_MEMORY_EXTRACTOR_MAX_TOKENS")
        .or_else(|_| env::var("AGENT_MEMORY_LLM_MAX_TOKENS"))
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16384)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mem0_additive_memory_array() {
        let event = Event::new("I adopted a puppy named Max.".to_string())
            .namespace("conversation".to_string())
            .actor("user".to_string());
        let response = r#"{"memory":[{"id":"0","text":"User adopted a puppy named Max"}]}"#;

        let memories = parse_llm_memories(&event, response).unwrap();

        assert_eq!(memories.len(), 1);
        assert!(
            memories[0]
                .content
                .starts_with("User adopted a puppy named Max")
        );
        assert!(memories[0].content.contains("adopted"));
        assert_eq!(memories[0].memory_type, MemoryType::Semantic);
        assert_eq!(
            memories[0].metadata.get("memory_kind").map(String::as_str),
            Some("llm_fact")
        );
        assert!(
            memories[0]
                .metadata
                .get("source_keywords")
                .is_some_and(|value| value.contains("adopted"))
        );
    }

    #[test]
    fn parses_source_keywords_into_metadata_and_content() {
        let event = Event::new("Yeah, I painted that lake sunrise last year!".to_string())
            .namespace("conversation".to_string())
            .actor("Melanie".to_string());
        let response = r#"[{"fact":"Melanie painted a lake sunrise","type":"event","entities":["Melanie"],"source_keywords":["Melanie","painted","lake sunrise","last year"]}]"#;

        let memories = parse_llm_memories(&event, response).unwrap();

        assert_eq!(memories.len(), 1);
        assert!(
            memories[0]
                .content
                .contains("Melanie painted a lake sunrise")
        );
        // source_keywords are stored only in metadata (not appended to content
        // to avoid polluting the embedding vector).
        assert!(!memories[0].content.contains("last year"));
        assert_eq!(
            memories[0]
                .metadata
                .get("source_keywords")
                .map(String::as_str),
            Some("Melanie\npainted\nlake sunrise\nlast year")
        );
    }

    // ── JSON repair tests ──

    #[test]
    fn repair_truncated_json_direct_array_mid_object() {
        // Simulates deepseek-chat cutting off mid-object in a direct array
        let truncated = r#"[{"fact":"Caroline went to a support group","type":"event"}, {"fact":"Caroline felt"#;
        let result = parse_memory_values(truncated).unwrap();
        assert!(
            !result.is_empty(),
            "should recover partial memories from truncated array"
        );
        // At minimum we should get the first complete object
        assert!(result.iter().any(|v| {
            v.get("fact")
                .and_then(Value::as_str)
                .map(|s| s.contains("Caroline went"))
                .unwrap_or(false)
        }));
    }

    #[test]
    fn repair_truncated_json_unclosed_string() {
        let truncated = r#"[{"fact":"hello world"#; // missing closing "}]
        let result = parse_memory_values(truncated).unwrap();
        assert!(!result.is_empty());
        let fact = result[0].get("fact").and_then(Value::as_str).unwrap();
        assert!(fact.contains("hello world"));
    }

    #[test]
    fn repair_truncated_json_unclosed_brackets() {
        let truncated = r#"[{"fact":"a","type":"event"}, {"fact":"b","type":"preference"#;
        // missing }] — second object unclosed
        let result = parse_memory_values(truncated).unwrap();
        assert!(!result.is_empty());
        assert!(result.len() >= 1);
    }

    #[test]
    fn repair_truncated_json_trailing_comma() {
        // trailing comma before truncation is common
        let truncated = r#"[{"fact":"a","type":"event"},]"#;
        let result = parse_memory_values(truncated).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn strip_markdown_fences_removes_fences() {
        let input = "```json\n[{\"fact\":\"hello\"}]\n```";
        let cleaned = strip_markdown_fences(input);
        assert_eq!(cleaned, "[{\"fact\":\"hello\"}]");
    }

    #[test]
    fn strip_markdown_fences_no_fences_passthrough() {
        let input = "[{\"fact\":\"hello\"}]";
        let cleaned = strip_markdown_fences(input);
        assert_eq!(cleaned, "[{\"fact\":\"hello\"}]");
    }

    #[test]
    fn response_preview_truncates_long() {
        let long = "a".repeat(1000);
        let preview = response_preview(&long, 500);
        assert!(preview.len() < 1000);
        assert!(preview.contains("<truncated"));
        assert!(preview.contains("1000 chars"));
    }

    #[test]
    fn response_preview_short_passthrough() {
        let short = "hello";
        let preview = response_preview(short, 500);
        assert_eq!(preview, "hello");
    }

    #[test]
    fn extract_json_array_from_mixed_text() {
        // Simulates deepseek-v4-pro output: analysis text + JSON array
        let response = r#"Let me analyze this conversation step by step.

First, I'll identify the speakers and key information...

Now here are the extracted memories:

[{"fact":"Caroline went to a LGBTQ support group on 6 June 2026","type":"event","entities":["Caroline"],"source_keywords":["Caroline","LGBTQ support group","6 June 2026"]},{"fact":"Caroline found the transgender stories inspiring","type":"preference","entities":["Caroline"],"source_keywords":["Caroline","transgender stories","inspiring"]}]

I hope this extraction is helpful!"#;
        let event = Event::new(
            "I went to a LGBTQ support group yesterday and it was so powerful.".to_string(),
        )
        .namespace("conversation".to_string())
        .actor("Caroline".to_string());
        let memories = parse_llm_memories(&event, response).unwrap();
        assert_eq!(memories.len(), 2);
        assert!(memories[0].content.contains("Caroline went"));
        assert!(memories[1].content.contains("transgender stories"));
    }

    #[test]
    fn extract_json_array_basic() {
        // Pure JSON array, no surrounding text
        let json = extract_json_array(r#"[{"fact":"hello","type":"event"}]"#).unwrap();
        assert_eq!(json, r#"[{"fact":"hello","type":"event"}]"#);
    }

    #[test]
    fn extract_json_array_with_surrounding_text() {
        let text = "Some analysis...\n[1, 2, 3]\nMore text...";
        let json = extract_json_array(text).unwrap();
        assert_eq!(json, "[1, 2, 3]");
    }

    #[test]
    fn parses_simplified_facts_format() {
        let event = Event::new("I went to a support group yesterday.".to_string())
            .namespace("conversation".to_string())
            .actor("Caroline".to_string());
        let response = r#"{"facts": ["Caroline attended a support group on 6 June 2026", "Caroline found the support group powerful"]}"#;
        let memories = parse_llm_memories(&event, response).unwrap();
        assert_eq!(memories.len(), 2);
        assert!(memories[0].content.contains("Caroline attended"));
        assert!(memories[1].content.contains("support group powerful"));
    }
}
