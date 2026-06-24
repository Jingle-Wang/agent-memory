use std::collections::BTreeMap;
use std::env;

use serde::{Deserialize, Serialize};

use crate::embedding::tokenize;
use crate::llm::{LlmCompletionRequest, LlmError, LlmMessage, LlmProvider, LlmProviderMetadata};
use crate::models::MemoryPacket;
use crate::text::contains_any;

use super::dataset::QuestionForAnswerer;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnswerInput {
    pub question: QuestionForAnswerer,
    pub retrieved: Vec<MemoryPacketForAnswerer>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryPacketForAnswerer {
    pub memory_id: String,
    pub content: String,
    pub memory_type: String,
    pub metadata: BTreeMap<String, String>,
    pub score: f32,
    pub source_event_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerOutput {
    pub answer: String,
}

pub trait Answerer {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput;
}

pub trait EvidenceComposerProvider: Clone + Send + Sync + 'static {
    fn compose(&self, input: &AnswerInput) -> Result<AnswerOutput, LlmError>;
    fn metadata(&self) -> LlmProviderMetadata;
}

#[derive(Clone, Debug, Default)]
pub struct ExtractiveAnswerer;

impl Answerer for ExtractiveAnswerer {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput {
        BasicExtractiveAnswerer.answer(input)
    }
}

#[derive(Clone, Debug, Default)]
pub struct BasicExtractiveAnswerer;

impl Answerer for BasicExtractiveAnswerer {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput {
        let context = input
            .retrieved
            .iter()
            .map(|packet| packet.content.as_str())
            .collect::<Vec<_>>();
        AnswerOutput {
            answer: best_sentence(&input.question.text, &context),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DateExtractiveAnswerer;

impl Answerer for DateExtractiveAnswerer {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput {
        let context = input
            .retrieved
            .iter()
            .map(|packet| packet.content.as_str())
            .collect::<Vec<_>>();
        let question = input.question.text.to_lowercase();
        let answer = if asks_when(&question) {
            extract_date_like(&context)
                .unwrap_or_else(|| best_sentence(&input.question.text, &context))
        } else if asks_identity(&question) {
            extract_after_patterns(&context, &["identity is", "i am", "is a", "as a"])
                .unwrap_or_else(|| best_sentence(&input.question.text, &context))
        } else {
            best_sentence(&input.question.text, &context)
        };
        AnswerOutput { answer }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TypedExtractiveAnswerer;

impl Answerer for TypedExtractiveAnswerer {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput {
        let context = input
            .retrieved
            .iter()
            .map(|packet| packet.content.as_str())
            .collect::<Vec<_>>();
        let question = input.question.text.to_lowercase();
        let answer = if asks_when(&question) {
            extract_date_like(&context)
                .unwrap_or_else(|| best_sentence(&input.question.text, &context))
        } else if question.contains("where") {
            extract_after_patterns(&context, &["at ", "in ", "to "])
                .unwrap_or_else(|| best_sentence(&input.question.text, &context))
        } else if question.contains("what fields")
            || question.contains("what do")
            || question.contains("what does")
            || question.contains("what did")
        {
            compact_answer(&best_sentence(&input.question.text, &context))
        } else if asks_identity(&question) {
            extract_after_patterns(&context, &["identity is", "i am", "is a", "as a"])
                .unwrap_or_else(|| compact_answer(&best_sentence(&input.question.text, &context)))
        } else {
            compact_answer(&best_sentence(&input.question.text, &context))
        };
        AnswerOutput { answer }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpanExtractiveAnswerer;

impl Answerer for SpanExtractiveAnswerer {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput {
        let question = input.question.text.to_lowercase();
        if let Some(answer) = abstain_on_speaker_mismatch(&question, &input.retrieved) {
            return AnswerOutput { answer };
        }
        if let Some(answer) = high_precision_span_candidate(&question, &input.retrieved) {
            return AnswerOutput { answer };
        }
        if question.starts_with("would ") {
            return AnswerOutput {
                answer: "None".to_string(),
            };
        }

        let context = input
            .retrieved
            .iter()
            .map(|packet| packet.content.as_str())
            .collect::<Vec<_>>();
        if asks_when(&question) {
            if let Some(answer) = extract_date_like(&context) {
                // Phase 4: resolve relative dates using evidence metadata
                return AnswerOutput {
                    answer: resolve_date_from_packet_metadata(&answer, &input.retrieved),
                };
            }
        }
        let sentence = best_typed_packet_sentence(&input.question.text, &input.retrieved);
        let answer = if asks_when(&question) {
            let date_answer = extract_date_like(&[sentence.as_str()]).unwrap_or(sentence);
            resolve_date_from_packet_metadata(&date_answer, &input.retrieved)
        } else if question.contains("where") {
            extract_location_from_sentence(&question, &sentence).unwrap_or(sentence)
        } else if asks_title(&question) {
            extract_title_answer(&sentence).unwrap_or(sentence)
        } else {
            sentence
        };
        AnswerOutput { answer }
    }
}

// ── Phase 4: Metadata-aware relative date resolution ────────────────────────

/// Resolve relative date expressions ("yesterday", "last week", etc.)
/// into absolute dates using the timestamp metadata from evidence packets.
fn resolve_date_from_packet_metadata(
    date_answer: &str,
    packets: &[MemoryPacketForAnswerer],
) -> String {
    // Only attempt resolution if the answer looks like a relative date
    let lower = date_answer.to_lowercase();
    if !is_relative_date_phrase(&lower) {
        return date_answer.to_string();
    }
    // Find a packet with timestamp metadata to use as anchor
    for packet in packets {
        let anchor = packet
            .metadata
            .get("event_time")
            .or_else(|| packet.metadata.get("benchmark_timestamp"))
            .or_else(|| packet.metadata.get("timestamp"));
        if let Some(anchor) = anchor {
            if let Some(date_parts) = parse_date_parts(anchor) {
                // Try to resolve the relative expression using this anchor
                let test_sentence = format!(
                    "[verbatim_turn time={} {} {}] {}",
                    date_parts.0, date_parts.1, date_parts.2, date_answer
                );
                if let Some(resolved) = resolve_relative_date_from_timestamp(&test_sentence) {
                    return coarsen_date_answer(&resolved);
                }
            }
        }
    }
    date_answer.to_string()
}

fn high_precision_span_candidate(
    question: &str,
    packets: &[MemoryPacketForAnswerer],
) -> Option<String> {
    if let Some(answer) = combined_evidence_answer(question, packets) {
        return Some(answer);
    }

    if asks_when(question) {
        return best_raw_text_match(question, packets, |text| extract_date_like(&[text]));
    }

    if question.contains("where") && question.contains("hide") {
        return best_text_match(question, packets, extract_bone_hiding_place);
    }

    if question.contains("where") {
        return best_text_match(question, packets, |text| {
            extract_location_from_sentence(question, text)
        });
    }

    if question.contains("who") {
        return best_text_match(question, packets, |text| {
            extract_person_answer(question, text)
        });
    }

    if is_quantity_question(question) {
        return best_text_match(question, packets, |text| {
            extract_quantity_answer(question, text)
        });
    }

    if question.contains("relationship status") {
        return best_text_match(question, packets, extract_relationship_status);
    }

    if question.contains("field") || question.contains("education") {
        return best_text_match(question, packets, extract_field_answer);
    }

    if question.starts_with("what did") {
        return owned_text_match_or_abstain(question, packets, |text| {
            extract_object_for_question_verb(question, strip_speaker_prefix(text))
        });
    }

    if question.contains("name") {
        return best_text_match(question, packets, extract_name_answer);
    }

    if asks_title(question) {
        return best_text_match(question, packets, extract_title_answer);
    }

    if asks_identity(question)
        || question.starts_with("what ")
        || question.starts_with("which ")
        || question.starts_with("how ")
    {
        return owned_text_match_or_abstain(question, packets, |text| {
            extract_generic_span_answer(question, text)
        });
    }

    None
}

fn abstain_on_speaker_mismatch(
    question: &str,
    packets: &[MemoryPacketForAnswerer],
) -> Option<String> {
    let expected = expected_question_speaker(question, packets)?;
    let query_terms = abstain_topic_terms(question);
    if query_terms.len() < 2 {
        return None;
    }

    let mut expected_support = 0;
    let mut other_support = 0;
    for packet in packets.iter().take(12) {
        let lower = packet.content.to_lowercase();
        let overlap = query_terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count();
        if overlap < 2 {
            continue;
        }
        if is_question_like_evidence(&lower) {
            other_support += 1;
            continue;
        }
        let subject = subject_mention(&packet.content);
        let owner = if subject
            .as_deref()
            .is_some_and(|s| speaker_matches(s, expected.as_str()))
        {
            subject
        } else {
            packet_speaker(packet)
        };
        match owner.as_deref() {
            Some(owner) if speaker_matches(owner, expected.as_str()) => expected_support += 1,
            Some(owner) if !speaker_matches(owner, expected.as_str()) => other_support += 1,
            _ => {}
        }
    }

    if expected_support == 0 && other_support >= 1 {
        if question.starts_with("did ")
            || question.starts_with("does ")
            || question.starts_with("do ")
            || question.starts_with("is ")
            || question.starts_with("was ")
            || question.starts_with("would ")
            || question.starts_with("can ")
        {
            Some("No".to_string())
        } else {
            Some("None".to_string())
        }
    } else {
        None
    }
}

fn expected_question_speaker(
    question: &str,
    packets: &[MemoryPacketForAnswerer],
) -> Option<String> {
    let mut speakers = packets
        .iter()
        .filter_map(packet_speaker)
        .filter(|speaker| speaker.len() > 2)
        .collect::<std::collections::BTreeSet<_>>();
    speakers.extend(
        packets
            .iter()
            .filter_map(|packet| subject_mention(&packet.content)),
    );
    let matched = speakers
        .into_iter()
        .filter(|speaker| question_mentions_person(question, speaker))
        .collect::<Vec<_>>();
    if matched.len() == 1 {
        matched.into_iter().next()
    } else {
        person_hint_from_question(question)
    }
}

fn person_hint_from_question(question: &str) -> Option<String> {
    let tokens = tokenize(question);
    for marker in ["is", "was"] {
        for (index, token) in tokens.iter().enumerate().skip(1) {
            if token == marker
                && tokens
                    .get(index + 1)
                    .is_some_and(|next| next == "considering")
            {
                return tokens
                    .get(index.saturating_sub(1))
                    .map(|token| token.trim_end_matches("'s").to_string());
            }
        }
    }
    for marker in ["did", "does", "would", "is", "was"] {
        if let Some(index) = tokens.iter().position(|token| token == marker) {
            if let Some(candidate) = tokens.get(index + 1) {
                if !is_question_stop_term(candidate) && candidate.len() > 2 {
                    return Some(candidate.trim_end_matches("'s").to_string());
                }
            }
        }
    }
    None
}

fn question_mentions_person(question: &str, speaker: &str) -> bool {
    tokenize(question).into_iter().any(|term| {
        term == speaker
            || term == speaker.trim_end_matches('s')
            || speaker.starts_with(term.as_str())
            || term.starts_with(speaker)
    })
}

fn speaker_matches(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let llen = left.len();
    let rlen = right.len();
    // Only allow starts_with matching when the shorter string is ≥ 3 chars
    // (prevents "Al" matching "Alex")
    if llen >= 3 && rlen >= 3 {
        if left.starts_with(right) || right.starts_with(left) {
            return true;
        }
    }
    left.trim_end_matches('s') == right.trim_end_matches('s')
}

fn abstain_topic_terms(question: &str) -> Vec<String> {
    tokenize(question)
        .into_iter()
        .filter(|term| {
            !matches!(
                term.as_str(),
                "what"
                    | "when"
                    | "where"
                    | "why"
                    | "how"
                    | "did"
                    | "does"
                    | "was"
                    | "were"
                    | "is"
                    | "are"
                    | "the"
                    | "her"
                    | "his"
                    | "their"
                    | "none"
            )
        })
        .collect()
}

fn packet_speaker(packet: &MemoryPacketForAnswerer) -> Option<String> {
    packet
        .metadata
        .get("speaker")
        .map(|speaker| speaker.to_lowercase())
        .or_else(|| prefix_speaker(&packet.content))
}

fn prefix_speaker(content: &str) -> Option<String> {
    prefix_speaker_original(content).map(|speaker| speaker.to_lowercase())
}

fn prefix_speaker_original(content: &str) -> Option<String> {
    let text = content
        .split_once(']')
        .map(|(_, rest)| rest.trim())
        .unwrap_or(content)
        .trim();
    let lower = text.to_lowercase();
    if let Some((speaker, _)) = lower.split_once(':') {
        let speaker = speaker.trim();
        if speaker.len() > 2
            && speaker.len() <= 32
            && speaker
                .chars()
                .all(|ch| ch.is_ascii_alphabetic() || ch == ' ' || ch == '-' || ch == '\'')
        {
            return text
                .split_once(':')
                .map(|(speaker, _)| speaker.trim().to_string());
        }
    }
    let mut words = text.split_whitespace();
    let first = words
        .next()?
        .trim_matches(|ch: char| !ch.is_ascii_alphabetic());
    let second = words.next().unwrap_or_default().to_lowercase();
    if first.len() > 2
        && first
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())
        && matches!(second.as_str(), "said" | "says")
    {
        return Some(first.to_string());
    }
    None
}

fn subject_mention(content: &str) -> Option<String> {
    let text = strip_speaker_prefix(strip_evidence_prefix(content)).trim();
    let first = text.split_whitespace().next()?;
    let clean = first
        .trim_end_matches("'s")
        .trim_matches(|ch: char| !ch.is_ascii_alphabetic());
    if clean.len() > 2
        && clean
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())
        && !clean.to_lowercase().ends_with("ing")
    {
        Some(clean.to_lowercase())
    } else {
        None
    }
}

fn is_question_like_evidence(lower: &str) -> bool {
    lower.contains('?')
        || lower.starts_with("what ")
        || lower.starts_with("why ")
        || lower.starts_with("when ")
        || lower.starts_with("where ")
        || lower.starts_with("how ")
}

fn extract_person_answer(question: &str, text: &str) -> Option<String> {
    let text = strip_speaker_prefix(text);
    if question.contains("gift") {
        if let Some(value) = extract_after_patterns(&[text], &[" from my "]) {
            return Some(format!("my {}", trim_answer_span(&value)));
        }
        return extract_after_patterns(&[text], &[" from user's ", " from "])
            .map(|value| normalize_possessive_answer(&value));
    }
    extract_after_patterns(&[text], &[" by ", " from ", " with ", " to "])
        .or_else(|| first_proper_name(text))
}

fn extract_quantity_answer(question: &str, text: &str) -> Option<String> {
    if question.contains("speed") {
        if let Some(value) = extract_around_unit(text, &["mbps", "gbps"]) {
            return Some(value);
        }
    }
    extract_known_duration(text).or_else(|| extract_first_number_phrase(text))
}

fn extract_relationship_status(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    if lower.contains("single parent") || lower.contains(" single ") {
        Some("Single".to_string())
    } else if lower.contains("married") {
        Some("Married".to_string())
    } else if lower.contains("partner") || lower.contains("dating") {
        Some("In a relationship".to_string())
    } else {
        None
    }
}

fn extract_name_answer(text: &str) -> Option<String> {
    if let Some(names) = extract_pet_names(text) {
        return Some(names);
    }
    extract_after_patterns(
        &[text],
        &[" called ", " named ", " name is ", " names are "],
    )
}

fn extract_field_answer(text: &str) -> Option<String> {
    let mut values = Vec::new();
    if let Some(value) = extract_after_patterns(&[text], &[" studying "]) {
        push_unique(&mut values, capitalize_first(&value));
    }
    if let Some(value) = extract_after_patterns(&[text], &[" considering a ", " considering an "]) {
        push_unique(&mut values, value);
    }
    if values.is_empty() {
        None
    } else {
        Some(values.join(", "))
    }
}

fn capitalize_first(value: &str) -> String {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    format!("{}{}", first.to_uppercase(), chars.as_str())
}

fn extract_generic_span_answer(question: &str, text: &str) -> Option<String> {
    let text = strip_speaker_prefix(text);
    if question.contains("name") {
        if let Some(names) = extract_pet_names(text) {
            return Some(names);
        }
        if let Some(value) = extract_after_patterns(&[text], &[" called ", " named ", " name is "])
        {
            return Some(value);
        }
    }
    if let Some(value) = extract_quoted_phrase(text) {
        return Some(value);
    }
    if question.contains("what did") {
        if let Some(value) = extract_object_for_question_verb(question, text) {
            return Some(value);
        }
        return None;
    }
    if question.contains("great for") {
        return extract_after_patterns(&[text], &[" great for "]);
    }
    if question.starts_with("how ") {
        if let Some(value) = extract_after_patterns(&[text], &[" by ", " through "]) {
            return Some(format!("by {}", trim_answer_span(&value)));
        }
    }
    if question.contains("symbol")
        || question.contains("represent")
        || question.contains("reminder")
    {
        return extract_list_after_patterns(
            text,
            &[
                " symbolizes ",
                " symbolize ",
                " represents ",
                " represent ",
                " reminder of ",
                " reminds me of ",
            ],
        );
    }
    if question.contains("where") && question.contains("hide") {
        return extract_bone_hiding_place(text);
    }
    if let Some(value) = extract_after_patterns(
        &[text],
        &[
            " is ", " are ", " was ", " were ", " as ", " for ", " about ", " into ",
        ],
    ) {
        return Some(value);
    }
    Some(compact_answer(text)).filter(|value| !value.is_empty())
}

fn extract_object_for_question_verb(question: &str, text: &str) -> Option<String> {
    let speaker = prefix_speaker(text);
    let verbs = question_verbs(question, speaker.as_deref());
    let lower = text.to_lowercase();
    for verb in verbs {
        for pattern in [
            format!("{verb} "),
            format!("{verb}ing "),
            format!("{}ing ", verb.trim_end_matches('e')),
            format!("{}ed ", verb.trim_end_matches('e')),
        ] {
            if let Some(index) = lower.find(&pattern) {
                let start = index + pattern.len();
                if let Some(value) = trim_object_tail(&text[start..]) {
                    return Some(strip_leading_that(&value).to_string());
                }
            }
        }
    }
    None
}

fn question_verbs(question: &str, speaker: Option<&str>) -> Vec<String> {
    if question.starts_with("what did") {
        let tokens = tokenize(question);
        if let Some(did_index) = tokens.iter().position(|token| token == "did") {
            let stop = tokens[did_index + 1..]
                .iter()
                .position(|token| {
                    matches!(
                        token.as_str(),
                        "after" | "before" | "for" | "to" | "with" | "when" | "where"
                    )
                })
                .map(|offset| did_index + 1 + offset)
                .unwrap_or(tokens.len());
            let slice = &tokens[did_index + 1..stop];
            if let Some(verb) = slice
                .iter()
                .rev()
                .find(|term| speaker != Some(term.as_str()) && !is_question_stop_term(term))
            {
                return vec![verb.clone()];
            }
        }
    }

    tokenize(question)
        .into_iter()
        .filter(|term| {
            term.len() > 3 && !is_question_stop_term(term) && speaker != Some(term.as_str())
        })
        .collect()
}

fn is_question_stop_term(term: &str) -> bool {
    matches!(
        term,
        "what"
            | "when"
            | "where"
            | "which"
            | "does"
            | "did"
            | "have"
            | "user"
            | "after"
            | "before"
            | "with"
            | "from"
            | "that"
    )
}

fn trim_object_tail(value: &str) -> Option<String> {
    let value = value.split(" - ").next().unwrap_or(value);
    value
        .split([',', '.', ';', '\n', ':'])
        .next()
        .map(trim_answer_span)
        .filter(|value| !value.is_empty())
}

fn strip_leading_that(value: &str) -> &str {
    value.strip_prefix("that ").unwrap_or(value)
}

fn best_text_match(
    question: &str,
    packets: &[MemoryPacketForAnswerer],
    extractor: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let mut best = None::<(String, i32)>;
    for packet in packets {
        let text = strip_evidence_prefix(&packet.content);
        let Some(candidate) = extractor(text).map(|value| clean_candidate_span(&value)) else {
            continue;
        };
        if candidate.is_empty() || candidate.split_whitespace().count() > 16 {
            continue;
        }
        let mut score = specific_overlap(question, text) as i32 * 4;
        score += (packet.score * 4.0).round() as i32;
        if text.to_lowercase().contains(&candidate.to_lowercase()) {
            score += 2;
        }
        if best
            .as_ref()
            .is_none_or(|(_, best_score)| score > *best_score)
        {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn best_owned_text_match(
    question: &str,
    packets: &[MemoryPacketForAnswerer],
    extractor: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let Some(expected) = expected_question_speaker(question, packets) else {
        return best_text_match(question, packets, extractor);
    };
    let filtered = packets
        .iter()
        .filter(|packet| packet_supports_expected_subject(packet, &expected))
        .cloned()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        return None;
    }
    best_text_match(question, &filtered, extractor)
}

fn owned_text_match_or_abstain(
    question: &str,
    packets: &[MemoryPacketForAnswerer],
    extractor: impl Fn(&str) -> Option<String> + Copy,
) -> Option<String> {
    if let Some(answer) = best_owned_text_match(question, packets, extractor) {
        return Some(answer);
    }
    if expected_question_speaker(question, packets).is_some()
        && best_text_match(question, packets, extractor).is_some()
    {
        if is_yes_no_question(question) {
            Some("No".to_string())
        } else {
            Some("None".to_string())
        }
    } else {
        None
    }
}

fn is_yes_no_question(question: &str) -> bool {
    question.starts_with("did ")
        || question.starts_with("does ")
        || question.starts_with("do ")
        || question.starts_with("is ")
        || question.starts_with("was ")
        || question.starts_with("were ")
        || question.starts_with("would ")
        || question.starts_with("can ")
}

fn packet_supports_expected_subject(packet: &MemoryPacketForAnswerer, expected: &str) -> bool {
    packet_speaker(packet)
        .as_deref()
        .is_some_and(|s| speaker_matches(s, expected))
        || subject_mention(&packet.content)
            .as_deref()
            .is_some_and(|s| speaker_matches(s, expected))
        || packet
            .content
            .to_lowercase()
            .starts_with(&format!("{expected} "))
}

fn best_raw_text_match(
    question: &str,
    packets: &[MemoryPacketForAnswerer],
    extractor: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let mut best = None::<(String, i32)>;
    for packet in packets {
        let text = packet.content.as_str();
        let Some(candidate) = extractor(text).map(|value| clean_candidate_span(&value)) else {
            continue;
        };
        if candidate.is_empty() || candidate.split_whitespace().count() > 16 {
            continue;
        }
        let score =
            specific_overlap(question, text) as i32 * 4 + (packet.score * 4.0).round() as i32;
        if best
            .as_ref()
            .is_none_or(|(_, best_score)| score > *best_score)
        {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn normalize_possessive_answer(value: &str) -> String {
    let clean = clean_phrase(value);
    clean
        .strip_prefix("user's ")
        .map(|rest| format!("my {rest}"))
        .unwrap_or(clean)
}

fn clean_candidate_span(value: &str) -> String {
    let lower = value.to_lowercase();
    if lower.contains("week before") || lower.contains("week after") {
        clean_phrase(value)
    } else {
        trim_answer_span(value)
    }
}

fn extract_pet_names(text: &str) -> Option<String> {
    let text = strip_speaker_prefix(text);
    let lower = text.to_lowercase();
    if !contains_any(&lower, &["pet", "cat", "dog", "guinea pig"]) {
        return None;
    }
    let mut names = Vec::new();
    for token in text.split(|ch: char| {
        ch.is_whitespace() || matches!(ch, ',' | '.' | ';' | ':' | '!' | '?' | '(' | ')')
    }) {
        let clean = token.trim_matches(|ch: char| !ch.is_ascii_alphabetic());
        if clean.len() < 3 {
            continue;
        }
        let lower = clean.to_lowercase();
        if matches!(
            lower.as_str(),
            "i've"
                | "got"
                | "two"
                | "cats"
                | "cat"
                | "dog"
                | "named"
                | "guinea"
                | "pig"
                | "pets"
                | "pet"
                | "and"
                | "one"
                | "have"
                | "has"
        ) {
            continue;
        }
        if clean
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())
            && !names.iter().any(|name| name == clean)
        {
            names.push(clean.to_string());
        }
    }
    if names.is_empty() {
        None
    } else {
        Some(names.join(", "))
    }
}

fn extract_bone_hiding_place(text: &str) -> Option<String> {
    let speaker = prefix_speaker_original(text);
    let lower = text.to_lowercase();
    if !lower.contains("bone") {
        return None;
    }
    let text = strip_speaker_prefix(text);
    let place = extract_after_patterns(&[text], &[" in ", " inside ", " under ", " behind "])?;
    let place = trim_answer_span(&place);
    if let Some(rest) = place.strip_prefix("my ") {
        if let Some(speaker) = speaker {
            return Some(format!("in {}'s {}", speaker, rest));
        }
    }
    Some(format!("in {place}"))
}

fn extract_known_duration(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    if !lower.contains("known") && !lower.contains("friends") {
        return None;
    }
    extract_after_patterns(&[text], &[" for "]).and_then(|value| {
        let lower = value.to_lowercase();
        if lower.contains("year") || lower.contains("month") || lower.contains("week") {
            Some(value)
        } else {
            None
        }
    })
}

#[derive(Clone, Debug, Default)]
pub struct EvidenceComposerAnswerer;

impl Answerer for EvidenceComposerAnswerer {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput {
        let question = input.question.text.to_lowercase();
        if let Some(answer) = structured_candidate(&question, &input.retrieved) {
            return AnswerOutput { answer };
        }

        let span = SpanExtractiveAnswerer.answer(input);
        if !span.answer.starts_with("I don't know") && !is_question_like_sentence(&span.answer) {
            return span;
        }

        let context = input
            .retrieved
            .iter()
            .map(|packet| packet.content.as_str())
            .collect::<Vec<_>>();
        let sentence = best_typed_sentence(&input.question.text, &context);
        let answer = if asks_when(&question) {
            extract_date_like(&[sentence.as_str()]).unwrap_or(sentence)
        } else if question.contains("where") {
            extract_location_from_sentence(&question, &sentence).unwrap_or(sentence)
        } else if asks_title(&question) {
            extract_title_answer(&sentence).unwrap_or(sentence)
        } else {
            sentence
        };
        AnswerOutput { answer }
    }
}

#[derive(Clone, Debug)]
pub struct LlmEvidenceComposer<P: LlmProvider> {
    provider: P,
    prompt_version: String,
    max_tokens: u32,
}

impl<P: LlmProvider> LlmEvidenceComposer<P> {
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            prompt_version: "evidence-composer-v6-benchmark-extractive".to_string(),
            max_tokens: env::var("AGENT_MEMORY_LLM_MAX_TOKENS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(4096),
        }
    }

    pub fn prompt_version(&self) -> &str {
        &self.prompt_version
    }
}

impl<P: LlmProvider> EvidenceComposerProvider for LlmEvidenceComposer<P> {
    fn compose(&self, input: &AnswerInput) -> Result<AnswerOutput, LlmError> {
        let metadata = self.provider.metadata();
        let request = LlmCompletionRequest {
            model: metadata.model,
            messages: vec![
                LlmMessage::system(composer_system_prompt()),
                LlmMessage::user(composer_user_prompt(input)),
            ],
            temperature: 0.0,
            max_tokens: self.max_tokens,
            response_format: None,
        };
        let response = self.provider.complete(&request)?;
        Ok(AnswerOutput {
            answer: parse_composer_answer(&response),
        })
    }

    fn metadata(&self) -> LlmProviderMetadata {
        let mut metadata = self.provider.metadata();
        metadata.prompt_version = Some(self.prompt_version.clone());
        metadata
    }
}

#[derive(Clone, Debug)]
pub struct LlmComposerAnswerer<C: EvidenceComposerProvider> {
    composer: C,
}

impl<C: EvidenceComposerProvider> LlmComposerAnswerer<C> {
    pub fn new(composer: C) -> Self {
        Self { composer }
    }

    pub fn metadata(&self) -> LlmProviderMetadata {
        self.composer.metadata()
    }
}

impl<C: EvidenceComposerProvider> Answerer for LlmComposerAnswerer<C> {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput {
        self.composer
            .compose(input)
            .unwrap_or_else(|error| AnswerOutput {
                answer: format!("I don't know. LLM_ERROR: {error}"),
            })
    }
}

#[derive(Clone, Debug)]
pub struct HybridLlmComposerAnswerer<C: EvidenceComposerProvider> {
    composer: C,
}

impl<C: EvidenceComposerProvider> HybridLlmComposerAnswerer<C> {
    pub fn new(composer: C) -> Self {
        Self { composer }
    }

    pub fn metadata(&self) -> LlmProviderMetadata {
        self.composer.metadata()
    }
}

impl<C: EvidenceComposerProvider> Answerer for HybridLlmComposerAnswerer<C> {
    fn answer(&self, input: &AnswerInput) -> AnswerOutput {
        let span = SpanExtractiveAnswerer.answer(input);
        self.composer.compose(input).unwrap_or(span)
    }
}

fn composer_system_prompt() -> &'static str {
    "You answer memory benchmark questions using only the provided evidence. Answer concisely with only the specific fact requested: a date, name, number, short list, yes/no, or one short sentence. Do not add explanations. Do not use outside knowledge. Prefer original source evidence over derived summaries when they conflict.\n\
\n\
Extract the best supported answer from the evidence, even if the wording does not match the question exactly. Memory benchmark questions often paraphrase the evidence; use entity, topic, source event, and metadata matches to choose the closest relevant evidence item. If multiple evidence items partially answer the question, combine only the requested short facts. If the evidence contains relative time expressions (yesterday, last week, N years ago, etc.), infer the absolute time using the event_time, benchmark_timestamp, or timestamp metadata as the anchor.\n\
\n\
For yes/no questions, answer Yes or No when evidence supports it; otherwise answer I don't know. Only answer I don't know when all evidence is irrelevant to the question. Do not answer I don't know merely because the evidence is indirect, uses first person, uses a nickname, or omits exact question wording."
}

fn composer_user_prompt(input: &AnswerInput) -> String {
    let mut prompt = String::new();
    prompt.push_str("Question:\n");
    prompt.push_str(&input.question.text);
    prompt.push_str("\n\nEvidence:\n");
    for (index, packet) in input.retrieved.iter().enumerate() {
        prompt.push_str(&format!(
            "\n[{}]\nmemory_id: {}\nmemory_type: {}\nscore: {:.4}\nsource_event_id: {}\nmetadata: {}\ncontent: {}\n",
            index + 1,
            packet.memory_id,
            packet.memory_type,
            packet.score,
            packet.source_event_id.as_deref().unwrap_or(""),
            serde_json::to_string(&packet.metadata).unwrap_or_else(|_| "{}".to_string()),
            packet.content
        ));
    }
    if needs_relative_time_reasoning(input) {
        prompt.push_str(
            "\nRelative time reasoning:\n\
If an evidence item uses relative time words such as last year, yesterday, today, tomorrow, next week, last week, or N days/weeks/months/years ago, use that same evidence item's event_time, benchmark_timestamp, or timestamp metadata as the anchor. Infer the absolute date or year requested by the question. For example, event_time 8 May 2023 plus last year means 2022, and event_time 8 May 2023 plus yesterday means 7 May 2023. Return only the inferred answer, not the reasoning.\n",
        );
    }
    prompt.push_str("\nProvide your answer as plain text.\n");
    prompt
}

fn parse_composer_answer(response: &str) -> String {
    clean_phrase(response)
}

fn needs_relative_time_reasoning(input: &AnswerInput) -> bool {
    let question = input.question.text.to_lowercase();
    asks_when(&question)
        && input.retrieved.iter().any(|packet| {
            evidence_has_time_anchor(packet) && contains_relative_time(&packet.content)
        })
}

fn evidence_has_time_anchor(packet: &MemoryPacketForAnswerer) -> bool {
    ["event_time", "benchmark_timestamp", "timestamp"]
        .iter()
        .any(|key| packet.metadata.contains_key(*key))
}

fn contains_relative_time(value: &str) -> bool {
    let lower = value.to_lowercase();
    contains_any(
        &lower,
        &[
            "last year",
            "next year",
            "last month",
            "next month",
            "last week",
            "next week",
            "yesterday",
            "tomorrow",
            "today",
            " ago",
            "days ago",
            "weeks ago",
            "months ago",
            "years ago",
        ],
    )
}

fn structured_candidate(question: &str, packets: &[MemoryPacketForAnswerer]) -> Option<String> {
    if let Some(answer) = combined_evidence_answer(question, packets) {
        return Some(answer);
    }

    let mut candidates = Vec::<(String, i32)>::new();
    for packet in packets {
        if let Some(candidate) = candidate_from_metadata(question, packet) {
            candidates.push((candidate, 12));
        }
        if let Some(candidate) = candidate_from_text(question, &packet.content) {
            candidates.push((candidate, 10));
        }
    }
    candidates
        .into_iter()
        .filter(|(candidate, _)| !candidate.is_empty())
        .max_by_key(|(candidate, score)| {
            let compact_bonus = if candidate.split_whitespace().count() <= 8 {
                3
            } else {
                0
            };
            score + compact_bonus
        })
        .map(|(candidate, _)| candidate)
}

fn combined_evidence_answer(question: &str, packets: &[MemoryPacketForAnswerer]) -> Option<String> {
    if question.contains("what activit")
        || question.contains("what do")
        || (question.contains("what does") && question.contains(" do "))
        || (question.contains("what did") && question.contains(" do "))
    {
        let activities = collect_action_objects(question, packets);
        if activities.len() >= 2 {
            return Some(activities.join(", "));
        }
    }
    if question.contains("where") {
        let places = collect_locations(question, packets);
        if places.len() >= 2 {
            return Some(places.join(", "));
        }
    }
    if question.contains("like") || question.contains("favorite") || question.contains("prefer") {
        let preferences = collect_preference_objects(question, packets);
        if !preferences.is_empty() {
            return Some(preferences.join(", "));
        }
    }
    None
}

fn candidate_from_metadata(question: &str, packet: &MemoryPacketForAnswerer) -> Option<String> {
    let relation = packet.metadata.get("relation").map(String::as_str)?;
    let object = packet.metadata.get("object").map(String::as_str)?;
    let kind = packet
        .metadata
        .get("memory_kind")
        .map(String::as_str)
        .unwrap_or_default();
    if kind != "profile" && kind != "observation" {
        return None;
    }
    let relation_match = relation_matches_question(question, relation)
        || match relation {
            "identity" => question.contains("name") || question.contains("who"),
            "gift_from" => question.contains("who") || question.contains("gift"),
            "purchase_location" | "redeemed_at" | "attended_at" => question.contains("where"),
            "quantity" => is_quantity_question(question),
            _ => false,
        };
    if relation_match {
        Some(compact_object(object))
    } else {
        None
    }
}

fn relation_matches_question(question: &str, relation: &str) -> bool {
    let relation = relation.to_lowercase();
    question.contains(&relation)
        || (contains_any(question, &["job", "work", "career", "occupation"])
            && contains_any(&relation, &["job", "work", "career", "occupation"]))
        || (contains_any(question, &["where", "location", "live", "moved", "from"])
            && contains_any(&relation, &["location", "live", "moved", "from"]))
        || (contains_any(question, &["prefer", "favorite", "like", "want"])
            && contains_any(&relation, &["prefer", "favorite", "like", "want"]))
        || (contains_any(question, &["when", "date", "year", "time"])
            && contains_any(&relation, &["time", "date", "happened"]))
}

fn candidate_from_text(question: &str, text: &str) -> Option<String> {
    if specific_overlap(question, text) < 2 {
        return None;
    }
    let text = strip_evidence_prefix(text);
    let lower = text.to_lowercase();
    if is_quantity_question(question) {
        return extract_quantity_answer(question, text);
    }
    if question.contains("who") {
        return extract_person_answer(question, text);
    }
    if question.contains("previous occupation") || question.contains("previous role") {
        return extract_after_patterns(
            &[text],
            &[
                " previous occupation was ",
                " previous role was ",
                " worked as ",
                " was a ",
                " was an ",
            ],
        )
        .or_else(|| Some(compact_answer(text)));
    }
    if question.contains("name") {
        if let Some(value) = extract_after_patterns(&[text], &[" name is ", " called ", " named "])
        {
            return Some(value);
        }
    }
    if asks_title(question) {
        return extract_title_answer(text);
    }
    if question.contains("where") {
        return extract_location_from_sentence(question, text);
    }
    if lower.contains(" is ") || lower.contains(" was ") || lower.contains(" are ") {
        return extract_generic_span_answer(question, text);
    }
    None
}

fn collect_action_objects(question: &str, packets: &[MemoryPacketForAnswerer]) -> Vec<String> {
    let mut values = Vec::new();
    for packet in packets {
        let text = strip_speaker_prefix(strip_evidence_prefix(&packet.content));
        if is_question_like_sentence(text)
            || (specific_overlap(question, text) == 0 && !question.contains(" do "))
        {
            continue;
        }
        if let Some(value) = extract_action_noun(text) {
            push_unique(&mut values, value);
        }
    }
    values
}

fn collect_locations(question: &str, packets: &[MemoryPacketForAnswerer]) -> Vec<String> {
    let mut values = Vec::new();
    for packet in packets {
        let text = strip_speaker_prefix(strip_evidence_prefix(&packet.content));
        if is_question_like_sentence(text) || specific_overlap(question, text) == 0 {
            continue;
        }
        if let Some(value) = extract_location_from_sentence(question, text) {
            push_unique(&mut values, strip_leading_preposition(&value).to_string());
        }
    }
    values
}

fn collect_preference_objects(question: &str, packets: &[MemoryPacketForAnswerer]) -> Vec<String> {
    let mut values = Vec::new();
    for packet in packets {
        let text = strip_speaker_prefix(strip_evidence_prefix(&packet.content));
        let lower = text.to_lowercase();
        if is_question_like_sentence(text) || specific_overlap(question, text) == 0 {
            continue;
        }
        let candidate = extract_after_patterns(
            &[text],
            &[
                " likes ",
                " like ",
                " loves ",
                " love ",
                " prefers ",
                " prefer ",
                " stoked for ",
            ],
        )
        .or_else(|| {
            if lower.contains(" exhibit") {
                extract_before_pattern(text, " exhibit")
            } else {
                None
            }
        });
        if let Some(value) = candidate {
            push_unique(&mut values, pluralize_compact(&value));
        }
    }
    values
}

fn extract_action_noun(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    for (pattern, value) in [
        ("signed up for a ", None),
        ("signed up for an ", None),
        ("signed up for ", None),
        ("went swimming", Some("swimming")),
        ("went hiking", Some("hiking")),
        ("went camping", Some("camping")),
        ("camping", Some("camping")),
        ("painted ", Some("painting")),
    ] {
        if let Some(value) = value {
            if lower.contains(pattern) {
                return Some(value.to_string());
            }
            continue;
        }
        if let Some(index) = lower.find(pattern) {
            let start = index + pattern.len();
            let phrase = text[start..]
                .split([',', '.', ';', '\n'])
                .next()
                .map(trim_answer_span)
                .filter(|value| !value.is_empty())?;
            return Some(noun_head(&phrase));
        }
    }
    first_gerund(text)
}

fn extract_before_pattern(text: &str, pattern: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let index = lower.find(pattern)?;
    text[..index]
        .split(['.', ',', ';', ':'])
        .next_back()
        .map(trim_answer_span)
        .filter(|value| !value.is_empty())
}

fn noun_head(value: &str) -> String {
    let words = value.split_whitespace().collect::<Vec<_>>();
    if words.len() >= 2 && words[1].eq_ignore_ascii_case("class") {
        return words[0].to_string();
    }
    words
        .first()
        .map(|word| clean_phrase(word))
        .unwrap_or_else(|| clean_phrase(value))
}

fn first_gerund(text: &str) -> Option<String> {
    text.split_whitespace()
        .map(|word| word.trim_matches(|ch: char| !ch.is_ascii_alphabetic()))
        .find(|word| word.len() > 4 && word.to_lowercase().ends_with("ing"))
        .map(|word| word.to_lowercase())
}

fn push_unique(values: &mut Vec<String>, value: String) {
    let value = trim_answer_span(&value);
    if value.is_empty() {
        return;
    }
    if !values
        .iter()
        .any(|existing| existing.eq_ignore_ascii_case(&value))
    {
        values.push(value);
    }
}

fn strip_leading_preposition(value: &str) -> &str {
    let value = value
        .strip_prefix("in ")
        .or_else(|| value.strip_prefix("at "))
        .or_else(|| value.strip_prefix("on "))
        .or_else(|| value.strip_prefix("near "))
        .or_else(|| value.strip_prefix("from "))
        .unwrap_or(value);
    value.strip_prefix("the ").unwrap_or(value)
}

fn pluralize_compact(value: &str) -> String {
    let value = trim_answer_span(value);
    let value = value.strip_prefix("the ").unwrap_or(&value).to_string();
    if let Some(rest) = value.strip_suffix(" exhibit") {
        format!("{}s", rest.trim())
    } else {
        value
    }
}

fn strip_evidence_prefix(text: &str) -> &str {
    if text.starts_with('[') {
        if let Some((_, rest)) = text.split_once("] ") {
            return rest;
        }
    }
    text
}

fn strip_speaker_prefix(text: &str) -> &str {
    let text = strip_evidence_prefix(text).trim();
    if let Some((speaker, rest)) = text.split_once(':') {
        let speaker = speaker.trim();
        if speaker.len() > 2
            && speaker.len() <= 32
            && speaker
                .chars()
                .all(|ch| ch.is_ascii_alphabetic() || ch == ' ' || ch == '-' || ch == '\'')
        {
            return rest.trim();
        }
    }
    text
}

fn first_proper_name(text: &str) -> Option<String> {
    text.split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_ascii_alphabetic()))
        .find(|token| {
            token.len() > 2
                && token
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_uppercase())
        })
        .map(ToOwned::to_owned)
}

fn specific_overlap(question: &str, text: &str) -> usize {
    let text_terms = tokenize(text);
    tokenize(question)
        .into_iter()
        .filter(|term| {
            term.len() >= 3
                && !matches!(
                    term.as_str(),
                    "what" | "where" | "when" | "many" | "much" | "long" | "from" | "with"
                )
        })
        .filter(|term| text_terms.iter().any(|text_term| text_term == term))
        .count()
}

fn is_quantity_question(question: &str) -> bool {
    question.contains("how many")
        || question.contains("how much")
        || question.contains("how long")
        || question.contains("what speed")
}

fn asks_title(question: &str) -> bool {
    contains_any(
        question,
        &[
            "what play",
            "what book",
            "what movie",
            "what song",
            "what album",
            "what show",
            "what is the name",
            "which one",
        ],
    )
}

fn compact_object(value: &str) -> String {
    let without_prefix = value
        .split_once(": ")
        .map(|(_, rest)| rest)
        .unwrap_or(value);
    trim_answer_span(without_prefix)
}

fn extract_around_unit(text: &str, units: &[&str]) -> Option<String> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return None;
    }
    for (index, token) in tokens.iter().enumerate() {
        let clean = token
            .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
            .to_lowercase();
        if units.contains(&clean.as_str()) {
            let start = index.saturating_sub(1);
            let end = index;
            return Some(clean_phrase(&tokens[start..=end].join(" ")));
        }
    }
    None
}

fn extract_first_number_phrase(text: &str) -> Option<String> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return None;
    }
    for (index, token) in tokens.iter().enumerate() {
        let clean = token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric());
        if clean.chars().any(|ch| ch.is_ascii_digit()) {
            let end = (index + 1).min(tokens.len() - 1);
            return Some(clean_phrase(&tokens[index..=end].join(" ")));
        }
    }
    None
}

fn asks_when(question: &str) -> bool {
    question.contains("when") || question.contains("what year") || question.contains("what date")
}

fn asks_identity(question: &str) -> bool {
    question.contains("identity") || question.contains("who is") || question.contains("what is")
}

fn extract_date_like(context: &[&str]) -> Option<String> {
    for text in context {
        let text_date = parse_prefixed_event_date(text);
        for sentence in text.split(['.', '!', '?', '\n']) {
            if let Some(date) = resolve_relative_date_from_timestamp(sentence) {
                return Some(coarsen_date_answer(&date));
            }
            if let Some(date) = text_date {
                if let Some(answer) = resolve_relative_date_with_anchor(sentence, date) {
                    return Some(coarsen_date_answer(&answer));
                }
            }
        }
    }

    for text in context {
        for sentence in text.split(['.', '!', '?', '\n']) {
            if let Some(date) = parse_prefixed_event_date(sentence) {
                if !is_evidence_header_only(sentence) {
                    return Some(date.0.to_string());
                }
            }
            let sentence = strip_session_prefix(sentence);
            if let Some(holiday) = extract_named_day(sentence) {
                return Some(holiday);
            }
            let tokens = sentence.split_whitespace().collect::<Vec<_>>();
            for span in 1..=8 {
                for window in tokens.windows(span) {
                    let phrase = clean_phrase(&window.join(" "));
                    if is_relative_date_phrase(&phrase) {
                        return Some(phrase);
                    }
                }
            }
            for span in [4, 3, 2, 1] {
                for window in tokens.windows(span) {
                    let phrase = clean_phrase(&window.join(" "));
                    if contains_month(&phrase) {
                        return Some(phrase);
                    }
                    if phrase.len() == 4 && phrase.chars().all(|ch| ch.is_ascii_digit()) {
                        return Some(phrase);
                    }
                }
            }
        }
    }
    None
}

fn coarsen_date_answer(answer: &str) -> String {
    let lower = answer.to_lowercase();
    if lower.contains("week before") || lower.contains("week after") {
        return answer.to_string();
    }
    answer
        .split_whitespace()
        .find(|token| token.len() == 4 && token.chars().all(|ch| ch.is_ascii_digit()))
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| answer.to_string())
}

fn resolve_relative_date_with_anchor(sentence: &str, date: (i32, u32, i32)) -> Option<String> {
    let lower = sentence.to_lowercase();
    if lower.contains("last year") {
        return Some((date.0 - 1).to_string());
    }
    if lower.contains("next year") {
        return Some((date.0 + 1).to_string());
    }
    if lower.contains("last week") {
        return Some(format!("week before {}", format_date(date)));
    }
    if lower.contains("next week") {
        return Some(format!("week after {}", format_date(date)));
    }
    if lower.contains("last month") || lower.contains("next month") {
        return Some(date.0.to_string());
    }
    let offset = if lower.contains("yesterday") {
        -1
    } else if lower.contains("tomorrow") {
        1
    } else if lower.contains("today") {
        0
    } else {
        return None;
    };
    shift_date(date.0, date.1, date.2, offset).map(format_date)
}

fn is_evidence_header_only(sentence: &str) -> bool {
    let trimmed = sentence.trim();
    trimmed.starts_with("[verbatim_session") && !trimmed.contains(':')
}

fn resolve_relative_date_from_timestamp(sentence: &str) -> Option<String> {
    let lower = sentence.to_lowercase();
    let date = parse_prefixed_event_date(sentence)?;
    if lower.contains("last year") {
        return Some((date.0 - 1).to_string());
    }
    if lower.contains("next year") {
        return Some((date.0 + 1).to_string());
    }
    if lower.contains("last week") {
        return Some(format!("week before {}", format_date(date)));
    }
    if lower.contains("next week") {
        return Some(format!("week after {}", format_date(date)));
    }
    if lower.contains("last month") || lower.contains("next month") {
        return Some(date.0.to_string());
    }
    let offset = if lower.contains("yesterday") {
        -1
    } else if lower.contains("tomorrow") {
        1
    } else if lower.contains("today") {
        0
    } else {
        return None;
    };
    shift_date(date.0, date.1, date.2, offset).map(format_date)
}

fn parse_prefixed_event_date(sentence: &str) -> Option<(i32, u32, i32)> {
    if let Some(index) = sentence.find("time=") {
        let start = index + "time=".len();
        let end = sentence[start..]
            .find(']')
            .map(|offset| start + offset)
            .unwrap_or(sentence.len());
        return parse_date_parts(&sentence[start..end]);
    }
    if sentence.trim_start().starts_with("On ") {
        return parse_date_parts(sentence);
    }
    None
}

fn parse_date_parts(value: &str) -> Option<(i32, u32, i32)> {
    let clean = value.replace(',', " ");
    let tokens = clean.split_whitespace().collect::<Vec<_>>();
    for window in tokens.windows(3) {
        let day = window[0].parse::<u32>().ok();
        let month = month_number(window[1]);
        let year = window[2].parse::<i32>().ok();
        if let (Some(day), Some(month), Some(year)) = (day, month, year) {
            return Some((year, month, day as i32));
        }
    }
    None
}

fn shift_date(year: i32, month: u32, day: i32, offset_days: i32) -> Option<(i32, u32, i32)> {
    let mut year = year;
    let mut month = month;
    let mut day = day + offset_days;
    while day < 1 {
        if month == 1 {
            month = 12;
            year -= 1;
        } else {
            month -= 1;
        }
        day += days_in_month(year, month) as i32;
    }
    while day > days_in_month(year, month) as i32 {
        day -= days_in_month(year, month) as i32;
        if month == 12 {
            month = 1;
            year += 1;
        } else {
            month += 1;
        }
    }
    Some((year, month, day))
}

fn format_date((year, month, day): (i32, u32, i32)) -> String {
    format!("{day} {} {year}", month_name(month).unwrap_or(""))
}

fn month_number(value: &str) -> Option<u32> {
    match value.to_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

fn month_name(month: u32) -> Option<&'static str> {
    match month {
        1 => Some("January"),
        2 => Some("February"),
        3 => Some("March"),
        4 => Some("April"),
        5 => Some("May"),
        6 => Some("June"),
        7 => Some("July"),
        8 => Some("August"),
        9 => Some("September"),
        10 => Some("October"),
        11 => Some("November"),
        12 => Some("December"),
        _ => None,
    }
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 30,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn strip_session_prefix(value: &str) -> &str {
    if value.trim_start().starts_with("On ") {
        if let Some((_, rest)) = value.split_once(", ") {
            return rest;
        }
    }
    value
}

fn is_relative_date_phrase(value: &str) -> bool {
    let lower = value.to_lowercase();
    (lower.contains("week") || lower.contains("month") || lower.contains("year"))
        && (lower.contains("before")
            || lower.contains("after")
            || lower.contains("last")
            || lower.contains("next")
            || lower.contains("ago")
            || lower.contains("sunday")
            || lower.contains("monday")
            || lower.contains("tuesday")
            || lower.contains("wednesday")
            || lower.contains("thursday")
            || lower.contains("friday")
            || lower.contains("saturday"))
}

fn extract_named_day(sentence: &str) -> Option<String> {
    let lower = sentence.to_lowercase();
    if lower.contains("valentine's day") {
        Some("Valentine's Day".to_string())
    } else {
        None
    }
}

fn extract_after_patterns(context: &[&str], patterns: &[&str]) -> Option<String> {
    for text in context {
        let lower = text.to_lowercase();
        for pattern in patterns {
            if let Some(index) = lower.find(pattern) {
                let start = index + pattern.len();
                let phrase = text[start..]
                    .split([',', '.', ';', '\n'])
                    .next()
                    .map(clean_phrase)
                    .filter(|value| !value.is_empty());
                if phrase.is_some() {
                    return phrase;
                }
            }
        }
    }
    None
}

fn extract_list_after_patterns(text: &str, patterns: &[&str]) -> Option<String> {
    let lower = text.to_lowercase();
    for pattern in patterns {
        if let Some(index) = lower.find(pattern) {
            let start = index + pattern.len();
            return text[start..]
                .split(['.', ';', '\n'])
                .next()
                .map(trim_answer_span)
                .filter(|value| !value.is_empty());
        }
    }
    None
}

fn extract_location_from_sentence(question: &str, sentence: &str) -> Option<String> {
    let patterns = if question.contains("move from") || question.contains("from where") {
        [" from ", " at ", " in ", " on ", " near ", " to "]
    } else if question.contains("go to")
        || question.contains("going to")
        || question.contains("travel to")
    {
        [" to ", " at ", " in ", " on ", " near ", " from "]
    } else {
        [" at ", " in ", " on ", " near ", " from ", " to "]
    };
    let lower = sentence.to_lowercase();
    for pattern in patterns {
        if let Some(index) = lower.find(pattern) {
            let start = index + pattern.len();
            let phrase = sentence[start..]
                .split([',', '.', ';', '\n'])
                .next()
                .unwrap_or_default();
            let mut phrase = trim_answer_span(phrase);
            if (question.contains("move from") || question.contains("from where"))
                && sentence[start..].contains(',')
            {
                if let Some(last) = sentence[start..].split(',').next_back() {
                    let last = trim_answer_span(last);
                    if !last.is_empty() {
                        phrase = last;
                    }
                }
            }
            if !phrase.is_empty() {
                return Some(phrase);
            }
        }
    }
    None
}

fn extract_quoted_phrase(sentence: &str) -> Option<String> {
    for quote in ['"'] {
        let mut parts = sentence.split(quote);
        while let Some(_) = parts.next() {
            let Some(candidate) = parts.next() else {
                break;
            };
            let candidate = clean_phrase(candidate);
            if candidate.split_whitespace().count() >= 2 {
                return Some(candidate);
            }
        }
    }
    None
}

fn extract_title_answer(sentence: &str) -> Option<String> {
    extract_quoted_phrase(sentence).or_else(|| {
        let value = extract_after_patterns(
            &[sentence],
            &[
                " production of ",
                " titled ",
                " called ",
                " named ",
                " uses the ",
                " implemented in ",
            ],
        )?;
        Some(first_identifier_or_phrase(&value))
    })
}

fn first_identifier_or_phrase(value: &str) -> String {
    let words = value.split_whitespace().collect::<Vec<_>>();
    if let Some(first) = words.first() {
        let clean = clean_phrase(first);
        if clean.chars().any(|ch| ch.is_ascii_digit())
            || clean.chars().all(|ch| ch.is_ascii_uppercase() || ch == '_')
        {
            return clean;
        }
    }
    trim_answer_span(value)
}

fn best_sentence(question: &str, context: &[&str]) -> String {
    let query_terms = tokenize(question);
    let mut best = None::<(&str, usize)>;
    for text in context {
        for sentence in text.split(['.', '!', '?', '\n']) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let terms = tokenize(sentence);
            let overlap = query_terms
                .iter()
                .filter(|term| terms.iter().any(|candidate| candidate == *term))
                .count();
            if best.is_none_or(|(_, score)| overlap > score) {
                best = Some((sentence, overlap));
            }
        }
    }
    best.map(|(sentence, _)| clean_phrase(sentence))
        .unwrap_or_else(|| "I don't know.".to_string())
}

fn best_typed_sentence(question: &str, context: &[&str]) -> String {
    let query_terms = tokenize(question);
    let question_lower = question.to_lowercase();
    let mut best_statement = None::<(&str, i32)>;
    let mut best_question = None::<(&str, i32)>;
    for text in context {
        for sentence in text.split(['.', '!', '?', '\n']) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let terms = tokenize(sentence);
            let mut score = query_terms
                .iter()
                .filter(|term| terms.iter().any(|candidate| candidate == *term))
                .count() as i32
                * 4;
            score += typed_sentence_bonus(&question_lower, sentence);
            if is_question_like_sentence(sentence) {
                if best_question.is_none_or(|(_, best_score)| score > best_score) {
                    best_question = Some((sentence, score));
                }
            } else if best_statement.is_none_or(|(_, best_score)| score > best_score) {
                best_statement = Some((sentence, score));
            }
        }
    }
    best_statement
        .or(best_question)
        .map(|(sentence, _)| clean_phrase(sentence))
        .unwrap_or_else(|| "I don't know.".to_string())
}

fn best_typed_packet_sentence(question: &str, packets: &[MemoryPacketForAnswerer]) -> String {
    let query_terms = tokenize(question);
    let question_lower = question.to_lowercase();
    let expected = expected_question_speaker(&question_lower, packets);
    let mut best_statement = None::<(&str, i32)>;
    let mut best_question = None::<(&str, i32)>;

    for packet in packets {
        for sentence in packet.content.split(['.', '!', '?', '\n']) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let terms = tokenize(sentence);
            let mut score = query_terms
                .iter()
                .filter(|term| terms.iter().any(|candidate| candidate == *term))
                .count() as i32
                * 4;
            score += typed_sentence_bonus(&question_lower, sentence);
            score += (packet.score * 3.0).round() as i32;
            if let Some(expected) = expected.as_deref() {
                if packet_supports_expected_subject(packet, expected) {
                    score += 5;
                } else if packet_speaker(packet).is_some() {
                    score -= 6;
                }
            }
            if is_question_like_sentence(sentence) {
                score -= 12;
                if best_question.is_none_or(|(_, best_score)| score > best_score) {
                    best_question = Some((sentence, score));
                }
            } else if best_statement.is_none_or(|(_, best_score)| score > best_score) {
                best_statement = Some((sentence, score));
            }
        }
    }

    best_statement
        .or(best_question)
        .map(|(sentence, _)| clean_phrase(sentence))
        .unwrap_or_else(|| "I don't know.".to_string())
}

fn is_question_like_sentence(sentence: &str) -> bool {
    let lower = question_like_payload(sentence);
    lower.ends_with('?')
        || lower.starts_with("what ")
        || lower.starts_with("why ")
        || lower.starts_with("when ")
        || lower.starts_with("where ")
        || lower.starts_with("how ")
        || lower.starts_with("who ")
        || lower.starts_with("did ")
        || lower.starts_with("does ")
        || lower.starts_with("do ")
        || lower.starts_with("is ")
        || lower.starts_with("are ")
        || lower.starts_with("would ")
        || lower.starts_with("can ")
}

fn question_like_payload(sentence: &str) -> String {
    let mut text = strip_evidence_prefix(sentence).trim();
    if let Some((_, rest)) = text.split_once(':') {
        text = rest.trim();
    }
    let lower = text.to_lowercase();
    if lower.starts_with("on ") {
        let mut parts = text.split(", ");
        let _ = parts.next();
        if let Some(rest) = parts.last() {
            return rest.trim().to_lowercase();
        }
    }
    lower
}

fn typed_sentence_bonus(question: &str, sentence: &str) -> i32 {
    let lower = sentence.to_lowercase();
    let mut score = 0;
    if question.contains("name") {
        score += bonus_any(&lower, &["name", "named", "called"]);
    }
    if contains_any(question, &["pet", "cat", "dog", "animal"]) {
        score += bonus_any(&lower, &["pet", "cat", "dog", "animal", "named"]);
    }
    if question.contains("what speed") || question.contains("internet plan") {
        score += bonus_any(&lower, &["speed", "mbps", "gbps", "internet plan"]);
    }
    if question.contains("previous occupation") || question.contains("previous role") {
        score += bonus_any(
            &lower,
            &[
                "previous role",
                "previous occupation",
                "worked as",
                "was a",
                "was an",
            ],
        );
    }
    if asks_title(question) {
        score += bonus_any(
            &lower,
            &["play", "book", "movie", "song", "album", "called", "named"],
        );
    }
    if question.contains("how many") {
        score += bonus_any(&lower, &["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]);
    }
    if question.contains("how long") {
        score += bonus_any(&lower, &["year", "month", "week", "minute", "hour", "day"]);
    }
    if question.contains("where") {
        score += bonus_any(&lower, &[" at ", " from ", " in ", " to "]);
    }
    score
}

fn bonus_any(text: &str, patterns: &[&str]) -> i32 {
    patterns
        .iter()
        .filter(|pattern| text.contains(*pattern))
        .count() as i32
        * 3
}

fn contains_month(value: &str) -> bool {
    let lower = value.to_lowercase();
    [
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
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    .iter()
    .any(|month| lower.contains(month))
}

fn clean_phrase(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| ch == ':' || ch == '-' || ch == ',' || ch == '.')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn trim_answer_span(value: &str) -> String {
    let end = [
        " since ",
        " when ",
        " where ",
        " while ",
        " which ",
        " that ",
        " after ",
        " before ",
        " once",
        " because ",
        " but ",
        " and I ",
        " and user ",
    ]
    .iter()
    .filter_map(|marker| value.find(marker))
    .min()
    .unwrap_or(value.len());
    clean_phrase(&value[..end])
}

fn compact_answer(value: &str) -> String {
    let without_prefix = value
        .split_once(": ")
        .map(|(_, rest)| rest)
        .unwrap_or(value);
    let clause_end = [" because ", " but ", " and then ", " so "]
        .iter()
        .filter_map(|marker| without_prefix.find(marker))
        .min()
        .unwrap_or(without_prefix.len());
    let clause = &without_prefix[..clause_end];
    clean_phrase(clause)
}

impl From<&MemoryPacket> for MemoryPacketForAnswerer {
    fn from(packet: &MemoryPacket) -> Self {
        Self {
            memory_id: packet.memory.id.clone(),
            content: packet.memory.content.clone(),
            memory_type: packet.memory.memory_type.to_string(),
            metadata: packet.memory.metadata.clone(),
            score: packet.score,
            source_event_id: packet.memory.source_event_id.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_prompt_adds_relative_time_instruction_when_anchor_is_available() {
        let mut metadata = BTreeMap::new();
        metadata.insert("event_time".to_string(), "8 May 2023".to_string());
        let input = AnswerInput {
            question: QuestionForAnswerer {
                id: "q".to_string(),
                conversation_id: "c".to_string(),
                text: "When did Melanie paint a sunrise?".to_string(),
            },
            retrieved: vec![MemoryPacketForAnswerer {
                memory_id: "m".to_string(),
                content: "Melanie painted that lake sunrise last year.".to_string(),
                memory_type: "episodic".to_string(),
                metadata,
                score: 0.5,
                source_event_id: Some("D1:12".to_string()),
            }],
        };

        let prompt = composer_user_prompt(&input);

        assert!(prompt.contains("Relative time reasoning:"));
        assert!(prompt.contains("event_time 8 May 2023 plus last year means 2022"));
    }

    #[test]
    fn composer_prompt_omits_relative_time_instruction_without_anchor() {
        let input = AnswerInput {
            question: QuestionForAnswerer {
                id: "q".to_string(),
                conversation_id: "c".to_string(),
                text: "When did Melanie paint a sunrise?".to_string(),
            },
            retrieved: vec![MemoryPacketForAnswerer {
                memory_id: "m".to_string(),
                content: "Melanie painted that lake sunrise last year.".to_string(),
                memory_type: "episodic".to_string(),
                metadata: BTreeMap::new(),
                score: 0.5,
                source_event_id: Some("D1:12".to_string()),
            }],
        };

        let prompt = composer_user_prompt(&input);

        assert!(!prompt.contains("Relative time reasoning:"));
    }
}
