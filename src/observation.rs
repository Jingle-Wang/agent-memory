use std::collections::BTreeSet;

use crate::embedding::tokenize;
use crate::models::{Event, Memory, MemoryType};
use crate::text::contains_any;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub kind: ObservationKind,
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub entities: Vec<String>,
    pub event_time: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservationKind {
    AssistantFact,
    Attribute,
    Event,
    Preference,
    Temporal,
    Update,
}

impl ObservationKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::AssistantFact => "assistant_fact",
            Self::Attribute => "attribute",
            Self::Event => "event",
            Self::Preference => "preference",
            Self::Temporal => "temporal",
            Self::Update => "update",
        }
    }
}

impl Observation {
    pub fn to_memory(&self, event: &Event) -> Memory {
        let content = format!(
            "[{}] {} {} {}",
            self.kind.as_str(),
            self.subject,
            self.relation,
            self.object
        );
        let mut memory = Memory::new(content, MemoryType::Semantic)
            .namespace(event.namespace.clone())
            .source_event(event.id.clone())
            .importance(importance(&self.kind))
            .confidence(confidence(&self.kind));
        memory
            .metadata
            .insert("memory_kind".to_string(), "observation".to_string());
        memory
            .metadata
            .insert("side_channel".to_string(), "observation".to_string());
        memory.metadata.insert(
            "observation_kind".to_string(),
            self.kind.as_str().to_string(),
        );
        memory
            .metadata
            .insert("subject".to_string(), self.subject.clone());
        memory
            .metadata
            .insert("relation".to_string(), self.relation.clone());
        memory
            .metadata
            .insert("object".to_string(), self.object.clone());
        memory
            .metadata
            .insert("entities".to_string(), self.entities.join("\n"));
        if let Some(time) = &self.event_time {
            memory
                .metadata
                .insert("event_time".to_string(), time.clone());
        }
        memory
    }

    pub fn to_profile_memory(&self, event: &Event) -> Option<Memory> {
        if !matches!(
            self.kind,
            ObservationKind::Attribute
                | ObservationKind::Preference
                | ObservationKind::Update
                | ObservationKind::Temporal
        ) {
            return None;
        }
        let mut memory = Memory::new(
            format!(
                "[profile] {} {}: {}",
                self.subject, self.relation, self.object
            ),
            MemoryType::Semantic,
        )
        .namespace(event.namespace.clone())
        .source_event(event.id.clone())
        .importance(0.56)
        .confidence(0.64);
        memory
            .metadata
            .insert("memory_kind".to_string(), "profile".to_string());
        memory
            .metadata
            .insert("subject".to_string(), self.subject.clone());
        memory
            .metadata
            .insert("relation".to_string(), self.relation.clone());
        memory
            .metadata
            .insert("object".to_string(), self.object.clone());
        memory
            .metadata
            .insert("entities".to_string(), self.entities.join("\n"));
        if let Some(time) = &self.event_time {
            memory
                .metadata
                .insert("event_time".to_string(), time.clone());
        }
        Some(memory)
    }
}

pub fn extract_observations(event: &Event) -> Vec<Observation> {
    let timestamp = event
        .metadata
        .get("benchmark_timestamp")
        .filter(|value| !value.is_empty())
        .cloned();
    split_sentences(&event.text)
        .into_iter()
        .filter(|sentence| sentence.split_whitespace().count() >= 3)
        .filter(|sentence| sentence.chars().count() <= 360)
        .filter_map(|sentence| observation_from_sentence(&sentence, event, timestamp.clone()))
        .collect()
}

fn observation_from_sentence(
    sentence: &str,
    event: &Event,
    timestamp: Option<String>,
) -> Option<Observation> {
    let lower = sentence.to_lowercase();
    let actor = normalize_actor(&event.actor);
    let subject = extract_subject(sentence).unwrap_or(actor);
    let mut kind = None;
    let mut relation = None;
    let mut object = None;

    if lower.contains("gift") && lower.contains(" from ") {
        kind = Some(ObservationKind::Attribute);
        relation = Some("gift_from".to_string());
        object = extract_after_patterns(sentence, &[" from my ", " from user's ", " from "])
            .map(|value| format!("from {value}"));
    } else if lower.contains("redeem") && lower.contains("coupon") {
        kind = Some(ObservationKind::Attribute);
        relation = Some("redeemed_at".to_string());
        object = extract_after_patterns(sentence, &[" at ", " from "]);
    } else if contains_any(&lower, &["bought", "buy", "purchased", "got"])
        && lower.contains(" from ")
    {
        kind = Some(ObservationKind::Attribute);
        relation = Some("purchase_location".to_string());
        object = extract_after_patterns(sentence, &[" from ", " at "]);
    } else if contains_any(&lower, &["mbps", "gbps"]) {
        kind = Some(ObservationKind::Attribute);
        relation = Some("quantity".to_string());
        object = extract_around_unit(sentence, &["mbps", "gbps"]);
    } else if contains_any(
        &lower,
        &[
            "i prefer",
            "i like",
            "i love",
            "i dislike",
            "favorite",
            "rather",
            "want to",
            "looking for",
        ],
    ) {
        kind = Some(ObservationKind::Preference);
        relation = Some(preference_relation(&lower).to_string());
        object = if lower.contains("favorite") {
            extract_after_patterns(sentence, &[" favorite ", " favorite is ", " is "])
        } else {
            Some(rewrite_first_person(sentence, &subject))
        };
    } else if contains_any(
        &lower,
        &[
            "my name is",
            "call me",
            "i am",
            "i'm",
            "i work",
            "i live",
            "i moved",
            "i graduated",
            "my current",
            "my previous",
            "my new",
        ],
    ) {
        kind = Some(
            if contains_any(&lower, &["current", "previous", "new", "now", "changed"]) {
                ObservationKind::Update
            } else {
                ObservationKind::Attribute
            },
        );
        relation = Some(attribute_relation(&lower).to_string());
        object = Some(rewrite_first_person(sentence, &subject));
    } else if contains_any(
        &lower,
        &[
            "when",
            "last",
            "next",
            "ago",
            "before",
            "after",
            "today",
            "yesterday",
            "tomorrow",
        ],
    ) {
        kind = Some(ObservationKind::Temporal);
        relation = Some("happened_at_or_near".to_string());
        object = Some(rewrite_first_person(sentence, &subject));
    }

    let kind = kind?;
    let relation = relation?;
    let object = object?;
    if object.split_whitespace().count() < 2 {
        return None;
    }

    let mut entities = extract_entities(sentence);
    entities.push(subject.clone());
    Some(Observation {
        kind,
        subject,
        relation,
        object,
        entities: dedupe(entities),
        event_time: timestamp,
    })
}

fn importance(kind: &ObservationKind) -> f32 {
    match kind {
        ObservationKind::Preference | ObservationKind::Update => 0.58,
        ObservationKind::Temporal | ObservationKind::Attribute => 0.54,
        ObservationKind::AssistantFact => 0.50,
        ObservationKind::Event => 0.45,
    }
}

fn confidence(kind: &ObservationKind) -> f32 {
    match kind {
        ObservationKind::Preference | ObservationKind::Attribute | ObservationKind::Update => 0.62,
        ObservationKind::Temporal => 0.58,
        ObservationKind::AssistantFact => 0.55,
        ObservationKind::Event => 0.50,
    }
}

fn preference_relation(lower: &str) -> &'static str {
    if lower.contains("dislike") {
        "dislikes"
    } else if lower.contains("favorite") {
        "has_favorite"
    } else if lower.contains("want") || lower.contains("looking for") {
        "wants"
    } else {
        "prefers"
    }
}

fn attribute_relation(lower: &str) -> &'static str {
    if lower.contains("graduated") || lower.contains("degree") {
        "education"
    } else if lower.contains("work") || lower.contains("role") || lower.contains("job") {
        "work"
    } else if lower.contains("live") || lower.contains("moved") {
        "location"
    } else if lower.contains("name") || lower.contains("call me") {
        "identity"
    } else {
        "attribute"
    }
}

fn extract_subject(sentence: &str) -> Option<String> {
    if contains_any(&sentence.to_lowercase(), &[" i ", "i'm", "i am", " my "]) {
        return Some("user".to_string());
    }
    extract_entities(sentence).into_iter().next()
}

fn extract_entities(sentence: &str) -> Vec<String> {
    let mut entities = Vec::new();
    let mut current = Vec::new();
    for token in sentence.split_whitespace() {
        let clean = token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '\'');
        let is_entity = clean
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())
            && !is_sentence_start_word(clean);
        if is_entity {
            current.push(clean.to_string());
        } else if !current.is_empty() {
            entities.push(current.join(" "));
            current.clear();
        }
    }
    if !current.is_empty() {
        entities.push(current.join(" "));
    }
    entities
        .into_iter()
        .filter(|entity| entity.len() > 1)
        .collect()
}

fn is_sentence_start_word(value: &str) -> bool {
    matches!(
        value,
        "I" | "I'm"
            | "My"
            | "The"
            | "A"
            | "An"
            | "It"
            | "This"
            | "That"
            | "When"
            | "What"
            | "Where"
            | "How"
    )
}

fn rewrite_first_person(sentence: &str, subject: &str) -> String {
    let mut text = format!(" {sentence} ");
    for (from, to) in [
        (" I'm ", " is "),
        (" I am ", " is "),
        (" I've ", " has "),
        (" I have ", " has "),
        (" I prefer ", " prefers "),
        (" I like ", " likes "),
        (" I love ", " loves "),
        (" I dislike ", " dislikes "),
        (" I want ", " wants "),
        (" I work ", " works "),
        (" I live ", " lives "),
        (" I moved ", " moved "),
        (" I graduated ", " graduated "),
    ] {
        text = text.replace(from, &format!(" {subject}{to}"));
    }
    text = text.replace(" my ", &format!(" {subject}'s "));
    text = text.replace(" My ", &format!(" {subject}'s "));
    clean_object(&text)
}

fn clean_object(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| ch == ':' || ch == '-' || ch == ',' || ch == '.')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_after_patterns(sentence: &str, patterns: &[&str]) -> Option<String> {
    let lower = sentence.to_lowercase();
    for pattern in patterns {
        if let Some(index) = lower.find(pattern) {
            let start = index + pattern.len();
            let value = sentence[start..]
                .split([',', '.', ';', '\n'])
                .next()
                .map(clean_object)
                .filter(|value| !value.is_empty());
            if value.is_some() {
                return value;
            }
        }
    }
    None
}

fn extract_around_unit(sentence: &str, units: &[&str]) -> Option<String> {
    let tokens = sentence.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return None;
    }
    for (index, token) in tokens.iter().enumerate() {
        let clean = token
            .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
            .to_lowercase();
        if units.contains(&clean.as_str()) {
            let start = index.saturating_sub(1);
            let end = (index + 1).min(tokens.len() - 1);
            return Some(clean_object(&tokens[start..=end].join(" ")));
        }
    }
    None
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn normalize_actor(actor: &str) -> String {
    let lower = actor.to_lowercase();
    if lower.contains("user") || lower.contains("human") {
        "user".to_string()
    } else if lower.contains("assistant") || lower.contains("agent") {
        "assistant".to_string()
    } else {
        actor.to_string()
    }
}

fn dedupe(values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .flat_map(|value| {
            tokenize(&value)
                .into_iter()
                .filter(|token| token.len() > 1)
                .collect::<Vec<_>>()
        })
        .filter(|value| seen.insert(value.clone()))
        .collect()
}
