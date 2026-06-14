use std::fs;
use std::path::Path;

use serde_json::Value;

use super::dataset::{BenchmarkDataset, BenchmarkQuestion, BenchmarkTurn, Conversation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkKind {
    Generic,
    Locomo,
    LongMemEval,
}

impl BenchmarkKind {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_lowercase().as_str() {
            "generic" | "agent-memory" | "agent_memory" => Ok(Self::Generic),
            "locomo" => Ok(Self::Locomo),
            "longmemeval" | "long_mem_eval" | "lme" => Ok(Self::LongMemEval),
            other => Err(format!("unknown benchmark kind: {other}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BenchmarkKind::Generic => "generic",
            BenchmarkKind::Locomo => "locomo",
            BenchmarkKind::LongMemEval => "longmemeval",
        }
    }
}

pub fn load_dataset(
    path: impl AsRef<Path>,
    kind: BenchmarkKind,
) -> Result<BenchmarkDataset, String> {
    let raw = fs::read_to_string(path.as_ref()).map_err(|error| error.to_string())?;
    let value: Value = serde_json::from_str(&raw).map_err(|error| error.to_string())?;
    load_dataset_value(&value, kind)
}

pub fn load_dataset_value(value: &Value, kind: BenchmarkKind) -> Result<BenchmarkDataset, String> {
    match kind {
        BenchmarkKind::Locomo => {
            if let Some(dataset) = load_locomo_value(value)? {
                return Ok(dataset);
            }
        }
        BenchmarkKind::LongMemEval => {
            if let Some(dataset) = load_longmemeval_value(value)? {
                return Ok(dataset);
            }
        }
        BenchmarkKind::Generic => {}
    }

    if let Ok(dataset) = serde_json::from_value::<BenchmarkDataset>(value.clone()) {
        return Ok(dataset);
    }

    let records = records_from_value(value);
    if records.is_empty() {
        return Err("dataset contains no conversation records".to_string());
    }

    let mut conversations = Vec::new();
    let mut questions = Vec::new();

    for (record_index, record) in records.iter().enumerate() {
        let conversation_id = first_string(
            record,
            &[
                "conversation_id",
                "conversationId",
                "dialogue_id",
                "dialogueId",
                "sample_id",
                "id",
            ],
        )
        .unwrap_or_else(|| format!("conversation-{record_index}"));

        let turns = extract_turns(record, &conversation_id);
        if !turns.is_empty() {
            conversations.push(Conversation {
                id: conversation_id.clone(),
                turns,
            });
        }

        questions.extend(extract_questions(record, &conversation_id));
    }

    if conversations.is_empty() {
        return Err("dataset contains no turns".to_string());
    }
    if questions.is_empty() {
        return Err("dataset contains no questions".to_string());
    }

    Ok(BenchmarkDataset {
        name: kind.name().to_string(),
        version: "loaded-json".to_string(),
        conversations,
        questions,
    })
}

fn load_locomo_value(value: &Value) -> Result<Option<BenchmarkDataset>, String> {
    let Some(records) = value.as_array() else {
        return Ok(None);
    };
    if records
        .first()
        .and_then(|record| record.get("conversation"))
        .is_none()
    {
        return Ok(None);
    }

    let mut conversations = Vec::new();
    let mut questions = Vec::new();
    for (record_index, record) in records.iter().enumerate() {
        let conversation_id = first_string(record, &["sample_id"])
            .unwrap_or_else(|| format!("locomo-{record_index}"));
        let conversation = record
            .get("conversation")
            .and_then(Value::as_object)
            .ok_or_else(|| "LoCoMo record missing conversation object".to_string())?;
        let mut session_numbers = conversation
            .keys()
            .filter_map(|key| {
                key.strip_prefix("session_")?
                    .parse::<usize>()
                    .ok()
                    .map(|number| (number, key.clone()))
            })
            .collect::<Vec<_>>();
        session_numbers.sort_by_key(|(number, _)| *number);

        let mut turns = Vec::new();
        for (session_number, key) in session_numbers {
            let timestamp_key = format!("session_{session_number}_date_time");
            let timestamp = conversation
                .get(&timestamp_key)
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let Some(session_turns) = conversation.get(&key).and_then(Value::as_array) else {
                continue;
            };
            for (turn_index, turn) in session_turns.iter().enumerate() {
                let text = first_string(turn, &["text", "content"]).unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                turns.push(BenchmarkTurn {
                    id: first_string(turn, &["dia_id", "id"])
                        .map(|id| scoped_id(&conversation_id, &id))
                        .unwrap_or_else(|| {
                            scoped_id(
                                &conversation_id,
                                &format!("s{session_number}:t{turn_index}"),
                            )
                        }),
                    speaker: first_string(turn, &["speaker", "role"])
                        .unwrap_or_else(|| "unknown".to_string()),
                    text,
                    timestamp: timestamp.clone(),
                });
            }
        }

        conversations.push(Conversation {
            id: conversation_id.clone(),
            turns,
        });

        if let Some(qa) = record.get("qa").and_then(Value::as_array) {
            for (question_index, question) in qa.iter().enumerate() {
                let Some(text) = first_string(question, &["question"]) else {
                    continue;
                };
                let category = first_string(question, &["category"]);
                let mut gold_answers = answer_strings(question);
                if category.as_deref() == Some("5") && gold_answers.is_empty() {
                    gold_answers.push("None".to_string());
                }
                questions.push(BenchmarkQuestion {
                    id: format!("{conversation_id}:q:{question_index}"),
                    conversation_id: conversation_id.clone(),
                    text,
                    gold_answers,
                    evidence_turn_ids: evidence_strings(question)
                        .into_iter()
                        .map(|id| scoped_id(&conversation_id, &id))
                        .collect(),
                    category,
                });
            }
        }
    }

    Ok(Some(BenchmarkDataset {
        name: "locomo".to_string(),
        version: "locomo10".to_string(),
        conversations,
        questions,
    }))
}

fn load_longmemeval_value(value: &Value) -> Result<Option<BenchmarkDataset>, String> {
    let Some(records) = value.as_array() else {
        return Ok(None);
    };
    if records
        .first()
        .and_then(|record| record.get("haystack_sessions"))
        .is_none()
    {
        return Ok(None);
    }

    let mut conversations = Vec::new();
    let mut questions = Vec::new();
    for (record_index, record) in records.iter().enumerate() {
        let question_id = first_string(record, &["question_id"])
            .unwrap_or_else(|| format!("longmemeval-question-{record_index}"));
        let session_ids = record
            .get("haystack_session_ids")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let dates = record
            .get("haystack_dates")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let sessions = record
            .get("haystack_sessions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let mut turns = Vec::new();
        let mut evidence_turn_ids = Vec::new();
        for (session_index, session) in sessions.iter().enumerate() {
            let session_id = session_ids
                .get(session_index)
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("{question_id}:session:{session_index}"));
            let timestamp = dates
                .get(session_index)
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let Some(session_turns) = session.as_array() else {
                continue;
            };
            for (turn_index, turn) in session_turns.iter().enumerate() {
                let text = first_string(turn, &["content", "text", "message"]).unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                let turn_id = scoped_id(
                    &question_id,
                    &format!("session:{session_index}:{session_id}:turn:{turn_index}"),
                );
                if turn
                    .get("has_answer")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    evidence_turn_ids.push(turn_id.clone());
                }
                turns.push(BenchmarkTurn {
                    id: turn_id,
                    speaker: first_string(turn, &["role", "speaker"])
                        .unwrap_or_else(|| "unknown".to_string()),
                    text,
                    timestamp: timestamp.clone(),
                });
            }
        }

        if evidence_turn_ids.is_empty() {
            evidence_turn_ids = record
                .get("answer_session_ids")
                .map(strings_from_value)
                .unwrap_or_default()
                .into_iter()
                .map(|id| scoped_id(&question_id, &id))
                .collect();
        }

        conversations.push(Conversation {
            id: question_id.clone(),
            turns,
        });
        questions.push(BenchmarkQuestion {
            id: question_id.clone(),
            conversation_id: question_id,
            text: first_string(record, &["question"]).unwrap_or_default(),
            gold_answers: answer_strings(record),
            evidence_turn_ids,
            category: first_string(record, &["question_type"]),
        });
    }

    Ok(Some(BenchmarkDataset {
        name: "longmemeval".to_string(),
        version: "longmemeval_s_cleaned".to_string(),
        conversations,
        questions,
    }))
}

fn scoped_id(scope: &str, id: &str) -> String {
    format!("{scope}::{id}")
}

fn records_from_value(value: &Value) -> Vec<Value> {
    if let Some(array) = value.as_array() {
        return array.clone();
    }
    for key in ["conversations", "data", "samples", "dialogs", "dialogues"] {
        if let Some(array) = value.get(key).and_then(Value::as_array) {
            return array.clone();
        }
    }
    vec![value.clone()]
}

fn extract_turns(record: &Value, conversation_id: &str) -> Vec<BenchmarkTurn> {
    let Some(turn_values) = first_array(
        record,
        &[
            "turns",
            "messages",
            "conversation",
            "dialogue",
            "sessions",
            "haystack_sessions",
        ],
    ) else {
        return Vec::new();
    };

    flatten_turns(turn_values)
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let text = first_string(
                &value,
                &["text", "content", "message", "utterance", "response"],
            )?;
            Some(BenchmarkTurn {
                id: first_string(&value, &["id", "turn_id", "turnId", "message_id"])
                    .unwrap_or_else(|| format!("{conversation_id}:turn:{index}")),
                speaker: first_string(&value, &["speaker", "role", "sender", "author"])
                    .unwrap_or_else(|| "unknown".to_string()),
                text,
                timestamp: first_string(&value, &["timestamp", "time", "date"]),
            })
        })
        .collect()
}

fn flatten_turns(values: &[Value]) -> Vec<Value> {
    let mut turns = Vec::new();
    for value in values {
        if let Some(nested) = first_array(value, &["turns", "messages", "dialogue"]) {
            turns.extend(flatten_turns(nested));
        } else {
            turns.push(value.clone());
        }
    }
    turns
}

fn extract_questions(record: &Value, conversation_id: &str) -> Vec<BenchmarkQuestion> {
    let mut result = Vec::new();
    for (index, value) in question_records(record).iter().enumerate() {
        let Some(text) = first_string(value, &["question", "query", "input", "text"]) else {
            continue;
        };
        result.push(BenchmarkQuestion {
            id: first_string(value, &["id", "question_id", "questionId", "qid"])
                .unwrap_or_else(|| format!("{conversation_id}:question:{index}")),
            conversation_id: first_string(value, &["conversation_id", "conversationId"])
                .unwrap_or_else(|| conversation_id.to_string()),
            text,
            gold_answers: answer_strings(value),
            evidence_turn_ids: evidence_strings(value),
            category: first_string(value, &["category", "type", "question_type"]),
        });
    }
    result
}

fn question_records(record: &Value) -> Vec<Value> {
    for key in ["questions", "qa", "qas", "qa_pairs", "evaluation"] {
        if let Some(array) = record.get(key).and_then(Value::as_array) {
            return array.clone();
        }
    }
    if first_string(record, &["question", "query"]).is_some() {
        return vec![record.clone()];
    }
    Vec::new()
}

fn answer_strings(value: &Value) -> Vec<String> {
    for key in [
        "gold_answers",
        "gold_answer",
        "answers",
        "answer",
        "target",
        "reference",
    ] {
        if let Some(strings) = value.get(key).map(strings_from_value) {
            if !strings.is_empty() {
                return strings;
            }
        }
    }
    Vec::new()
}

fn evidence_strings(value: &Value) -> Vec<String> {
    for key in [
        "evidence_turn_ids",
        "evidence",
        "evidence_ids",
        "supporting_turns",
        "target_turn_ids",
    ] {
        if let Some(strings) = value.get(key).map(strings_from_value) {
            if !strings.is_empty() {
                return strings;
            }
        }
    }
    Vec::new()
}

fn strings_from_value(value: &Value) -> Vec<String> {
    match value {
        Value::Null => vec!["None".to_string()],
        Value::String(text) => vec![text.clone()],
        Value::Number(number) => vec![number.to_string()],
        Value::Array(values) => values.iter().flat_map(strings_from_value).collect(),
        Value::Object(map) => {
            for key in ["id", "turn_id", "text", "answer", "content"] {
                if let Some(strings) = map.get(key).map(strings_from_value) {
                    if !strings.is_empty() {
                        return strings;
                    }
                }
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn first_array<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Vec<Value>> {
    keys.iter().find_map(|key| value.get(key)?.as_array())
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        let value = value.get(key)?;
        match value {
            Value::String(text) if !text.is_empty() => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        }
    })
}
