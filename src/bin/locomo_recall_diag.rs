use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;

use agent_memory::benchmark::{BenchmarkKind, QuestionResult, load_dataset};
use agent_memory::embedding::token_overlap_score;
use agent_memory::{MemoryEngine, MemoryPacket, MemoryQuery, MemoryStore, SqliteMemoryStore};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let dataset = load_dataset(&args.dataset, BenchmarkKind::Locomo)?;
    let scores = read_scores(&args.scores)?;
    let miss_ids = scores
        .iter()
        .filter(|result| result.retrieval.recall_at_10 == 0.0)
        .take(args.limit)
        .map(|result| result.question_id.clone())
        .collect::<BTreeSet<_>>();
    let score_by_question = scores
        .into_iter()
        .map(|result| (result.question_id.clone(), result))
        .collect::<BTreeMap<_, _>>();

    let store = SqliteMemoryStore::open(&args.db)?;
    let engine = create_engine(store);

    for question in dataset
        .questions
        .iter()
        .filter(|question| miss_ids.contains(&question.id))
    {
        let score = score_by_question
            .get(&question.id)
            .ok_or("score missing for selected question")?;
        let conversation = dataset
            .conversations
            .iter()
            .find(|conversation| conversation.id == question.conversation_id)
            .ok_or("conversation missing")?;

        println!("## {}", question.id);
        println!("question: {}", question.text);
        println!("gold_answers: {}", question.gold_answers.join(" | "));
        println!("gold_turn_ids: {}", question.evidence_turn_ids.join(", "));
        println!(
            "benchmark_top10_sources: {}",
            score.retrieved_source_event_ids.join(", ")
        );

        let all_memories = engine.store().list_memories(
            &MemoryQuery::new("")
                .namespace(question.conversation_id.clone())
                .limit(usize::MAX),
        )?;
        let events = engine.store().list_events(&question.conversation_id);
        let event_ids = events
            .iter()
            .map(|event| event.id.as_str())
            .collect::<BTreeSet<_>>();
        let turn_text_by_id = conversation
            .turns
            .iter()
            .map(|turn| (turn.id.as_str(), turn.text.as_str()))
            .collect::<BTreeMap<_, _>>();

        for gold_id in &question.evidence_turn_ids {
            println!("  gold {}", gold_id);
            println!("    event_stored: {}", event_ids.contains(gold_id.as_str()));
            if let Some(text) = turn_text_by_id.get(gold_id.as_str()) {
                println!("    turn_text: {}", one_line(text, 220));
            }
            let source_memories = all_memories
                .iter()
                .filter(|memory| memory.source_event_id.as_deref() == Some(gold_id.as_str()))
                .collect::<Vec<_>>();
            println!("    source_memory_count: {}", source_memories.len());
            for memory in source_memories.iter().take(6) {
                println!(
                    "      mem {} kind={} type={} overlap={:.3} text={}",
                    memory.id,
                    memory
                        .metadata
                        .get("memory_kind")
                        .map(String::as_str)
                        .unwrap_or(""),
                    memory.memory_type,
                    token_overlap_score(&question.text, &memory.content),
                    one_line(&memory.content, 180)
                );
            }
        }

        let raw_candidates = engine.search(
            MemoryQuery::new(question.text.clone())
                .namespace(question.conversation_id.clone())
                .limit(200),
        )?;
        let full_candidates = engine.search(
            MemoryQuery::new(question.text.clone())
                .namespace(question.conversation_id.clone())
                .limit(5000),
        )?;
        let reranked = rerank_answer_candidates(&question.text, raw_candidates.clone());
        print_gold_ranks("raw_search_200", question, &raw_candidates);
        print_gold_ranks("raw_search_full", question, &full_candidates);
        print_gold_ranks("answer_rerank_200", question, &reranked);

        println!("  top10_after_answer_rerank:");
        for (index, packet) in reranked.iter().take(10).enumerate() {
            println!(
                "    #{:<2} score={:.4} source={} kind={} reasons={} text={}",
                index + 1,
                packet.score,
                packet.memory.source_event_id.as_deref().unwrap_or(""),
                packet
                    .memory
                    .metadata
                    .get("memory_kind")
                    .map(String::as_str)
                    .unwrap_or(""),
                packet.reasons.join("+"),
                one_line(&packet.memory.content, 180)
            );
        }
        println!();
    }

    Ok(())
}

#[allow(unused_mut)]
fn create_engine(store: SqliteMemoryStore) -> MemoryEngine<SqliteMemoryStore> {
    #[cfg(feature = "embed-ollama")]
    {
        MemoryEngine::new_with_embedding(
            store,
            Box::new(agent_memory::embedding::OllamaEmbeddingProvider::from_env()),
        )
    }
    #[cfg(not(feature = "embed-ollama"))]
    {
        MemoryEngine::new(store)
    }
}

fn read_scores(path: &PathBuf) -> Result<Vec<QuestionResult>, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    raw.lines()
        .map(|line| serde_json::from_str::<QuestionResult>(line).map_err(Into::into))
        .collect()
}

fn print_gold_ranks(
    label: &str,
    question: &agent_memory::benchmark::BenchmarkQuestion,
    packets: &[MemoryPacket],
) {
    println!("  {label}:");
    for gold_id in &question.evidence_turn_ids {
        let ranks = packets
            .iter()
            .enumerate()
            .filter(|(_, packet)| {
                packet.memory.source_event_id.as_deref() == Some(gold_id.as_str())
            })
            .map(|(index, packet)| {
                format!(
                    "#{} score={:.4} kind={} reasons={} overlap={:.3}",
                    index + 1,
                    packet.score,
                    packet
                        .memory
                        .metadata
                        .get("memory_kind")
                        .map(String::as_str)
                        .unwrap_or(""),
                    packet.reasons.join("+"),
                    token_overlap_score(&question.text, &packet.memory.content),
                )
            })
            .collect::<Vec<_>>();
        if ranks.is_empty() {
            println!("    {}: not_in_candidates", gold_id);
        } else {
            println!("    {}: {}", gold_id, ranks.join("; "));
        }
    }
}

fn rerank_answer_candidates(
    question: &str,
    mut candidates: Vec<MemoryPacket>,
) -> Vec<MemoryPacket> {
    let lower = question.to_lowercase();
    for (index, packet) in candidates.iter_mut().enumerate() {
        let bonus = rerank_bonus(&lower, packet);
        if bonus != 0.0 {
            packet.score += bonus;
            packet.reasons.push("answer_rerank".to_string());
        }
        packet.score -= (index as f32) * 0.0001;
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    diversify_answer_candidates(candidates)
}

fn diversify_answer_candidates(candidates: Vec<MemoryPacket>) -> Vec<MemoryPacket> {
    let mut selected = Vec::new();
    let mut deferred = Vec::new();
    let mut seen_sources = BTreeSet::new();
    for packet in candidates {
        let duplicate_source = packet
            .memory
            .source_event_id
            .as_ref()
            .is_some_and(|source| !seen_sources.insert(source.clone()));
        if duplicate_source {
            deferred.push(packet);
        } else {
            selected.push(packet);
        }
    }
    selected.extend(deferred);
    selected
}

fn rerank_bonus(question_lower: &str, packet: &MemoryPacket) -> f32 {
    let content = packet.memory.content.to_lowercase();
    let metadata = &packet.memory.metadata;
    let kind = metadata
        .get("memory_kind")
        .map(String::as_str)
        .unwrap_or_default();
    let mut bonus = 0.0;

    if matches!(kind, "verbatim_turn" | "llm_fact" | "observation") {
        bonus += 0.03;
    }
    if kind == "verbatim_session" {
        bonus -= 0.04;
    }
    if content.trim_end().ends_with('?') || content.contains("guess what") {
        bonus -= 0.20;
    }
    if contains_any(question_lower, &["when", "what date", "what year"])
        && (metadata.contains_key("event_time")
            || contains_any(
                &content,
                &["last ", "yesterday", "friday", "sunday", "2022", "2023"],
            ))
    {
        bonus += 0.06;
    }
    if contains_any(question_lower, &["how many", "times"])
        && contains_any(&content, &[" once ", " twice ", " two ", " 2 ", "second"])
    {
        bonus += 0.08;
    }
    if token_overlap_score(question_lower, &content) > 0.0 {
        bonus += 0.02;
    }
    bonus
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn one_line(text: &str, limit: usize) -> String {
    let mut value = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if value.len() > limit {
        value.truncate(limit);
        value.push_str("...");
    }
    value
}

#[derive(Clone, Debug)]
struct Args {
    dataset: PathBuf,
    db: PathBuf,
    scores: PathBuf,
    limit: usize,
}

impl Args {
    fn parse(values: Vec<String>) -> Result<Self, String> {
        let mut dataset = PathBuf::from("data/benchmarks/locomo/locomo10.json");
        let mut db = PathBuf::from("runs/latest/memory.db");
        let mut scores = PathBuf::from("runs/latest/scores.jsonl");
        let mut limit = 5_usize;
        let mut index = 0;
        while index < values.len() {
            let key = &values[index];
            let value = values
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {key}"))?;
            match key.as_str() {
                "--dataset" => dataset = PathBuf::from(value),
                "--db" => db = PathBuf::from(value),
                "--scores" => scores = PathBuf::from(value),
                "--limit" => {
                    limit = value
                        .parse()
                        .map_err(|_| format!("invalid --limit value: {value}"))?
                }
                other => return Err(format!("unknown argument: {other}")),
            }
            index += 2;
        }
        Ok(Self {
            dataset,
            db,
            scores,
            limit,
        })
    }
}
