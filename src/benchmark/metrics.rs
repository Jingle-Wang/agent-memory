use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    pub recall_at_1: f32,
    pub recall_at_3: f32,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub recall_at_20: f32,
    pub recall_at_50: f32,
    pub recall_at_100: f32,
    pub recall_at_200: f32,
    pub mrr: f32,
    pub ndcg_at_5: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuestionResult {
    pub question_id: String,
    pub conversation_id: String,
    pub category: Option<String>,
    pub retrieved_memory_ids: Vec<String>,
    pub retrieved_source_event_ids: Vec<String>,
    pub answer: Option<String>,
    pub answer_correct: Option<bool>,
    pub answer_score: Option<f32>,
    pub retrieval: RetrievalMetrics,
    pub latency_ms: u128,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub benchmark: String,
    pub mode: String,
    pub question_count: usize,
    pub answered_count: usize,
    pub accuracy: Option<f32>,
    pub retrieval_miss_at_10_rate: f32,
    pub hit_at_10_answer_wrong_rate: f32,
    pub hit_at_1_answer_wrong_rate: f32,
    pub recall_at_1: f32,
    pub recall_at_3: f32,
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub recall_at_20: f32,
    pub recall_at_50: f32,
    pub recall_at_100: f32,
    pub recall_at_200: f32,
    pub mrr: f32,
    pub ndcg_at_5: f32,
    pub avg_latency_ms: f32,
}

pub fn retrieval_metrics(
    evidence_ids: &[String],
    retrieved_source_ids: &[String],
) -> RetrievalMetrics {
    let evidence: BTreeSet<_> = evidence_ids.iter().collect();
    if evidence.is_empty() {
        return RetrievalMetrics::default();
    }

    RetrievalMetrics {
        recall_at_1: recall_at(&evidence, retrieved_source_ids, 1),
        recall_at_3: recall_at(&evidence, retrieved_source_ids, 3),
        recall_at_5: recall_at(&evidence, retrieved_source_ids, 5),
        recall_at_10: recall_at(&evidence, retrieved_source_ids, 10),
        recall_at_20: recall_at(&evidence, retrieved_source_ids, 20),
        recall_at_50: recall_at(&evidence, retrieved_source_ids, 50),
        recall_at_100: recall_at(&evidence, retrieved_source_ids, 100),
        recall_at_200: recall_at(&evidence, retrieved_source_ids, 200),
        mrr: reciprocal_rank(&evidence, retrieved_source_ids),
        ndcg_at_5: ndcg_at(&evidence, retrieved_source_ids, 5),
    }
}

pub fn summarize(benchmark: &str, mode: &str, results: &[QuestionResult]) -> BenchmarkSummary {
    if results.is_empty() {
        return BenchmarkSummary {
            benchmark: benchmark.to_string(),
            mode: mode.to_string(),
            ..BenchmarkSummary::default()
        };
    }
    let count = results.len() as f32;
    let answered: Vec<_> = results
        .iter()
        .filter(|result| result.answer_correct.is_some())
        .collect();
    let accuracy = if answered.is_empty() {
        None
    } else {
        Some(
            answered
                .iter()
                .filter(|result| result.answer_correct == Some(true))
                .count() as f32
                / answered.len() as f32,
        )
    };

    BenchmarkSummary {
        benchmark: benchmark.to_string(),
        mode: mode.to_string(),
        question_count: results.len(),
        answered_count: answered.len(),
        accuracy,
        retrieval_miss_at_10_rate: avg(
            results,
            |result| {
                if result.retrieval.recall_at_10 <= 0.0 {
                    1.0
                } else {
                    0.0
                }
            },
            count,
        ),
        hit_at_10_answer_wrong_rate: avg(
            results,
            |result| {
                if result.retrieval.recall_at_10 > 0.0 && result.answer_correct == Some(false) {
                    1.0
                } else {
                    0.0
                }
            },
            count,
        ),
        hit_at_1_answer_wrong_rate: avg(
            results,
            |result| {
                if result.retrieval.recall_at_1 > 0.0 && result.answer_correct == Some(false) {
                    1.0
                } else {
                    0.0
                }
            },
            count,
        ),
        recall_at_1: avg(results, |result| result.retrieval.recall_at_1, count),
        recall_at_3: avg(results, |result| result.retrieval.recall_at_3, count),
        recall_at_5: avg(results, |result| result.retrieval.recall_at_5, count),
        recall_at_10: avg(results, |result| result.retrieval.recall_at_10, count),
        recall_at_20: avg(results, |result| result.retrieval.recall_at_20, count),
        recall_at_50: avg(results, |result| result.retrieval.recall_at_50, count),
        recall_at_100: avg(results, |result| result.retrieval.recall_at_100, count),
        recall_at_200: avg(results, |result| result.retrieval.recall_at_200, count),
        mrr: avg(results, |result| result.retrieval.mrr, count),
        ndcg_at_5: avg(results, |result| result.retrieval.ndcg_at_5, count),
        avg_latency_ms: results
            .iter()
            .map(|result| result.latency_ms as f32)
            .sum::<f32>()
            / count,
    }
}

fn recall_at(evidence: &BTreeSet<&String>, retrieved: &[String], k: usize) -> f32 {
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|source_id| evidence.contains(source_id))
        .collect::<BTreeSet<_>>()
        .len();
    (hits as f32 / evidence.len() as f32).min(1.0)
}

fn reciprocal_rank(evidence: &BTreeSet<&String>, retrieved: &[String]) -> f32 {
    retrieved
        .iter()
        .position(|source_id| evidence.contains(source_id))
        .map(|index| 1.0 / (index as f32 + 1.0))
        .unwrap_or(0.0)
}

fn ndcg_at(evidence: &BTreeSet<&String>, retrieved: &[String], k: usize) -> f32 {
    let mut seen = BTreeSet::new();
    let mut dcg = 0.0;
    for (index, source_id) in retrieved.iter().take(k).enumerate() {
        if evidence.contains(source_id) && seen.insert(source_id) {
            dcg += 1.0 / ((index + 2) as f32).log2();
        }
    }
    let ideal_hits = evidence.len().min(k);
    if ideal_hits == 0 {
        return 0.0;
    }
    let idcg = (0..ideal_hits)
        .map(|index| 1.0 / ((index + 2) as f32).log2())
        .sum::<f32>();
    dcg / idcg
}

fn avg(results: &[QuestionResult], value: impl Fn(&QuestionResult) -> f32, count: f32) -> f32 {
    results.iter().map(value).sum::<f32>() / count
}
