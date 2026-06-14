use serde::{Deserialize, Serialize};

use super::answerer::AnswerOutput;
use super::dataset::QuestionForJudge;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeInput {
    pub question: QuestionForJudge,
    pub answer: AnswerOutput,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JudgeOutput {
    pub correct: bool,
    pub score: f32,
    pub reason: String,
}

pub trait Judge {
    fn judge(&self, input: &JudgeInput) -> JudgeOutput;
}

#[derive(Clone, Debug, Default)]
pub struct ExactContainsJudge;

impl Judge for ExactContainsJudge {
    fn judge(&self, input: &JudgeInput) -> JudgeOutput {
        let answer = normalize(&input.answer.answer);
        for gold in &input.question.gold_answers {
            let gold = normalize(gold);
            if !gold.is_empty() && (answer.contains(&gold) || gold.contains(&answer)) {
                return JudgeOutput {
                    correct: true,
                    score: 1.0,
                    reason: "answer contains a gold answer string".to_string(),
                };
            }
        }
        JudgeOutput {
            correct: false,
            score: 0.0,
            reason: "no gold answer string matched".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NormalizedContainsJudge;

impl Judge for NormalizedContainsJudge {
    fn judge(&self, input: &JudgeInput) -> JudgeOutput {
        let answer = normalize_for_fuzzy_match(&input.answer.answer);
        for gold in &input.question.gold_answers {
            let gold = normalize_for_fuzzy_match(gold);
            if !gold.is_empty()
                && (answer.contains(&gold) || gold.contains(&answer) || same_date(&answer, &gold))
            {
                return JudgeOutput {
                    correct: true,
                    score: 1.0,
                    reason: "normalized answer contains a gold answer string".to_string(),
                };
            }
        }
        JudgeOutput {
            correct: false,
            score: 0.0,
            reason: "no normalized gold answer string matched".to_string(),
        }
    }
}

fn normalize(value: &str) -> String {
    value
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_for_fuzzy_match(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|token| !matches!(*token, "a" | "an" | "the"))
        .map(normalize_number_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_number_token(token: &str) -> String {
    let ordinal = token
        .strip_suffix("st")
        .or_else(|| token.strip_suffix("nd"))
        .or_else(|| token.strip_suffix("rd"))
        .or_else(|| token.strip_suffix("th"))
        .unwrap_or(token);
    match ordinal {
        "zero" => "0",
        "one" => "1",
        "two" => "2",
        "three" => "3",
        "four" => "4",
        "five" => "5",
        "six" => "6",
        "seven" => "7",
        "eight" => "8",
        "nine" => "9",
        "ten" => "10",
        "eleven" => "11",
        "twelve" => "12",
        "thirteen" => "13",
        "fourteen" => "14",
        "fifteen" => "15",
        "sixteen" => "16",
        "seventeen" => "17",
        "eighteen" => "18",
        "nineteen" => "19",
        "twenty" => "20",
        _ => ordinal,
    }
    .to_string()
}

fn same_date(answer: &str, gold: &str) -> bool {
    let Some(answer_date) = extract_date(answer) else {
        return false;
    };
    let Some(gold_date) = extract_date(gold) else {
        return false;
    };
    answer_date == gold_date
}

fn extract_date(value: &str) -> Option<(u32, u32, u32)> {
    let tokens = value.split_whitespace().collect::<Vec<_>>();
    let month_index = tokens
        .iter()
        .position(|token| month_number(token).is_some())?;
    let month = month_number(tokens[month_index])?;
    let day = neighbor_number(&tokens, month_index, -1)
        .or_else(|| neighbor_number(&tokens, month_index, 1))?;
    let year = tokens
        .iter()
        .filter_map(|token| token.parse::<u32>().ok())
        .find(|number| (1900..=2100).contains(number))?;
    Some((year, month, day))
}

fn neighbor_number(tokens: &[&str], index: usize, offset: isize) -> Option<u32> {
    let neighbor = index.checked_add_signed(offset)?;
    let value = tokens.get(neighbor)?.parse::<u32>().ok()?;
    (1..=31).contains(&value).then_some(value)
}

fn month_number(token: &str) -> Option<u32> {
    match token {
        "january" | "jan" => Some(1),
        "february" | "feb" => Some(2),
        "march" | "mar" => Some(3),
        "april" | "apr" => Some(4),
        "may" => Some(5),
        "june" | "jun" => Some(6),
        "july" | "jul" => Some(7),
        "august" | "aug" => Some(8),
        "september" | "sep" | "sept" => Some(9),
        "october" | "oct" => Some(10),
        "november" | "nov" => Some(11),
        "december" | "dec" => Some(12),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::dataset::QuestionForJudge;

    #[test]
    fn normalized_judge_accepts_equivalent_month_day_order() {
        let judge = NormalizedContainsJudge;
        let output = judge.judge(&JudgeInput {
            question: QuestionForJudge {
                id: "q".to_string(),
                gold_answers: vec!["2 July, 2023".to_string()],
                evidence_turn_ids: Vec::new(),
            },
            answer: AnswerOutput {
                answer: "July 2, 2023".to_string(),
            },
        });

        assert!(output.correct);
    }
}
