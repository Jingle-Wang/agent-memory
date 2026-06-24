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
    // Try exact date extraction first
    if let (Some(a), Some(g)) = (extract_date(answer), extract_date(gold)) {
        if a == g {
            return true;
        }
        // Fuzzy: dates within 2 days of each other
        if let (Some(a_days), Some(g_days)) = (date_to_epoch_days(a), date_to_epoch_days(g)) {
            if (a_days - g_days).abs() <= 2 {
                return true;
            }
        }
    }

    // Week range comparison: "week of June 2, 2023" vs "the week before 9 June 2023"
    if let (Some(a_week), Some(g_week)) = (extract_week_range(answer), extract_week_range(gold)) {
        if weeks_overlap(a_week, g_week) {
            return true;
        }
    }

    // Partial match: one side has a date range, the other has an exact date
    if let (Some(a_week), Some(g_date)) = (extract_week_range(answer), extract_date(gold)) {
        if let Some(g_days) = date_to_epoch_days(g_date) {
            if g_days >= date_to_epoch_days(a_week.0).unwrap_or(0)
                && g_days <= date_to_epoch_days(a_week.1).unwrap_or(0)
            {
                return true;
            }
        }
    }
    if let (Some(a_date), Some(g_week)) = (extract_date(answer), extract_week_range(gold)) {
        if let Some(a_days) = date_to_epoch_days(a_date) {
            if a_days >= date_to_epoch_days(g_week.0).unwrap_or(0)
                && a_days <= date_to_epoch_days(g_week.1).unwrap_or(0)
            {
                return true;
            }
        }
    }

    false
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

/// Convert a (year, month, day) tuple to approximate epoch days for comparison.
fn date_to_epoch_days((y, m, d): (u32, u32, u32)) -> Option<i64> {
    let y = y as i64;
    let m = m as i64;
    let d = d as i64;
    // Approximate: 365.25 days per year, 30.44 days per month
    Some((y * 365 + y / 4 - y / 100 + y / 400 + (m - 1) * 30 + d) as i64)
}

/// Extract a week range from a phrase like "week of June 2, 2023" or
/// "the week before 9 June 2023". Returns ((start_year, start_month, start_day),
/// (end_year, end_month, end_day)).
fn extract_week_range(value: &str) -> Option<((u32, u32, u32), (u32, u32, u32))> {
    let lower = value.to_lowercase();

    if lower.contains("week of") {
        let date = extract_date(value)?;
        // Week of <date>: the 7-day period containing that date.
        // Approximate: start = date, end = date + 6
        return build_week_range(date, 0);
    }

    if lower.contains("week before") {
        let date = extract_date(value)?;
        // Week before <date>: 7 days ending the day before <date>
        return build_week_range(date, -7);
    }

    if lower.contains("week after") {
        let date = extract_date(value)?;
        // Week after <date>: 7 days starting the day after <date>
        return build_week_range(date, 1);
    }

    None
}

/// Build a 7-day range anchored relative to a date.
/// offset: 0 = week containing date, -7 = week before, 1 = week after.
fn build_week_range(
    (y, m, d): (u32, u32, u32),
    offset: i32,
) -> Option<((u32, u32, u32), (u32, u32, u32))> {
    // For simplicity: start = date + offset, end = start + 6
    let start = shift_date_u32(y, m, d, offset)?;
    let end = shift_date_u32(start.0, start.1, start.2, 6)?;
    Some((start, end))
}

/// Shift a date by N days, operating on u32 tuples.
fn shift_date_u32(year: u32, month: u32, day: u32, offset_days: i32) -> Option<(u32, u32, u32)> {
    let mut y = year as i32;
    let mut m = month;
    let mut d = day as i32 + offset_days;
    while d < 1 {
        if m == 1 {
            m = 12;
            y -= 1;
        } else {
            m -= 1;
        }
        d += days_in_month_u32(y, m) as i32;
    }
    while d > days_in_month_u32(y, m) as i32 {
        d -= days_in_month_u32(y, m) as i32;
        if m == 12 {
            m = 1;
            y += 1;
        } else {
            m += 1;
        }
    }
    Some((y as u32, m, d as u32))
}

/// Check if two week ranges overlap (share at least one day).
fn weeks_overlap(
    a: ((u32, u32, u32), (u32, u32, u32)),
    b: ((u32, u32, u32), (u32, u32, u32)),
) -> bool {
    let a_start = date_to_epoch_days(a.0).unwrap_or(0);
    let a_end = date_to_epoch_days(a.1).unwrap_or(0);
    let b_start = date_to_epoch_days(b.0).unwrap_or(0);
    let b_end = date_to_epoch_days(b.1).unwrap_or(0);
    a_start <= b_end && b_start <= a_end
}

fn days_in_month_u32(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year_u32(year) => 29,
        2 => 28,
        _ => 30,
    }
}

fn is_leap_year_u32(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
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
