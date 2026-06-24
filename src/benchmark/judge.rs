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

// ── Date type aliases ────────────────────────────────────────────────────────

/// A date as (year, month, day).
type DateTriple = (u32, u32, u32);
/// A week range as (monday, sunday).
type WeekRange = (DateTriple, DateTriple);

// ── Date utilities ──────────────────────────────────────────────────────────

/// Check if a year is a leap year.
fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// Number of days in a given month (1-indexed).
fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Convert (year, month, day) to days since 0001-01-01.
fn date_to_days_since_epoch(year: u32, month: u32, day: u32) -> i64 {
    let mut days: i64 = 0;
    for y in 1..year as i64 {
        days += if is_leap_year(y as u32) { 366 } else { 365 };
    }
    for m in 1..month {
        days += days_in_month(year, m) as i64;
    }
    days + day as i64 - 1
}

/// Convert days since 0001-01-01 back to (year, month, day).
fn days_since_epoch_to_date(mut days: i64) -> (u32, u32, u32) {
    let mut year: u32 = 1;
    loop {
        let year_days: i64 = if is_leap_year(year) { 366 } else { 365 };
        if days < year_days {
            break;
        }
        days -= year_days;
        year += 1;
    }
    let mut month: u32 = 1;
    loop {
        let month_days = days_in_month(year, month) as i64;
        if days < month_days {
            break;
        }
        days -= month_days;
        month += 1;
    }
    (year, month, (days + 1) as u32)
}

/// Zeller's congruence: returns 0=Saturday, 1=Sunday, 2=Monday, …, 6=Friday.
fn weekday_zeller(year: u32, month: u32, day: u32) -> u32 {
    let (m, y) = if month <= 2 {
        (month + 12, year - 1)
    } else {
        (month, year)
    };
    let q = day;
    let k = y % 100;
    let j = y / 100;
    (q + (13 * (m + 1) / 5) + k + (k / 4) + (j / 4) - 2 * j) % 7
}

/// ISO weekday: 1=Monday … 7=Sunday.
fn weekday_iso(year: u32, month: u32, day: u32) -> u32 {
    let z = weekday_zeller(year, month, day);
    ((z + 5) % 7) + 1
}

/// Parse a weekday name to its ISO number (1=Mon … 7=Sun).
fn weekday_name_to_iso(name: &str) -> Option<u32> {
    match name {
        "monday" | "mon" => Some(1),
        "tuesday" | "tue" | "tues" => Some(2),
        "wednesday" | "wed" | "weds" => Some(3),
        "thursday" | "thu" | "thur" | "thurs" => Some(4),
        "friday" | "fri" => Some(5),
        "saturday" | "sat" => Some(6),
        "sunday" | "sun" => Some(7),
        _ => None,
    }
}

/// Extract a simple (year, month, day) from a normalized string.
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

/// Get a number adjacent to the given token index (offset -1 = left, +1 = right).
fn neighbor_number(tokens: &[&str], index: usize, offset: isize) -> Option<u32> {
    let neighbor = index.checked_add_signed(offset)?;
    let value = tokens.get(neighbor)?.parse::<u32>().ok()?;
    (1..=31).contains(&value).then_some(value)
}

/// Convert a month name to its 1-based number.
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

/// Check whether two dates are within `tolerance` days of each other.
fn dates_fuzzy_match(d1: (u32, u32, u32), d2: (u32, u32, u32), tolerance_days: u32) -> bool {
    let days1 = date_to_days_since_epoch(d1.0, d1.1, d1.2);
    let days2 = date_to_days_since_epoch(d2.0, d2.1, d2.2);
    (days1 - days2).unsigned_abs() <= tolerance_days as u64
}

/// Parse a relative-date expression like "friday before 15 july 2023".
///
/// Pattern: `<weekday> (before|after) <anchor-date>`
fn parse_relative_date(normalized: &str) -> Option<(u32, u32, u32)> {
    let tokens: Vec<&str> = normalized.split_whitespace().collect();

    // Find "before" or "after"
    let dir_pos = tokens.iter().position(|&t| t == "before" || t == "after")?;

    // Weekday must be immediately before the direction word
    let target_weekday = if dir_pos > 0 {
        weekday_name_to_iso(tokens[dir_pos - 1])?
    } else {
        return None;
    };

    let direction = tokens[dir_pos];

    // Anchor date is everything after the direction word
    let date_str = tokens[dir_pos + 1..].join(" ");
    let anchor = extract_date(&date_str)?;

    let anchor_dow = weekday_iso(anchor.0, anchor.1, anchor.2);

    let offset_days = if direction == "before" {
        // Days to go back from anchor to reach target weekday
        (anchor_dow as i32 - target_weekday as i32 + 7) % 7
    } else {
        // "after": days to go forward
        (target_weekday as i32 - anchor_dow as i32 + 7) % 7
    };

    let anchor_days = date_to_days_since_epoch(anchor.0, anchor.1, anchor.2);
    let result_days = if direction == "before" {
        anchor_days - offset_days as i64
    } else {
        anchor_days + offset_days as i64
    };

    Some(days_since_epoch_to_date(result_days))
}

/// Parse a week-range expression into ((start_date), (end_date)) where start is
/// Monday and end is Sunday.
///
/// Patterns:
/// - `week of <date>` — the Mon–Sun range containing the date
/// - `week before <date>` — the Mon–Sun range before the one containing the date
/// - `week after <date>` — the Mon–Sun range after the one containing the date
fn parse_week_range(normalized: &str) -> Option<WeekRange> {
    let tokens: Vec<&str> = normalized.split_whitespace().collect();

    let week_pos = tokens.iter().position(|&t| t == "week")?;

    // Determine shift: 0 = "week of", -1 = "week before", +1 = "week after"
    let (shift, date_start) = if week_pos + 1 < tokens.len() {
        match tokens[week_pos + 1] {
            "before" => (-1i32, week_pos + 2),
            "after" => (1, week_pos + 2),
            "of" => (0, week_pos + 2),
            _ => (0, week_pos + 1),
        }
    } else {
        (0, week_pos + 1)
    };

    let date_str = tokens[date_start..].join(" ");
    let anchor = extract_date(&date_str)?;

    let anchor_days = date_to_days_since_epoch(anchor.0, anchor.1, anchor.2);
    let shifted_days = anchor_days + (shift * 7) as i64;
    let shifted_date = days_since_epoch_to_date(shifted_days);

    // Find Monday of the week containing shifted_date
    let shifted_dow = weekday_iso(shifted_date.0, shifted_date.1, shifted_date.2);
    let days_to_monday = (shifted_dow as i64 - 1) % 7; // 0 if already Monday

    let monday_days = shifted_days - days_to_monday;
    let sunday_days = monday_days + 6;

    Some((
        days_since_epoch_to_date(monday_days),
        days_since_epoch_to_date(sunday_days),
    ))
}

/// Check whether two week ranges overlap.
fn week_ranges_overlap(r1: WeekRange, r2: WeekRange) -> bool {
    let start1 = date_to_days_since_epoch(r1.0.0, r1.0.1, r1.0.2);
    let end1 = date_to_days_since_epoch(r1.1.0, r1.1.1, r1.1.2);
    let start2 = date_to_days_since_epoch(r2.0.0, r2.0.1, r2.0.2);
    let end2 = date_to_days_since_epoch(r2.1.0, r2.1.1, r2.1.2);
    start1 <= end2 && start2 <= end1
}

/// Check whether a single date falls within a week range (inclusive).
fn date_in_week_range(date: DateTriple, week: WeekRange) -> bool {
    let d = date_to_days_since_epoch(date.0, date.1, date.2);
    let start = date_to_days_since_epoch(week.0.0, week.0.1, week.0.2);
    let end = date_to_days_since_epoch(week.1.0, week.1.1, week.1.2);
    d >= start && d <= end
}

// ── same_date ───────────────────────────────────────────────────────────────

/// Fuzzy date matching that handles:
/// - Simple dates (±2 day tolerance)
/// - Relative-date expressions ("Friday before 15 July 2023")
/// - Week-range expressions ("week of June 2, 2023")
fn same_date(answer: &str, gold: &str) -> bool {
    // Extract simple dates
    let a_simple = extract_date(answer);
    let g_simple = extract_date(gold);

    // Parse relative-date expressions
    let a_relative = parse_relative_date(answer);
    let g_relative = parse_relative_date(gold);

    // Parse week-range expressions
    let a_week = parse_week_range(answer);
    let g_week = parse_week_range(gold);

    // Collect all resolved exact dates for each side
    let a_dates: Vec<(u32, u32, u32)> = [a_simple, a_relative].into_iter().flatten().collect();
    let g_dates: Vec<(u32, u32, u32)> = [g_simple, g_relative].into_iter().flatten().collect();

    // 1. Date-to-date comparison with ±2 day tolerance
    for ad in &a_dates {
        for gd in &g_dates {
            if dates_fuzzy_match(*ad, *gd, 2) {
                return true;
            }
        }
    }

    // 2. Date falls within week range
    for ad in &a_dates {
        if let Some(gw) = &g_week
            && date_in_week_range(*ad, *gw)
        {
            return true;
        }
    }
    for gd in &g_dates {
        if let Some(aw) = &a_week
            && date_in_week_range(*gd, *aw)
        {
            return true;
        }
    }

    // 3. Week range overlap
    if let (Some(aw), Some(gw)) = (&a_week, &g_week)
        && week_ranges_overlap(*aw, *gw)
    {
        return true;
    }

    false
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::dataset::QuestionForJudge;

    // ── Date utility tests ───────────────────────────────────────────────

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2023));
    }

    #[test]
    fn test_days_in_month() {
        assert_eq!(days_in_month(2023, 1), 31);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 4), 30);
        assert_eq!(days_in_month(2023, 7), 31);
    }

    #[test]
    fn test_date_roundtrip() {
        let cases = [
            (2023, 7, 15),
            (2023, 1, 1),
            (2023, 12, 31),
            (2024, 2, 29),
            (2000, 3, 1),
            (1999, 12, 31),
        ];
        for (y, m, d) in cases {
            let days = date_to_days_since_epoch(y, m, d);
            let (y2, m2, d2) = days_since_epoch_to_date(days);
            assert_eq!(
                (y2, m2, d2),
                (y, m, d),
                "roundtrip failed for {y}-{m}-{d}: got {y2}-{m2}-{d2} (days={days})"
            );
        }
    }

    #[test]
    fn test_date_to_days_known_intervals() {
        // 2023-01-01 to 2023-07-15 = 195 days
        let jan1 = date_to_days_since_epoch(2023, 1, 1);
        let jul15 = date_to_days_since_epoch(2023, 7, 15);
        assert_eq!(jul15 - jan1, 195);
    }

    // ── Weekday (Zeller) tests ───────────────────────────────────────────

    #[test]
    fn test_weekday_zeller_known_dates() {
        // July 15, 2023 = Saturday → Zeller h=0
        assert_eq!(weekday_zeller(2023, 7, 15), 0);
        // July 14, 2023 = Friday → Zeller h=6
        assert_eq!(weekday_zeller(2023, 7, 14), 6);
        // May 25, 2023 = Thursday → Zeller h=5
        assert_eq!(weekday_zeller(2023, 5, 25), 5);
        // January 1, 2023 = Sunday → Zeller h=1
        assert_eq!(weekday_zeller(2023, 1, 1), 1);
        // March 1, 2023 = Wednesday → Zeller h=4
        assert_eq!(weekday_zeller(2023, 3, 1), 4);
    }

    #[test]
    fn test_weekday_iso_conversion() {
        // Saturday → 6
        assert_eq!(weekday_iso(2023, 7, 15), 6);
        // Sunday → 7
        assert_eq!(weekday_iso(2023, 1, 1), 7);
        // Monday → 1
        assert_eq!(weekday_iso(2023, 1, 2), 1);
        // Thursday → 4
        assert_eq!(weekday_iso(2023, 5, 25), 4);
        // Friday → 5
        assert_eq!(weekday_iso(2023, 7, 14), 5);
    }

    // ── Fuzzy date match tests ───────────────────────────────────────────

    #[test]
    fn test_dates_fuzzy_match_exact() {
        assert!(dates_fuzzy_match((2023, 7, 15), (2023, 7, 15), 2));
    }

    #[test]
    fn test_dates_fuzzy_match_within_tolerance() {
        // Same day → within 2
        assert!(dates_fuzzy_match((2023, 7, 14), (2023, 7, 15), 2));
        assert!(dates_fuzzy_match((2023, 7, 13), (2023, 7, 15), 2));
    }

    #[test]
    fn test_dates_fuzzy_match_outside_tolerance() {
        assert!(!dates_fuzzy_match((2023, 7, 12), (2023, 7, 15), 2));
        assert!(!dates_fuzzy_match((2023, 7, 18), (2023, 7, 15), 2));
    }

    #[test]
    fn test_dates_fuzzy_match_month_boundary() {
        // July 31 ↔ August 2 (2 days apart across month boundary)
        assert!(dates_fuzzy_match((2023, 7, 31), (2023, 8, 2), 2));
    }

    // ── Relative date parsing tests ──────────────────────────────────────

    #[test]
    fn test_parse_relative_date_friday_before_july15() {
        // "friday before 15 july 2023" → July 14, 2023
        let result = parse_relative_date("friday before 15 july 2023");
        assert_eq!(result, Some((2023, 7, 14)));
    }

    #[test]
    fn test_parse_relative_date_monday_after_july15() {
        // July 15, 2023 = Saturday, Monday after = July 17
        let result = parse_relative_date("monday after 15 july 2023");
        assert_eq!(result, Some((2023, 7, 17)));
    }

    #[test]
    fn test_parse_relative_date_same_weekday() {
        // "friday before 14 july 2023" → Friday before Friday = same day
        let result = parse_relative_date("friday before 14 july 2023");
        assert_eq!(result, Some((2023, 7, 14)));
    }

    #[test]
    fn test_parse_relative_date_cross_month() {
        // "monday before 1 august 2023" → July 31, 2023
        // Aug 1, 2023 = Tuesday, Monday before = July 31
        let result = parse_relative_date("monday before 1 august 2023");
        assert_eq!(result, Some((2023, 7, 31)));
    }

    #[test]
    fn test_parse_relative_date_sunday_before_may21() {
        // Preceding Sunday before May 25, 2023 (Thursday) → May 21
        let result = parse_relative_date("sunday before 25 may 2023");
        assert_eq!(result, Some((2023, 5, 21)));
    }

    // ── Week range parsing tests ─────────────────────────────────────────

    #[test]
    fn test_parse_week_range_week_of() {
        // June 2, 2023 = Friday → week Mon May 29 – Sun June 4
        let result = parse_week_range("week of june 2 2023");
        assert_eq!(result, Some(((2023, 5, 29), (2023, 6, 4))));
    }

    #[test]
    fn test_parse_week_range_week_before() {
        // "week before 9 june 2023" → previous week = Mon May 29 – Sun June 4
        let result = parse_week_range("week before 9 june 2023");
        assert_eq!(result, Some(((2023, 5, 29), (2023, 6, 4))));
    }

    #[test]
    fn test_parse_week_range_week_after() {
        // "week after 2 june 2023" → next week = Mon June 5 – Sun June 11
        let result = parse_week_range("week after 2 june 2023");
        assert_eq!(result, Some(((2023, 6, 5), (2023, 6, 11))));
    }

    #[test]
    fn test_parse_week_range_sunday_anchor() {
        // June 4, 2023 = Sunday → week Mon May 29 – Sun June 4 (same)
        let result = parse_week_range("week of 4 june 2023");
        assert_eq!(result, Some(((2023, 5, 29), (2023, 6, 4))));
    }

    #[test]
    fn test_parse_week_range_monday_anchor() {
        // May 29, 2023 = Monday → week Mon May 29 – Sun June 4
        let result = parse_week_range("week of 29 may 2023");
        assert_eq!(result, Some(((2023, 5, 29), (2023, 6, 4))));
    }

    // ── Week range overlap tests ─────────────────────────────────────────

    #[test]
    fn test_week_ranges_overlap_same_week() {
        let r = ((2023, 5, 29), (2023, 6, 4));
        assert!(week_ranges_overlap(r, r));
    }

    #[test]
    fn test_week_ranges_overlap_adjacent_weeks() {
        // May 29–June 4 and June 5–11 are adjacent (no overlap)
        let r1 = ((2023, 5, 29), (2023, 6, 4));
        let r2 = ((2023, 6, 5), (2023, 6, 11));
        assert!(!week_ranges_overlap(r1, r2));
    }

    #[test]
    fn test_week_ranges_overlap_separate() {
        let r1 = ((2023, 5, 29), (2023, 6, 4));
        let r2 = ((2023, 6, 12), (2023, 6, 18));
        assert!(!week_ranges_overlap(r1, r2));
    }

    // ── Integrated same_date tests ───────────────────────────────────────

    #[test]
    fn test_same_date_simple_exact() {
        assert!(same_date("14 july 2023", "14 july 2023"));
    }

    #[test]
    fn test_same_date_fuzzy_2day() {
        // ±2 day tolerance
        assert!(same_date("14 july 2023", "15 july 2023"));
        assert!(same_date("14 july 2023", "16 july 2023"));
    }

    #[test]
    fn test_same_date_fuzzy_outside() {
        assert!(!same_date("14 july 2023", "17 july 2023"));
    }

    #[test]
    fn test_same_date_relative_friday_before() {
        // LLM: "14 July 2023" vs Gold: "The Friday before 15 July 2023"
        assert!(same_date("14 july 2023", "friday before 15 july 2023"));
    }

    #[test]
    fn test_same_date_relative_monday_after() {
        assert!(same_date("17 july 2023", "monday after 15 july 2023"));
    }

    #[test]
    fn test_same_date_week_overlap() {
        // "week of June 2, 2023" vs "The week before 9 June 2023" → SAME WEEK
        assert!(same_date("week of june 2 2023", "week before 9 june 2023"));
    }

    #[test]
    fn test_same_date_date_in_week() {
        // A specific date that falls inside a gold week range
        assert!(same_date("31 may 2023", "week of june 2 2023"));
    }

    #[test]
    fn test_same_date_no_match() {
        assert!(!same_date("1 january 2023", "15 july 2023"));
    }

    // ── Existing judge integration tests ─────────────────────────────────

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

    #[test]
    fn normalized_judge_relative_date_gold() {
        // Regression: LLM says "14 July 2023", gold says "The Friday before 15 July 2023"
        let judge = NormalizedContainsJudge;
        let output = judge.judge(&JudgeInput {
            question: QuestionForJudge {
                id: "q".to_string(),
                gold_answers: vec!["The Friday before 15 July 2023".to_string()],
                evidence_turn_ids: Vec::new(),
            },
            answer: AnswerOutput {
                answer: "14 July 2023".to_string(),
            },
        });
        assert!(
            output.correct,
            "Expected match: 14 July 2023 == Friday before 15 July 2023"
        );
    }

    #[test]
    fn normalized_judge_week_overlap() {
        // Regression: LLM says "week of June 2, 2023", gold says "The week before 9 June 2023"
        let judge = NormalizedContainsJudge;
        let output = judge.judge(&JudgeInput {
            question: QuestionForJudge {
                id: "q".to_string(),
                gold_answers: vec!["The week before 9 June 2023".to_string()],
                evidence_turn_ids: Vec::new(),
            },
            answer: AnswerOutput {
                answer: "week of June 2, 2023".to_string(),
            },
        });
        assert!(
            output.correct,
            "Expected match: week of June 2 == week before June 9"
        );
    }

    #[test]
    fn normalized_judge_fuzzy_date_answer() {
        // LLM off by 1–2 days should still match
        let judge = NormalizedContainsJudge;
        let output = judge.judge(&JudgeInput {
            question: QuestionForJudge {
                id: "q".to_string(),
                gold_answers: vec!["15 July 2023".to_string()],
                evidence_turn_ids: Vec::new(),
            },
            answer: AnswerOutput {
                answer: "14 July 2023".to_string(),
            },
        });
        assert!(output.correct, "±2 day tolerance should accept 14≈15 July");
    }

    #[test]
    fn normalized_judge_still_rejects_unrelated_dates() {
        let judge = NormalizedContainsJudge;
        let output = judge.judge(&JudgeInput {
            question: QuestionForJudge {
                id: "q".to_string(),
                gold_answers: vec!["15 July 2023".to_string()],
                evidence_turn_ids: Vec::new(),
            },
            answer: AnswerOutput {
                answer: "1 January 2020".to_string(),
            },
        });
        assert!(!output.correct, "Unrelated dates should still be rejected");
    }
}
