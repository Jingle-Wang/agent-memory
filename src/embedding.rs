use std::collections::BTreeSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric() && ch != '\'')
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

#[derive(Clone, Debug)]
pub struct HashEmbedding {
    dimensions: usize,
}

impl HashEmbedding {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(8),
        }
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; self.dimensions];
        for token in tokenize(text) {
            let mut hasher = DefaultHasher::new();
            token.hash(&mut hasher);
            let hash = hasher.finish();
            let bucket = hash as usize % self.dimensions;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            vector[bucket] += sign;
        }
        normalize(vector)
    }
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub fn token_overlap_score(left: &str, right: &str) -> f32 {
    let left_tokens: BTreeSet<_> = tokenize(left).into_iter().collect();
    let right_tokens: BTreeSet<_> = tokenize(right).into_iter().collect();
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }
    let intersection = left_tokens.intersection(&right_tokens).count() as f32;
    let union = left_tokens.union(&right_tokens).count() as f32;
    intersection / union
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}
