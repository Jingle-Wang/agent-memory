use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    pub name: String,
    pub version: String,
    pub conversations: Vec<Conversation>,
    pub questions: Vec<BenchmarkQuestion>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub turns: Vec<BenchmarkTurn>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkTurn {
    pub id: String,
    pub speaker: String,
    pub text: String,
    pub timestamp: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkQuestion {
    pub id: String,
    pub conversation_id: String,
    pub text: String,
    pub gold_answers: Vec<String>,
    pub evidence_turn_ids: Vec<String>,
    pub category: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionForAnswerer {
    pub id: String,
    pub conversation_id: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionForJudge {
    pub id: String,
    pub gold_answers: Vec<String>,
    pub evidence_turn_ids: Vec<String>,
}

impl BenchmarkQuestion {
    pub fn for_answerer(&self) -> QuestionForAnswerer {
        QuestionForAnswerer {
            id: self.id.clone(),
            conversation_id: self.conversation_id.clone(),
            text: self.text.clone(),
        }
    }

    pub fn for_judge(&self) -> QuestionForJudge {
        QuestionForJudge {
            id: self.id.clone(),
            gold_answers: self.gold_answers.clone(),
            evidence_turn_ids: self.evidence_turn_ids.clone(),
        }
    }
}
