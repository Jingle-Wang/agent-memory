use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub type Metadata = BTreeMap<String, String>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryType {
    Working,
    Episodic,
    Semantic,
    Procedural,
    Reflection,
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            MemoryType::Working => "working",
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Procedural => "procedural",
            MemoryType::Reflection => "reflection",
        };
        f.write_str(value)
    }
}

impl FromStr for MemoryType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "working" => Ok(MemoryType::Working),
            "episodic" => Ok(MemoryType::Episodic),
            "semantic" => Ok(MemoryType::Semantic),
            "procedural" => Ok(MemoryType::Procedural),
            "reflection" => Ok(MemoryType::Reflection),
            other => Err(format!("unknown memory type: {other}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub id: String,
    pub text: String,
    pub actor: String,
    pub namespace: String,
    pub metadata: Metadata,
    pub created_at: SystemTime,
}

impl Event {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: new_id(),
            text: text.into(),
            actor: "user".to_string(),
            namespace: "default".to_string(),
            metadata: Metadata::new(),
            created_at: SystemTime::now(),
        }
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    pub fn actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = actor.into();
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub namespace: String,
    pub source_event_id: Option<String>,
    pub importance: f32,
    pub confidence: f32,
    pub metadata: Metadata,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub valid_from: Option<SystemTime>,
    pub valid_to: Option<SystemTime>,
    pub deleted_at: Option<SystemTime>,
}

impl Memory {
    pub fn new(content: impl Into<String>, memory_type: MemoryType) -> Self {
        let now = SystemTime::now();
        Self {
            id: new_id(),
            content: content.into(),
            memory_type,
            namespace: "default".to_string(),
            source_event_id: None,
            importance: 0.5,
            confidence: 0.75,
            metadata: Metadata::new(),
            created_at: now,
            updated_at: now,
            valid_from: None,
            valid_to: None,
            deleted_at: None,
        }
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    pub fn source_event(mut self, event_id: impl Into<String>) -> Self {
        self.source_event_id = Some(event_id.into());
        self
    }

    pub fn importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryQuery {
    pub text: String,
    pub namespace: String,
    pub memory_types: Vec<MemoryType>,
    pub limit: usize,
    pub include_expired: bool,
}

impl MemoryQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            namespace: "default".to_string(),
            memory_types: Vec::new(),
            limit: 8,
            include_expired: false,
        }
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    pub fn memory_types(mut self, memory_types: Vec<MemoryType>) -> Self {
        self.memory_types = memory_types;
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPacket {
    pub memory: Memory,
    pub score: f32,
    pub reasons: Vec<String>,
}

pub fn new_id() -> String {
    let millis = epoch_millis(SystemTime::now());
    let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("mem_{millis:x}_{seq:x}")
}

pub fn epoch_seconds(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

pub fn epoch_millis(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
}

pub fn from_epoch_seconds(seconds: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(seconds)
}
