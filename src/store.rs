use std::fmt;

use crate::models::{Event, Memory, MemoryQuery};

pub type StoreResult<T> = Result<T, StoreError>;

#[derive(Debug)]
pub enum StoreError {
    Io(std::io::Error),
    Corrupt(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Io(error) => write!(f, "io error: {error}"),
            StoreError::Corrupt(message) => write!(f, "corrupt store: {message}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<std::io::Error> for StoreError {
    fn from(value: std::io::Error) -> Self {
        StoreError::Io(value)
    }
}

pub trait MemoryStore {
    fn add_event(&mut self, event: Event) -> StoreResult<Event>;
    fn add_memory(&mut self, memory: Memory) -> StoreResult<Memory>;
    fn update_memory(&mut self, memory: Memory) -> StoreResult<Memory>;
    fn delete_memory(&mut self, memory_id: &str) -> StoreResult<()>;
    fn get_memory(&self, memory_id: &str) -> StoreResult<Option<Memory>>;
    fn list_memories(&self, query: &MemoryQuery) -> StoreResult<Vec<Memory>>;
}
