//! Lightweight local-first memory for AI agents.
//!
//! The crate is intentionally std-only for the first implementation. It gives
//! edge runtimes a small durable memory layer while keeping the storage trait
//! open for SQLite, RocksDB, Postgres, or cloud object-store backends.

pub mod embedding;
pub mod engine;
pub mod file_store;
pub mod models;
pub mod retriever;
pub mod store;

#[cfg(feature = "sqlite")]
pub mod sqlite_store;

pub use engine::MemoryEngine;
pub use file_store::FileMemoryStore;
pub use models::{Event, Memory, MemoryPacket, MemoryQuery, MemoryType};
pub use retriever::HybridMemoryRetriever;
pub use store::{MemoryStore, StoreResult};

#[cfg(feature = "sqlite")]
pub use sqlite_store::SqliteMemoryStore;
