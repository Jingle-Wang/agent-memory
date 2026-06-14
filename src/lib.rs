//! Lightweight local-first memory for AI agents.
//!
//! The crate is intentionally std-only for the first implementation. It gives
//! edge runtimes a small durable memory layer while keeping the storage trait
//! open for SQLite, RocksDB, Postgres, or cloud object-store backends.

pub mod embedding;
pub mod engine;
pub mod entity;
pub mod extractor;
pub mod file_store;
pub mod ingestion_buffer;
pub mod llm;
pub mod models;
pub mod observation;
pub mod retriever;
pub mod store;
pub mod text;

#[cfg(feature = "sqlite")]
pub mod sqlite_store;

pub mod volatile_store;

#[cfg(feature = "benchmark")]
pub mod benchmark;

pub use embedding::EmbeddingProvider;
#[cfg(feature = "embed-ollama")]
pub use embedding::OllamaEmbeddingProvider;
pub use engine::DEFAULT_INGESTION_WINDOW;
pub use engine::MemoryEngine;
pub use extractor::{LlmMemoryExtractor, MemoryExtractor, RuleBasedMemoryExtractor};
pub use file_store::FileMemoryStore;
pub use ingestion_buffer::IngestionBuffer;
pub use llm::{ConfiguredLlmProvider, LlmProvider, LlmProviderConfig};
pub use models::{Event, Memory, MemoryPacket, MemoryQuery, MemoryType};
pub use retriever::HybridMemoryRetriever;
pub use store::{MemoryStore, StoreResult};
pub use volatile_store::VolatileMemoryStore;

#[cfg(feature = "sqlite")]
pub use sqlite_store::SqliteMemoryStore;
