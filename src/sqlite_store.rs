use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;
use std::time::SystemTime;

use rusqlite::{Connection, params};

use crate::models::{Event, Memory, MemoryQuery, MemoryType, epoch_seconds, from_epoch_seconds};
use crate::store::{MemoryStore, StoreError, StoreResult};

/// Convert SystemTime → i64 seconds for SQLite storage.
/// epoch_seconds returns u64, but SQLite only supports i64.
fn to_i64_secs(t: SystemTime) -> i64 {
    epoch_seconds(t) as i64
}

/// Convert Option<SystemTime> → Option<i64> for SQLite storage.
fn opt_to_i64_secs(t: Option<SystemTime>) -> Option<i64> {
    t.map(to_i64_secs)
}

/// Convert i64 seconds from SQLite → SystemTime.
fn from_i64_secs(s: i64) -> SystemTime {
    from_epoch_seconds(s as u64)
}

/// Convert Option<i64> from SQLite → Option<SystemTime>.
fn opt_from_i64_secs(s: Option<i64>) -> Option<SystemTime> {
    s.map(from_i64_secs)
}

#[derive(Debug)]
pub struct SqliteMemoryStore {
    conn: Connection,
}

impl SqliteMemoryStore {
    /// Open (or create) a SQLite-backed memory store at the given path.
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let conn = Connection::open(path).map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Open an in-memory SQLite store (useful for testing).
    pub fn open_in_memory() -> StoreResult<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> StoreResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS events (
                    id          TEXT PRIMARY KEY,
                    namespace   TEXT NOT NULL DEFAULT 'default',
                    actor       TEXT NOT NULL DEFAULT 'user',
                    text        TEXT NOT NULL,
                    metadata    TEXT NOT NULL DEFAULT '',
                    created_at  INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id              TEXT PRIMARY KEY,
                    namespace       TEXT NOT NULL DEFAULT 'default',
                    memory_type     TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    source_event_id TEXT,
                    importance      REAL NOT NULL DEFAULT 0.5,
                    confidence      REAL NOT NULL DEFAULT 0.75,
                    metadata        TEXT NOT NULL DEFAULT '',
                    created_at      INTEGER NOT NULL,
                    updated_at      INTEGER NOT NULL,
                    valid_from      INTEGER,
                    valid_to        INTEGER,
                    deleted_at      INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_events_namespace ON events(namespace);
                CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                ",
            )
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
        Ok(())
    }

    /// List all events in a namespace (mirrors FileMemoryStore::list_events).
    pub fn list_events(&self, namespace: &str) -> Vec<Event> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, namespace, actor, text, metadata, created_at
                 FROM events WHERE namespace = ?1",
            )
            .expect("prepare list_events");
        let rows = stmt
            .query_map(params![namespace], |row| Ok(row_to_event(row)))
            .expect("query list_events");
        rows.filter_map(|r| r.ok()).collect()
    }
}

fn row_to_event(row: &rusqlite::Row<'_>) -> Event {
    let id: String = row.get(0).unwrap_or_default();
    let namespace: String = row.get(1).unwrap_or_default();
    let actor: String = row.get(2).unwrap_or_default();
    let text: String = row.get(3).unwrap_or_default();
    let metadata_str: String = row.get(4).unwrap_or_default();
    let created_at_secs: i64 = row.get(5).unwrap_or(0);
    Event {
        id,
        namespace,
        actor,
        text,
        metadata: decode_metadata(&metadata_str).unwrap_or_default(),
        created_at: from_i64_secs(created_at_secs),
    }
}

fn row_to_memory(row: &rusqlite::Row<'_>) -> Memory {
    let id: String = row.get(0).unwrap_or_default();
    let namespace: String = row.get(1).unwrap_or_default();
    let memory_type_str: String = row.get(2).unwrap_or_default();
    let content: String = row.get(3).unwrap_or_default();
    let source_event_id: Option<String> = row.get(4).unwrap_or(None);
    let importance: f32 = row.get::<_, f64>(5).unwrap_or(0.5) as f32;
    let confidence: f32 = row.get::<_, f64>(6).unwrap_or(0.75) as f32;
    let metadata_str: String = row.get(7).unwrap_or_default();
    let created_at_secs: i64 = row.get(8).unwrap_or(0);
    let updated_at_secs: i64 = row.get(9).unwrap_or(0);
    let valid_from_secs: Option<i64> = row.get(10).unwrap_or(None);
    let valid_to_secs: Option<i64> = row.get(11).unwrap_or(None);
    let deleted_at_secs: Option<i64> = row.get(12).unwrap_or(None);

    Memory {
        id,
        namespace,
        memory_type: MemoryType::from_str(&memory_type_str).unwrap_or(MemoryType::Semantic),
        content,
        source_event_id,
        importance,
        confidence,
        metadata: decode_metadata(&metadata_str).unwrap_or_default(),
        created_at: from_i64_secs(created_at_secs),
        updated_at: from_i64_secs(updated_at_secs),
        valid_from: opt_from_i64_secs(valid_from_secs),
        valid_to: opt_from_i64_secs(valid_to_secs),
        deleted_at: opt_from_i64_secs(deleted_at_secs),
    }
}

impl MemoryStore for SqliteMemoryStore {
    fn add_event(&mut self, event: Event) -> StoreResult<Event> {
        let metadata_str = encode_metadata(&event.metadata);
        let created_at_secs = to_i64_secs(event.created_at);
        self.conn
            .execute(
                "INSERT INTO events (id, namespace, actor, text, metadata, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    event.id,
                    event.namespace,
                    event.actor,
                    event.text,
                    metadata_str,
                    created_at_secs,
                ],
            )
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
        Ok(event)
    }

    fn add_memory(&mut self, memory: Memory) -> StoreResult<Memory> {
        let metadata_str = encode_metadata(&memory.metadata);
        let created_at_secs = to_i64_secs(memory.created_at);
        let updated_at_secs = to_i64_secs(memory.updated_at);
        let valid_from_secs = opt_to_i64_secs(memory.valid_from);
        let valid_to_secs = opt_to_i64_secs(memory.valid_to);

        self.conn
            .execute(
                "INSERT INTO memories
                 (id, namespace, memory_type, content, source_event_id,
                  importance, confidence, metadata, created_at, updated_at,
                  valid_from, valid_to, deleted_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    memory.id,
                    memory.namespace,
                    memory.memory_type.to_string(),
                    memory.content,
                    memory.source_event_id,
                    memory.importance as f64,
                    memory.confidence as f64,
                    metadata_str,
                    created_at_secs,
                    updated_at_secs,
                    valid_from_secs,
                    valid_to_secs,
                    Option::<i64>::None,
                ],
            )
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
        Ok(memory)
    }

    fn update_memory(&mut self, mut memory: Memory) -> StoreResult<Memory> {
        memory.updated_at = SystemTime::now();
        let metadata_str = encode_metadata(&memory.metadata);
        let updated_at_secs = to_i64_secs(memory.updated_at);
        let valid_from_secs = opt_to_i64_secs(memory.valid_from);
        let valid_to_secs = opt_to_i64_secs(memory.valid_to);

        self.conn
            .execute(
                "UPDATE memories SET
                 namespace = ?2, memory_type = ?3, content = ?4,
                 source_event_id = ?5, importance = ?6, confidence = ?7,
                 metadata = ?8, updated_at = ?9, valid_from = ?10, valid_to = ?11
                 WHERE id = ?1",
                params![
                    memory.id,
                    memory.namespace,
                    memory.memory_type.to_string(),
                    memory.content,
                    memory.source_event_id,
                    memory.importance as f64,
                    memory.confidence as f64,
                    metadata_str,
                    updated_at_secs,
                    valid_from_secs,
                    valid_to_secs,
                ],
            )
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
        Ok(memory)
    }

    fn delete_memory(&mut self, memory_id: &str) -> StoreResult<()> {
        let deleted_at_secs = to_i64_secs(SystemTime::now());
        self.conn
            .execute(
                "UPDATE memories SET deleted_at = ?1 WHERE id = ?2",
                params![deleted_at_secs, memory_id],
            )
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
        Ok(())
    }

    fn get_memory(&self, memory_id: &str) -> StoreResult<Option<Memory>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, namespace, memory_type, content, source_event_id,
                        importance, confidence, metadata, created_at, updated_at,
                        valid_from, valid_to, deleted_at
                 FROM memories WHERE id = ?1 AND deleted_at IS NULL",
            )
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;

        let result = stmt
            .query_row(params![memory_id], |row| Ok(row_to_memory(row)))
            .ok();
        Ok(result)
    }

    fn list_memories(&self, query: &MemoryQuery) -> StoreResult<Vec<Memory>> {
        let now_secs = to_i64_secs(SystemTime::now());

        let mut sql = String::from(
            "SELECT id, namespace, memory_type, content, source_event_id,
                    importance, confidence, metadata, created_at, updated_at,
                    valid_from, valid_to, deleted_at
             FROM memories WHERE namespace = ?1 AND deleted_at IS NULL",
        );

        let mut param_index = 2u32;
        let mut type_filters: Vec<String> = Vec::new();

        if !query.memory_types.is_empty() {
            for _mt in &query.memory_types {
                type_filters.push(format!("memory_type = ?{param_index}"));
                param_index += 1;
            }
            sql.push_str(" AND (");
            sql.push_str(&type_filters.join(" OR "));
            sql.push(')');
        }

        if !query.include_expired {
            sql.push_str(&format!(
                " AND (valid_from IS NULL OR valid_from <= ?{param_index})"
            ));
            param_index += 1;
            sql.push_str(&format!(
                " AND (valid_to IS NULL OR valid_to >= ?{param_index})"
            ));
            param_index += 1;
        }

        // Add LIMIT at SQL level to avoid fetching all rows (O(n²) when
        // called repeatedly during ingestion via find_duplicate).
        sql.push_str(&format!(" ORDER BY updated_at DESC LIMIT ?{param_index}"));

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;

        // Build parameter list
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(query.namespace.clone())];
        for mt in &query.memory_types {
            param_values.push(Box::new(mt.to_string()));
        }
        if !query.include_expired {
            param_values.push(Box::new(now_secs));
            param_values.push(Box::new(now_secs));
        }
        param_values.push(Box::new(query.limit as i64));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let rows = stmt
            .query_map(param_refs.as_slice(), |row| Ok(row_to_memory(row)))
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;

        let results: Vec<Memory> = rows
            .filter_map(|r| r.ok())
            .filter(|memory| query.include_side_channel || !memory.is_side_channel())
            .collect();
        // LIMIT is already applied in SQL; truncate is a safety net.
        // REMOVED: results.truncate(query.limit) — the retriever layer handles Top-K.
        // Keeping truncation here would silently discard gold evidence from earlier sessions
        // when benchmark sets query.limit to a small value (e.g. 10). See architecture-review.md P0#1.
        Ok(results)
    }
}

// --- Metadata encoding (same hex format as FileMemoryStore) ---

fn encode_metadata(metadata: &BTreeMap<String, String>) -> String {
    metadata
        .iter()
        .map(|(key, value)| format!("{}={}", hex_encode(key), hex_encode(value)))
        .collect::<Vec<_>>()
        .join(";")
}

fn decode_metadata(value: &str) -> Result<BTreeMap<String, String>, String> {
    let mut metadata = BTreeMap::new();
    if value.is_empty() {
        return Ok(metadata);
    }
    for pair in value.split(';') {
        let Some((key, val)) = pair.split_once('=') else {
            return Err("invalid metadata pair".to_string());
        };
        metadata.insert(hex_decode(key)?, hex_decode(val)?);
    }
    Ok(metadata)
}

fn hex_encode(value: &str) -> String {
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn hex_decode(value: &str) -> Result<String, String> {
    if !value.len().is_multiple_of(2) {
        return Err("invalid hex length".to_string());
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let mut index = 0;
    while index < value.len() {
        let byte = u8::from_str_radix(&value[index..index + 2], 16)
            .map_err(|_| "invalid hex metadata".to_string())?;
        bytes.push(byte);
        index += 2;
    }
    String::from_utf8(bytes).map_err(|_| "metadata is not utf-8".to_string())
}
