# SqliteMemoryStore 实现计划

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task via Codex.

**Goal:** 为 agent-memory 添加 SqliteMemoryStore，实现与 FileMemoryStore 相同的 MemoryStore trait，提供更好的查询性能和并发支持。

**Architecture:** 新增 `src/sqlite_store.rs` 模块，使用 rusqlite 实现 MemoryStore trait。数据库采用单文件存储，事件和记忆分表管理，保留 append-only 语义。通过 feature flag `sqlite` 控制是否编译。

**Tech Stack:** Rust, rusqlite (bundled sqlite), MemoryStore trait (已有)

---

### Task 1: 添加 rusqlite 依赖和 feature flag

**Objective:** 在 Cargo.toml 中添加 rusqlite 依赖，通过 feature flag 控制

**Files:**
- Modify: `Cargo.toml`

**Step 1: 修改 Cargo.toml**

在 `[dependencies]` 下添加：
```toml
rusqlite = { version = "0.31", features = ["bundled"], optional = true }
```

在文件末尾添加 feature：
```toml
[features]
sqlite = ["rusqlite"]
```

**Step 2: 验证编译**

Run: `cargo check`
Expected: 编译成功（rusqlite 是 optional，默认不编译）

**Step 3: 验证 feature 编译**

Run: `cargo check --features sqlite`
Expected: 编译成功，rusqlite 被拉入

**Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "feat: add rusqlite dependency behind sqlite feature flag"
```

---

### Task 2: 创建 SqliteMemoryStore 骨架和数据库 schema

**Objective:** 创建 src/sqlite_store.rs，实现数据库初始化和 schema 创建

**Files:**
- Create: `src/sqlite_store.rs`
- Modify: `src/lib.rs`

**Step 1: 创建 sqlite_store.rs 骨架**

```rust
use std::path::Path;
use std::str::FromStr;
use std::time::SystemTime;

use rusqlite::{params, Connection, Result as SqlResult};

use crate::models::{
    Event, Memory, MemoryQuery, MemoryType, epoch_seconds, from_epoch_seconds,
};
use crate::store::{MemoryStore, StoreError, StoreResult};

#[derive(Debug)]
pub struct SqliteMemoryStore {
    conn: Connection,
}

impl SqliteMemoryStore {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let conn = Connection::open(path).map_err(to_store_error)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .map_err(to_store_error)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    pub fn open_in_memory() -> StoreResult<Self> {
        let conn = Connection::open_in_memory().map_err(to_store_error)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> StoreResult<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_event_id TEXT,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    valid_from INTEGER,
                    valid_to INTEGER,
                    deleted_at INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted_at);",
            )
            .map_err(to_store_error)
    }
}

fn to_store_error(err: rusqlite::Error) -> StoreError {
    StoreError::Io(std::io::Error::other(err.to_string()))
}
```

**Step 2: 修改 lib.rs，条件编译导出**

在 `pub mod store;` 后面添加：
```rust
#[cfg(feature = "sqlite")]
pub mod sqlite_store;

#[cfg(feature = "sqlite")]
pub use sqlite_store::SqliteMemoryStore;
```

**Step 3: 验证编译**

Run: `cargo check --features sqlite`
Expected: 编译成功

Run: `cargo check`
Expected: 编译成功（不含 sqlite 模块）

**Step 4: Commit**

```bash
git add src/sqlite_store.rs src/lib.rs
git commit -m "feat: add SqliteMemoryStore skeleton with schema init"
```

---

### Task 3: 实现 MemoryStore trait for SqliteMemoryStore

**Objective:** 为 SqliteMemoryStore 实现完整的 MemoryStore trait

**Files:**
- Modify: `src/sqlite_store.rs`

**Step 1: 实现 add_event**

```rust
impl MemoryStore for SqliteMemoryStore {
    fn add_event(&mut self, event: Event) -> StoreResult<Event> {
        self.conn
            .execute(
                "INSERT INTO events (id, namespace, actor, text, metadata, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    event.id,
                    event.namespace,
                    event.actor,
                    event.text,
                    encode_metadata(&event.metadata),
                    epoch_seconds(event.created_at) as i64,
                ],
            )
            .map_err(to_store_error)?;
        Ok(event)
    }
```

**Step 2: 实现 add_memory**

```rust
    fn add_memory(&mut self, memory: Memory) -> StoreResult<Memory> {
        self.conn
            .execute(
                "INSERT INTO memories (id, namespace, memory_type, content, source_event_id,
                 importance, confidence, metadata, created_at, updated_at, valid_from, valid_to, deleted_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    memory.id,
                    memory.namespace,
                    memory.memory_type.to_string(),
                    memory.content,
                    memory.source_event_id,
                    memory.importance,
                    memory.confidence,
                    encode_metadata(&memory.metadata),
                    epoch_seconds(memory.created_at) as i64,
                    epoch_seconds(memory.updated_at) as i64,
                    memory.valid_from.map(|t| epoch_seconds(t) as i64),
                    memory.valid_to.map(|t| epoch_seconds(t) as i64),
                    memory.deleted_at.map(|t| epoch_seconds(t) as i64),
                ],
            )
            .map_err(to_store_error)?;
        Ok(memory)
    }
```

**Step 3: 实现 update_memory**

```rust
    fn update_memory(&mut self, mut memory: Memory) -> StoreResult<Memory> {
        memory.updated_at = SystemTime::now();
        self.conn
            .execute(
                "UPDATE memories SET content=?1, importance=?2, confidence=?3,
                 metadata=?4, updated_at=?5, valid_from=?6, valid_to=?7
                 WHERE id=?8",
                params![
                    memory.content,
                    memory.importance,
                    memory.confidence,
                    encode_metadata(&memory.metadata),
                    epoch_seconds(memory.updated_at) as i64,
                    memory.valid_from.map(|t| epoch_seconds(t) as i64),
                    memory.valid_to.map(|t| epoch_seconds(t) as i64),
                    memory.id,
                ],
            )
            .map_err(to_store_error)?;
        Ok(memory)
    }
```

**Step 4: 实现 delete_memory, get_memory, list_memories**

```rust
    fn delete_memory(&mut self, memory_id: &str) -> StoreResult<()> {
        self.conn
            .execute(
                "UPDATE memories SET deleted_at=?1 WHERE id=?2",
                params![epoch_seconds(SystemTime::now()) as i64, memory_id],
            )
            .map_err(to_store_error)?;
        Ok(())
    }

    fn get_memory(&self, memory_id: &str) -> StoreResult<Option<Memory>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM memories WHERE id=?1 AND deleted_at IS NULL")
            .map_err(to_store_error)?;
        let result = stmt.query_row(params![memory_id], row_to_memory).ok();
        Ok(result)
    }

    fn list_memories(&self, query: &MemoryQuery) -> StoreResult<Vec<Memory>> {
        let mut sql = String::from(
            "SELECT * FROM memories WHERE namespace=?1 AND deleted_at IS NULL",
        );
        if !query.memory_types.is_empty() {
            sql.push_str(" AND memory_type IN (");
            sql.push_str(
                &query
                    .memory_types
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("?{}", i + 2))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            sql.push(')');
        }
        if !query.include_expired {
            sql.push_str(" AND (valid_from IS NULL OR valid_from <= ?99) AND (valid_to IS NULL OR valid_to >= ?99)");
        }

        let mut stmt = self.conn.prepare(&sql).map_err(to_store_error)?;
        let now_seconds = if !query.include_expired {
            Some(epoch_seconds(SystemTime::now()) as i64)
        } else {
            None
        };

        let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(query.namespace.clone())];
        for mt in &query.memory_types {
            params_vec.push(Box::new(mt.to_string()));
        }
        if let Some(ns) = now_seconds {
            params_vec.push(Box::new(ns));
        }

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let rows = stmt
            .query_map(param_refs.as_slice(), row_to_memory)
            .map_err(to_store_error)?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(to_store_error)?);
        }
        Ok(results)
    }
}
```

**Step 5: 添加 row_to_memory 和 metadata 编解码辅助函数**

```rust
fn row_to_memory(row: &rusqlite::Row<'_>) -> SqlResult<Memory> {
    Ok(Memory {
        id: row.get("id")?,
        namespace: row.get("namespace")?,
        memory_type: MemoryType::from_str(&row.get::<_, String>("memory_type")?)
            .unwrap_or(MemoryType::Episodic),
        content: row.get("content")?,
        source_event_id: row.get("source_event_id")?,
        importance: row.get("importance")?,
        confidence: row.get("confidence")?,
        metadata: decode_metadata(&row.get::<_, String>("metadata")?).unwrap_or_default(),
        created_at: from_epoch_seconds(row.get::<_, i64>("created_at")? as u64),
        updated_at: from_epoch_seconds(row.get::<_, i64>("updated_at")? as u64),
        valid_from: row
            .get::<_, Option<i64>>("valid_from")?
            .map(|s| from_epoch_seconds(s as u64)),
        valid_to: row
            .get::<_, Option<i64>>("valid_to")?
            .map(|s| from_epoch_seconds(s as u64)),
        deleted_at: row
            .get::<_, Option<i64>>("deleted_at")?
            .map(|s| from_epoch_seconds(s as u64)),
    })
}

fn encode_metadata(metadata: &std::collections::BTreeMap<String, String>) -> String {
    if metadata.is_empty() {
        return String::new();
    }
    serde_json::to_string(metadata).unwrap_or_default()
}

fn decode_metadata(value: &str) -> Result<std::collections::BTreeMap<String, String>, String> {
    if value.is_empty() {
        return Ok(std::collections::BTreeMap::new());
    }
    serde_json::from_str(value).map_err(|e| e.to_string())
}
```

**Step 6: 验证编译**

Run: `cargo check --features sqlite`
Expected: 编译成功

**Step 7: Commit**

```bash
git add src/sqlite_store.rs
git commit -m "feat: implement MemoryStore trait for SqliteMemoryStore"
```

---

### Task 4: 添加 sqlite feature 的测试

**Objective:** 为 SqliteMemoryStore 编写与 FileMemoryStore 平行的测试

**Files:**
- Create: `tests/sqlite_tests.rs`

**Step 1: 创建测试文件**

```rust
#[cfg(feature = "sqlite")]
mod sqlite {
    use agent_memory::{
        Event, Memory, MemoryEngine, MemoryQuery, MemoryType, SqliteMemoryStore,
    };
    use std::time::{Duration, SystemTime};

    fn temp_db(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("agent_memory_{name}_{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn extracts_and_retrieves_semantic_memory() {
        let store = SqliteMemoryStore::open_in_memory().unwrap();
        let mut engine = MemoryEngine::new(store);
        let committed = engine
            .ingest_event(
                Event::new("Remember that I prefer Rust for edge components.").namespace("u1"),
            )
            .unwrap();
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].memory_type, MemoryType::Semantic);
        let results = engine
            .search(MemoryQuery::new("edge Rust preference").namespace("u1"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].memory.content.contains("prefer Rust"));
    }

    #[test]
    fn namespace_isolation_is_enforced() {
        let store = SqliteMemoryStore::open_in_memory().unwrap();
        let mut engine = MemoryEngine::new(store);
        engine
            .remember(Memory::new("I prefer Rust", MemoryType::Semantic).namespace("u1"))
            .unwrap();
        engine
            .remember(Memory::new("I prefer C++", MemoryType::Semantic).namespace("u2"))
            .unwrap();
        let results = engine
            .search(MemoryQuery::new("preference").namespace("u2"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].memory.content.contains("C++"));
    }

    #[test]
    fn deletion_hides_memory() {
        let store = SqliteMemoryStore::open_in_memory().unwrap();
        let mut engine = MemoryEngine::new(store);
        let memory = engine
            .remember(Memory::new("Do not expose secrets", MemoryType::Semantic))
            .unwrap();
        engine.delete_memory(&memory.id).unwrap();
        let results = engine.search(MemoryQuery::new("secrets")).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn expired_memory_is_filtered_by_default() {
        let store = SqliteMemoryStore::open_in_memory().unwrap();
        let mut engine = MemoryEngine::new(store);
        let mut memory = Memory::new("Temporary deployment window", MemoryType::Working);
        memory.valid_to = Some(SystemTime::now() - Duration::from_secs(60));
        engine.remember(memory).unwrap();
        let results = engine
            .search(MemoryQuery::new("deployment").limit(4))
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn sqlite_store_persists_to_file() {
        let path = temp_db("persist");
        {
            let store = SqliteMemoryStore::open(&path).unwrap();
            let mut engine = MemoryEngine::new(store);
            engine
                .remember(
                    Memory::new("Always run tests", MemoryType::Procedural)
                        .namespace("project"),
                )
                .unwrap();
        }
        let store = SqliteMemoryStore::open(&path).unwrap();
        let mut engine = MemoryEngine::new(store);
        let results = engine
            .search(MemoryQuery::new("tests").namespace("project"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.content, "Always run tests");
    }
}
```

**Step 2: 运行测试**

Run: `cargo test --features sqlite`
Expected: 所有测试通过（包括原有测试 + 新增 sqlite 测试）

Run: `cargo test`
Expected: 原有测试全部通过（sqlite 测试被跳过）

**Step 3: Commit**

```bash
git add tests/sqlite_tests.rs
git commit -m "test: add SqliteMemoryStore tests behind sqlite feature flag"
```

---

### Task 5: 更新 README 和 lib.rs 导出

**Objective:** 更新文档和公共 API 导出

**Files:**
- Modify: `README.md`

**Step 1: 在 README 的 Quick Start 后添加 SQLite 示例**

```markdown
### SQLite Backend

Enable the `sqlite` feature for a query-efficient persistent backend:

```toml
[dependencies]
agent-memory = { version = "0.1.0", features = ["sqlite"] }
```

```rust
use agent_memory::{SqliteMemoryStore, MemoryEngine, Event, MemoryQuery};

let store = SqliteMemoryStore::open("agent.memory.db")?;
let mut engine = MemoryEngine::new(store);
```
```

**Step 2: 更新 Evolution Path，标记第1步完成**

将 `1. Add SqliteMemoryStore behind the existing MemoryStore trait.` 改为：
`1. ~~Add SqliteMemoryStore behind the existing MemoryStore trait.~~ ✅ Done`

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add SQLite backend usage to README"
```

---

## 完成验证

所有任务完成后，运行：

```bash
# 无 feature 编译 + 原有测试
cargo test

# 含 sqlite feature 编译 + 全部测试
cargo test --features sqlite

# 确认 example 也能编译
cargo run --example basic --features sqlite
```
