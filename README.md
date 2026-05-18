# agent-memory

A lightweight local-first memory layer for AI agents, implemented in Rust.

The first implementation is intentionally small and std-only:

- append-only local file store with replay recovery
- working, episodic, semantic, procedural, and reflection memory types
- rule-based memory extraction from events
- duplicate merge on write
- hybrid retrieval with hashed vectors, keyword overlap, recency, importance, and confidence
- context packet generation for agent prompts

## Quick Start

```bash
cargo test
cargo run --example basic
```

```rust
use agent_memory::{Event, FileMemoryStore, MemoryEngine, MemoryQuery};

let store = FileMemoryStore::open("agent.memory.log")?;
let mut engine = MemoryEngine::new(store);

engine.ingest_event(
    Event::new("Remember that I prefer concise answers.")
        .namespace("user:42")
        .actor("user"),
)?;

let context = engine.build_context(
    MemoryQuery::new("How should you answer?")
        .namespace("user:42")
        .limit(4),
)?;
```

## Architecture

`MemoryEngine` is the agent-facing API. It writes raw events, extracts candidate memories,
merges duplicates, searches relevant memories, and builds prompt-ready context.

`MemoryStore` is the persistence boundary. `FileMemoryStore` is the edge-friendly default;
cloud or richer local backends can implement the same trait without changing agent code.

`HybridMemoryRetriever` ranks memories using a local hashed embedding plus lexical,
importance, confidence, recency, and memory-type signals. The hash embedding is a stable
placeholder for an on-device or hosted embedding model.

## Evolution Path

1. Add `SqliteMemoryStore` behind the existing `MemoryStore` trait.
2. Replace `HashEmbedding` with a pluggable embedding provider.
3. Add background consolidation for reflection and stale-fact expiry.
4. Add local-first sync with conflict resolution.
5. Add a compact entity-relation layer for temporal facts and multi-hop retrieval.

