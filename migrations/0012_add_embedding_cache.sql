-- Migration: Add embedding cache table
-- Date: 2026-05-21
-- Description: Persist semantic embedding vectors across process restarts

-- Cache keys bind provider/model and content. The content column lets callers
-- reject the unlikely event of a hash collision.
CREATE TABLE IF NOT EXISTS memory_embeddings (
    cache_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_embeddings_provider
    ON memory_embeddings(provider);

CREATE TABLE IF NOT EXISTS memory_entities (
    namespace TEXT NOT NULL,
    entity TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    PRIMARY KEY (namespace, entity, memory_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_entities_memory
    ON memory_entities(memory_id);
