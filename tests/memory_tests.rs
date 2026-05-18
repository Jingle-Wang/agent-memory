use std::fs;
use std::time::{Duration, SystemTime};

use agent_memory::{
    Event, FileMemoryStore, Memory, MemoryEngine, MemoryQuery, MemoryStore, MemoryType,
};

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("agent_memory_{name}_{}.log", std::process::id()));
    let _ = fs::remove_file(&path);
    path
}

#[test]
fn extracts_and_retrieves_semantic_memory() {
    let path = temp_path("semantic");
    let store = FileMemoryStore::open(&path).unwrap();
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
    let path = temp_path("namespace");
    let store = FileMemoryStore::open(&path).unwrap();
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
fn file_store_replays_append_log() {
    let path = temp_path("replay");
    {
        let mut store = FileMemoryStore::open(&path).unwrap();
        store
            .add_memory(
                Memory::new("Always run tests", MemoryType::Procedural).namespace("project"),
            )
            .unwrap();
    }

    let store = FileMemoryStore::open(&path).unwrap();
    let memories = store
        .list_memories(&MemoryQuery::new("tests").namespace("project"))
        .unwrap();

    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].content, "Always run tests");
}

#[test]
fn deletion_hides_memory() {
    let path = temp_path("delete");
    let store = FileMemoryStore::open(&path).unwrap();
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
    let path = temp_path("expiry");
    let store = FileMemoryStore::open(&path).unwrap();
    let mut engine = MemoryEngine::new(store);

    let mut memory = Memory::new("Temporary deployment window", MemoryType::Working);
    memory.valid_to = Some(SystemTime::now() - Duration::from_secs(60));
    engine.remember(memory).unwrap();

    let results = engine
        .search(MemoryQuery::new("deployment").limit(4))
        .unwrap();
    assert!(results.is_empty());
}
