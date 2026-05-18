#[cfg(feature = "sqlite")]
mod sqlite_tests {
    use agent_memory::{
        Event, Memory, MemoryQuery, MemoryStore, MemoryType, SqliteMemoryStore,
    };

    fn make_store() -> SqliteMemoryStore {
        SqliteMemoryStore::open_in_memory().expect("open in-memory store")
    }

    #[test]
    fn test_add_and_get_event() {
        let mut store = make_store();
        let event = Event::new("hello world").namespace("test").actor("bot");
        let added = store.add_event(event.clone()).expect("add event");

        assert_eq!(added.id, event.id);
        assert_eq!(added.text, "hello world");

        let events = store.list_events("test");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].text, "hello world");
    }

    #[test]
    fn test_add_and_get_memory() {
        let mut store = make_store();
        let memory = Memory::new("remember this", MemoryType::Episodic)
            .namespace("test")
            .importance(0.8)
            .confidence(0.9);
        let added = store.add_memory(memory.clone()).expect("add memory");

        assert_eq!(added.id, memory.id);
        assert_eq!(added.content, "remember this");

        let fetched = store
            .get_memory(&memory.id)
            .expect("get memory")
            .expect("memory exists");
        assert_eq!(fetched.content, "remember this");
        assert_eq!(fetched.memory_type, MemoryType::Episodic);
    }

    #[test]
    fn test_update_memory() {
        let mut store = make_store();
        let memory = Memory::new("original", MemoryType::Semantic).namespace("test");
        store.add_memory(memory.clone()).expect("add memory");

        let mut updated = memory.clone();
        updated.content = "updated".to_string();
        updated.importance = 0.95;
        let result = store.update_memory(updated).expect("update memory");

        assert_eq!(result.content, "updated");
        assert!((result.importance - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_delete_memory() {
        let mut store = make_store();
        let memory = Memory::new("to delete", MemoryType::Working).namespace("test");
        store.add_memory(memory.clone()).expect("add memory");

        store.delete_memory(&memory.id).expect("delete memory");

        let fetched = store.get_memory(&memory.id).expect("get memory");
        assert!(fetched.is_none(), "deleted memory should not be returned");
    }

    #[test]
    fn test_list_memories_by_namespace() {
        let mut store = make_store();
        store
            .add_memory(
                Memory::new("ns1 memory", MemoryType::Semantic).namespace("ns1"),
            )
            .expect("add");
        store
            .add_memory(
                Memory::new("ns2 memory", MemoryType::Semantic).namespace("ns2"),
            )
            .expect("add");

        let query = MemoryQuery::new("anything").namespace("ns1");
        let results = store.list_memories(&query).expect("list");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "ns1 memory");
    }

    #[test]
    fn test_list_memories_by_type() {
        let mut store = make_store();
        store
            .add_memory(
                Memory::new("episodic", MemoryType::Episodic).namespace("test"),
            )
            .expect("add");
        store
            .add_memory(
                Memory::new("semantic", MemoryType::Semantic).namespace("test"),
            )
            .expect("add");
        store
            .add_memory(
                Memory::new("working", MemoryType::Working).namespace("test"),
            )
            .expect("add");

        let query = MemoryQuery::new("anything")
            .namespace("test")
            .memory_types(vec![MemoryType::Episodic, MemoryType::Working]);
        let results = store.list_memories(&query).expect("list");
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|m| m.memory_type != MemoryType::Semantic));
    }

    #[test]
    fn test_list_memories_limit() {
        let mut store = make_store();
        for i in 0..10 {
            store
                .add_memory(
                    Memory::new(format!("memory {i}"), MemoryType::Semantic)
                        .namespace("test"),
                )
                .expect("add");
        }

        let query = MemoryQuery::new("anything").namespace("test").limit(3);
        let results = store.list_memories(&query).expect("list");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_memory_with_metadata() {
        let mut store = make_store();
        let mut memory = Memory::new("with meta", MemoryType::Procedural).namespace("test");
        memory.metadata.insert("key1".to_string(), "value1".to_string());
        memory.metadata.insert("key2".to_string(), "hello world".to_string());

        store.add_memory(memory.clone()).expect("add memory");

        let fetched = store
            .get_memory(&memory.id)
            .expect("get")
            .expect("exists");
        assert_eq!(fetched.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(fetched.metadata.get("key2"), Some(&"hello world".to_string()));
    }

    #[test]
    fn test_multiple_events_same_namespace() {
        let mut store = make_store();
        store
            .add_event(Event::new("event 1").namespace("ns"))
            .expect("add");
        store
            .add_event(Event::new("event 2").namespace("ns"))
            .expect("add");
        store
            .add_event(Event::new("event 3").namespace("other"))
            .expect("add");

        let events = store.list_events("ns");
        assert_eq!(events.len(), 2);
    }
}
