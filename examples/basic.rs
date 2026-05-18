use agent_memory::{Event, FileMemoryStore, MemoryEngine, MemoryQuery};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = FileMemoryStore::open("target/example.memory.log")?;
    let mut engine = MemoryEngine::new(store);

    engine.ingest_event(
        Event::new("Remember that I prefer concise answers. When you change code, run tests.")
            .namespace("user:demo")
            .actor("user"),
    )?;

    let context = engine.build_context(
        MemoryQuery::new("How should you answer and verify code changes?")
            .namespace("user:demo")
            .limit(4),
    )?;

    println!("{context}");
    Ok(())
}
