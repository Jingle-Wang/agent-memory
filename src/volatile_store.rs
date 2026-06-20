use std::collections::BTreeMap;

use crate::models::{Event, Memory, MemoryQuery};
use crate::store::{MemoryStore, StoreResult};

#[derive(Clone, Debug, Default)]
pub struct VolatileMemoryStore {
    events: BTreeMap<String, Event>,
    memories: BTreeMap<String, Memory>,
}

impl VolatileMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MemoryStore for VolatileMemoryStore {
    fn add_event(&mut self, event: Event) -> StoreResult<Event> {
        self.events.insert(event.id.clone(), event.clone());
        Ok(event)
    }

    fn add_memory(&mut self, memory: Memory) -> StoreResult<Memory> {
        self.memories.insert(memory.id.clone(), memory.clone());
        Ok(memory)
    }

    fn update_memory(&mut self, memory: Memory) -> StoreResult<Memory> {
        self.memories.insert(memory.id.clone(), memory.clone());
        Ok(memory)
    }

    fn delete_memory(&mut self, memory_id: &str) -> StoreResult<()> {
        if let Some(memory) = self.memories.get_mut(memory_id) {
            memory.deleted_at = Some(std::time::SystemTime::now());
        }
        Ok(())
    }

    fn get_memory(&self, memory_id: &str) -> StoreResult<Option<Memory>> {
        Ok(self
            .memories
            .get(memory_id)
            .filter(|memory| memory.deleted_at.is_none())
            .cloned())
    }

    fn list_memories(&self, query: &MemoryQuery) -> StoreResult<Vec<Memory>> {
        let now = std::time::SystemTime::now();
        let mut memories = Vec::new();
        for memory in self.memories.values() {
            if memory.namespace != query.namespace || memory.deleted_at.is_some() {
                continue;
            }
            if !query.memory_types.is_empty()
                && !query
                    .memory_types
                    .iter()
                    .any(|kind| kind == &memory.memory_type)
            {
                continue;
            }
            if !query.include_expired {
                if memory.valid_from.is_some_and(|from| from > now)
                    || memory.valid_to.is_some_and(|to| to < now)
                {
                    continue;
                }
            }
            if !query.include_side_channel && memory.is_side_channel() {
                continue;
            }
            memories.push(memory.clone());
        }
        memories.truncate(query.limit.max(5000));
        Ok(memories)
    }
}
