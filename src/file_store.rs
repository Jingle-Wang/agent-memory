use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::SystemTime;

use crate::models::{Event, Memory, MemoryQuery, MemoryType, epoch_seconds, from_epoch_seconds};
use crate::store::{MemoryStore, StoreError, StoreResult};

#[derive(Debug)]
pub struct FileMemoryStore {
    path: PathBuf,
    events: BTreeMap<String, Event>,
    memories: BTreeMap<String, Memory>,
}

impl FileMemoryStore {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            File::create(&path)?;
        }
        let mut store = Self {
            path,
            events: BTreeMap::new(),
            memories: BTreeMap::new(),
        };
        store.load()?;
        Ok(store)
    }

    pub fn list_events(&self, namespace: &str) -> Vec<Event> {
        self.events
            .values()
            .filter(|event| event.namespace == namespace)
            .cloned()
            .collect()
    }

    fn load(&mut self) -> StoreResult<()> {
        let file = File::open(&self.path)?;
        for (line_number, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            self.apply_line(&line).map_err(|message| {
                StoreError::Corrupt(format!("line {}: {message}", line_number + 1))
            })?;
        }
        Ok(())
    }

    fn apply_line(&mut self, line: &str) -> Result<(), String> {
        let parts: Vec<String> = line.split('\t').map(unescape).collect::<Result<_, _>>()?;
        match parts.first().map(String::as_str) {
            Some("event") => {
                if parts.len() != 7 {
                    return Err("event record must have 7 fields".to_string());
                }
                let event = Event {
                    id: parts[1].clone(),
                    namespace: parts[2].clone(),
                    actor: parts[3].clone(),
                    text: parts[4].clone(),
                    metadata: decode_metadata(&parts[5])?,
                    created_at: decode_time(&parts[6])?,
                };
                self.events.insert(event.id.clone(), event);
            }
            Some("memory") => {
                if parts.len() != 15 {
                    return Err("memory record must have 15 fields".to_string());
                }
                let memory = decode_memory(&parts)?;
                self.memories.insert(memory.id.clone(), memory);
            }
            Some("delete") => {
                if parts.len() != 3 {
                    return Err("delete record must have 3 fields".to_string());
                }
                if let Some(memory) = self.memories.get_mut(&parts[1]) {
                    memory.deleted_at = Some(decode_time(&parts[2])?);
                }
            }
            Some(other) => return Err(format!("unknown record type: {other}")),
            None => return Err("empty record".to_string()),
        }
        Ok(())
    }

    fn append_line(&self, line: String) -> StoreResult<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        file.sync_data()?;
        Ok(())
    }
}

impl MemoryStore for FileMemoryStore {
    fn add_event(&mut self, event: Event) -> StoreResult<Event> {
        self.append_line(encode_event(&event))?;
        self.events.insert(event.id.clone(), event.clone());
        Ok(event)
    }

    fn add_memory(&mut self, memory: Memory) -> StoreResult<Memory> {
        self.append_line(encode_memory(&memory))?;
        self.memories.insert(memory.id.clone(), memory.clone());
        Ok(memory)
    }

    fn update_memory(&mut self, mut memory: Memory) -> StoreResult<Memory> {
        memory.updated_at = SystemTime::now();
        self.append_line(encode_memory(&memory))?;
        self.memories.insert(memory.id.clone(), memory.clone());
        Ok(memory)
    }

    fn delete_memory(&mut self, memory_id: &str) -> StoreResult<()> {
        let deleted_at = SystemTime::now();
        self.append_line(join_fields(vec![
            "delete".to_string(),
            memory_id.to_string(),
            encode_time(Some(deleted_at)),
        ]))?;
        if let Some(memory) = self.memories.get_mut(memory_id) {
            memory.deleted_at = Some(deleted_at);
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
        let now = SystemTime::now();
        let mut result = Vec::new();
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
            if !query.include_expired && !is_valid_at(memory, now) {
                continue;
            }
            if !query.include_side_channel && memory.is_side_channel() {
                continue;
            }
            result.push(memory.clone());
        }
        Ok(result)
    }
}

fn is_valid_at(memory: &Memory, now: SystemTime) -> bool {
    if memory.valid_from.is_some_and(|from| from > now) {
        return false;
    }
    if memory.valid_to.is_some_and(|to| to < now) {
        return false;
    }
    true
}

fn encode_event(event: &Event) -> String {
    join_fields(vec![
        "event".to_string(),
        event.id.clone(),
        event.namespace.clone(),
        event.actor.clone(),
        event.text.clone(),
        encode_metadata(&event.metadata),
        encode_time(Some(event.created_at)),
    ])
}

fn encode_memory(memory: &Memory) -> String {
    join_fields(vec![
        "memory".to_string(),
        memory.id.clone(),
        memory.namespace.clone(),
        memory.memory_type.to_string(),
        memory.content.clone(),
        memory.source_event_id.clone().unwrap_or_default(),
        format!("{:.6}", memory.importance),
        format!("{:.6}", memory.confidence),
        encode_metadata(&memory.metadata),
        encode_time(Some(memory.created_at)),
        encode_time(Some(memory.updated_at)),
        encode_time(memory.valid_from),
        encode_time(memory.valid_to),
        encode_time(memory.deleted_at),
        String::new(),
    ])
}

fn decode_memory(parts: &[String]) -> Result<Memory, String> {
    Ok(Memory {
        id: parts[1].clone(),
        namespace: parts[2].clone(),
        memory_type: MemoryType::from_str(&parts[3])?,
        content: parts[4].clone(),
        source_event_id: if parts[5].is_empty() {
            None
        } else {
            Some(parts[5].clone())
        },
        importance: parts[6]
            .parse()
            .map_err(|_| "invalid importance".to_string())?,
        confidence: parts[7]
            .parse()
            .map_err(|_| "invalid confidence".to_string())?,
        metadata: decode_metadata(&parts[8])?,
        created_at: decode_time(&parts[9])?,
        updated_at: decode_time(&parts[10])?,
        valid_from: decode_optional_time(&parts[11])?,
        valid_to: decode_optional_time(&parts[12])?,
        deleted_at: decode_optional_time(&parts[13])?,
    })
}

fn join_fields(fields: impl IntoIterator<Item = String>) -> String {
    fields
        .into_iter()
        .map(|field| escape(&field))
        .collect::<Vec<_>>()
        .join("\t")
}

fn escape(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => output.push_str("\\\\"),
            '\t' => output.push_str("\\t"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '=' => output.push_str("\\e"),
            ';' => output.push_str("\\s"),
            _ => output.push(ch),
        }
    }
    output
}

fn unescape(value: &str) -> Result<String, String> {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some('\\') => output.push('\\'),
            Some('t') => output.push('\t'),
            Some('n') => output.push('\n'),
            Some('r') => output.push('\r'),
            Some('e') => output.push('='),
            Some('s') => output.push(';'),
            Some(other) => return Err(format!("invalid escape: \\{other}")),
            None => return Err("trailing escape".to_string()),
        }
    }
    Ok(output)
}

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
        let Some((key, value)) = pair.split_once('=') else {
            return Err("invalid metadata pair".to_string());
        };
        metadata.insert(hex_decode(key)?, hex_decode(value)?);
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

fn encode_time(time: Option<SystemTime>) -> String {
    time.map(epoch_seconds)
        .map(|seconds| seconds.to_string())
        .unwrap_or_default()
}

fn decode_time(value: &str) -> Result<SystemTime, String> {
    value
        .parse()
        .map(from_epoch_seconds)
        .map_err(|_| "invalid timestamp".to_string())
}

fn decode_optional_time(value: &str) -> Result<Option<SystemTime>, String> {
    if value.is_empty() {
        Ok(None)
    } else {
        decode_time(value).map(Some)
    }
}
