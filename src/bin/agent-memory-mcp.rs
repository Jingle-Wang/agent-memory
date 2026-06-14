#[cfg(feature = "sqlite")]
fn main() {
    if let Err(error) = mcp::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "sqlite"))]
fn main() {
    eprintln!("error: agent-memory-mcp requires --features sqlite");
    std::process::exit(1);
}

#[cfg(feature = "sqlite")]
mod mcp {
    use std::env;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::PathBuf;
    use std::str::FromStr;

    use agent_memory::llm::{ConfiguredLlmProvider, LlmProvider};
    use agent_memory::{
        Event, LlmMemoryExtractor, MemoryEngine, MemoryPacket, MemoryQuery, MemoryStore,
        MemoryType, SqliteMemoryStore,
    };
    use serde_json::{Value, json};

    type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

    pub fn run() -> AppResult<()> {
        let args = Args::parse(env::args().skip(1).collect())?;
        let store = SqliteMemoryStore::open(&args.db_path)?;
        let llm_provider = if args.extractor == "llm" {
            Some(ConfiguredLlmProvider::from_env()?)
        } else {
            None
        };
        let mut service = MemoryMcpService::new(create_engine(store), args.namespace, llm_provider);

        match args.transport {
            Transport::Stdio => run_stdio(&mut service),
            Transport::Http => run_http(&mut service, &args.addr),
        }
    }

    #[cfg(feature = "embed-ollama")]
    fn create_engine(store: SqliteMemoryStore) -> MemoryEngine<SqliteMemoryStore> {
        if env::var("AGENT_MEMORY_EMBEDDING_PROVIDER").as_deref() == Ok("ollama") {
            return MemoryEngine::new_with_embedding(
                store,
                Box::new(agent_memory::embedding::OllamaEmbeddingProvider::from_env()),
            );
        }
        MemoryEngine::new(store)
    }

    #[cfg(not(feature = "embed-ollama"))]
    fn create_engine(store: SqliteMemoryStore) -> MemoryEngine<SqliteMemoryStore> {
        MemoryEngine::new(store)
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Transport {
        Stdio,
        Http,
    }

    #[derive(Clone, Debug)]
    struct Args {
        transport: Transport,
        addr: String,
        db_path: PathBuf,
        namespace: String,
        extractor: String,
    }

    impl Args {
        fn parse(values: Vec<String>) -> AppResult<Self> {
            let mut transport = Transport::Stdio;
            let mut addr = "127.0.0.1:8787".to_string();
            let mut db_path = env::var("AGENT_MEMORY_DB")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("agent-memory.db"));
            let mut namespace =
                env::var("AGENT_MEMORY_NAMESPACE").unwrap_or_else(|_| "default".to_string());
            let mut extractor =
                env::var("AGENT_MEMORY_EXTRACTOR").unwrap_or_else(|_| "rule".to_string());

            let mut index = 0;
            while index < values.len() {
                let key = &values[index];
                let value = values
                    .get(index + 1)
                    .ok_or_else(|| format!("missing value for {key}"))?;
                match key.as_str() {
                    "--transport" => {
                        transport = match value.as_str() {
                            "stdio" => Transport::Stdio,
                            "http" | "streamable-http" => Transport::Http,
                            other => return Err(format!("unsupported transport: {other}").into()),
                        }
                    }
                    "--addr" => addr = value.clone(),
                    "--db" | "--db-path" => db_path = PathBuf::from(value),
                    "--namespace" => namespace = value.clone(),
                    "--extractor" => {
                        extractor = match value.as_str() {
                            "rule" | "llm" => value.clone(),
                            other => return Err(format!("unsupported extractor: {other}").into()),
                        }
                    }
                    "--help" | "-h" => return Err(usage().into()),
                    other => return Err(format!("unknown argument: {other}\n{}", usage()).into()),
                }
                index += 2;
            }

            Ok(Self {
                transport,
                addr,
                db_path,
                namespace,
                extractor,
            })
        }
    }

    fn usage() -> String {
        "usage: agent-memory-mcp [--transport <stdio|http>] [--addr <host:port>] [--db <path>] [--namespace <name>] [--extractor <rule|llm>]".to_string()
    }

    struct MemoryMcpService {
        engine: MemoryEngine<SqliteMemoryStore>,
        default_namespace: String,
        llm_provider: Option<ConfiguredLlmProvider>,
    }

    impl MemoryMcpService {
        fn new(
            engine: MemoryEngine<SqliteMemoryStore>,
            default_namespace: String,
            llm_provider: Option<ConfiguredLlmProvider>,
        ) -> Self {
            Self {
                engine,
                default_namespace,
                llm_provider,
            }
        }

        fn handle(&mut self, request: Value) -> Option<Value> {
            let id = request.get("id").cloned();
            let method = request
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or_default();

            if id.is_none() {
                return None;
            }

            let result = match method {
                "initialize" => Ok(json!({
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "agent-memory", "version": env!("CARGO_PKG_VERSION")}
                })),
                "ping" => Ok(json!({})),
                "tools/list" => Ok(json!({"tools": tool_definitions()})),
                "tools/call" => self.handle_tool_call(request.get("params")),
                "resources/list" => Ok(json!({"resources": []})),
                "prompts/list" => Ok(json!({"prompts": []})),
                _ => Err((-32601, format!("method not found: {method}"))),
            };

            Some(match result {
                Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                Err((code, message)) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": code, "message": message}
                }),
            })
        }

        fn handle_tool_call(&mut self, params: Option<&Value>) -> Result<Value, (i32, String)> {
            let params = params.ok_or_else(|| invalid_params("missing params"))?;
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_params("missing tool name"))?;
            let arguments = params
                .get("arguments")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();

            let value = match name {
                "remember" => self.tool_remember(&arguments),
                "search" => self.tool_search(&arguments),
                "build_context" => self.tool_build_context(&arguments),
                "delete_memory" => self.tool_delete_memory(&arguments),
                other => Err(invalid_params(format!("unknown tool: {other}"))),
            }?;

            Ok(tool_result(value))
        }

        fn tool_remember(
            &mut self,
            arguments: &serde_json::Map<String, Value>,
        ) -> Result<Value, (i32, String)> {
            let text = required_string(arguments, "text")?;
            let namespace = namespace(arguments, &self.default_namespace);
            let actor = optional_string(arguments, "actor").unwrap_or_else(|| "user".to_string());
            let mut event = Event::new(text).namespace(namespace).actor(actor);
            if let Some(metadata) = arguments.get("metadata").and_then(Value::as_object) {
                for (key, value) in metadata {
                    if let Some(value) = value.as_str() {
                        event.metadata.insert(key.clone(), value.to_string());
                    }
                }
            }

            let event_id = event.id.clone();
            let memories = if let Some(provider) = &self.llm_provider {
                let extractor =
                    LlmMemoryExtractor::new(provider.clone(), provider.metadata().model);
                self.engine
                    .ingest_event_with_extractor(event, &extractor)
                    .map_err(store_error)?
            } else {
                self.engine.ingest_event(event).map_err(store_error)?
            };
            Ok(json!({
                "event_id": event_id,
                "memory_ids": memories.into_iter().map(|memory| memory.id).collect::<Vec<_>>()
            }))
        }

        fn tool_search(
            &mut self,
            arguments: &serde_json::Map<String, Value>,
        ) -> Result<Value, (i32, String)> {
            let query_text = required_string(arguments, "query")?;
            let query = memory_query(arguments, &self.default_namespace, query_text)?;
            let packets = self.engine.search(query).map_err(store_error)?;
            Ok(json!({
                "memories": packets.into_iter().map(packet_json).collect::<Vec<_>>()
            }))
        }

        fn tool_build_context(
            &mut self,
            arguments: &serde_json::Map<String, Value>,
        ) -> Result<Value, (i32, String)> {
            let query_text = required_string(arguments, "query")?;
            let query = memory_query(arguments, &self.default_namespace, query_text)?;
            let context = self.engine.build_context(query).map_err(store_error)?;
            Ok(json!({"context": context}))
        }

        fn tool_delete_memory(
            &mut self,
            arguments: &serde_json::Map<String, Value>,
        ) -> Result<Value, (i32, String)> {
            let memory_id = required_string(arguments, "memory_id")?;
            let namespace = namespace(arguments, &self.default_namespace);
            let memory = self
                .engine
                .store()
                .get_memory(&memory_id)
                .map_err(store_error)?
                .ok_or_else(|| invalid_params("memory not found"))?;
            if memory.namespace != namespace {
                return Err(invalid_params("memory not found in namespace"));
            }
            self.engine.delete_memory(&memory_id).map_err(store_error)?;
            Ok(json!({"deleted": true}))
        }
    }

    fn run_stdio(service: &mut MemoryMcpService) -> AppResult<()> {
        let stdin = std::io::stdin();
        let mut reader = BufReader::new(stdin.lock());
        let mut stdout = std::io::stdout().lock();
        while let Some(message) = read_stdio_message(&mut reader)? {
            if message.is_empty() {
                continue;
            }
            let request: Value = serde_json::from_str(&message)?;
            if let Some(response) = service.handle(request) {
                write_stdio_message(&mut stdout, &serde_json::to_string(&response)?)?;
            }
        }
        Ok(())
    }

    fn read_stdio_message(reader: &mut impl BufRead) -> AppResult<Option<String>> {
        let mut first = String::new();
        if reader.read_line(&mut first)? == 0 {
            return Ok(None);
        }
        if first.trim().is_empty() {
            return Ok(Some(String::new()));
        }
        if !first.to_ascii_lowercase().starts_with("content-length:") {
            return Ok(Some(first.trim_end().to_string()));
        }

        let mut content_length = parse_content_length(&first)?;
        loop {
            let mut header = String::new();
            reader.read_line(&mut header)?;
            let trimmed = header.trim();
            if trimmed.is_empty() {
                break;
            }
            if trimmed.to_ascii_lowercase().starts_with("content-length:") {
                content_length = parse_content_length(trimmed)?;
            }
        }
        let mut buffer = vec![0_u8; content_length];
        reader.read_exact(&mut buffer)?;
        Ok(Some(String::from_utf8(buffer)?))
    }

    fn write_stdio_message(writer: &mut impl Write, message: &str) -> AppResult<()> {
        write!(
            writer,
            "Content-Length: {}\r\n\r\n{}",
            message.len(),
            message
        )?;
        writer.flush()?;
        Ok(())
    }

    fn run_http(service: &mut MemoryMcpService, addr: &str) -> AppResult<()> {
        let listener = TcpListener::bind(addr)?;
        eprintln!("agent-memory-mcp listening on http://{addr}/mcp");
        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    if let Err(error) = handle_http_stream(service, &mut stream) {
                        let _ = write_http_error(&mut stream, 500, &error.to_string());
                    }
                }
                Err(error) => eprintln!("http accept error: {error}"),
            }
        }
        Ok(())
    }

    fn handle_http_stream(service: &mut MemoryMcpService, stream: &mut TcpStream) -> AppResult<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line)?;
        let parts = request_line.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 2 || parts[0] != "POST" || parts[1] != "/mcp" {
            return write_http_error(stream, 404, "not found");
        }

        let mut content_length = 0_usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if trimmed.to_ascii_lowercase().starts_with("content-length:") {
                content_length = parse_content_length(trimmed)?;
            }
        }

        let mut body = vec![0_u8; content_length];
        reader.read_exact(&mut body)?;
        let request: Value = serde_json::from_slice(&body)?;
        if let Some(response) = service.handle(request) {
            write_http_json(stream, 200, &serde_json::to_string(&response)?)?;
        } else {
            write_http_no_content(stream)?;
        }
        Ok(())
    }

    fn write_http_json(stream: &mut TcpStream, status: u16, body: &str) -> AppResult<()> {
        write!(
            stream,
            "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            status_text(status),
            body.len(),
            body
        )?;
        stream.flush()?;
        Ok(())
    }

    fn write_http_no_content(stream: &mut TcpStream) -> AppResult<()> {
        write!(
            stream,
            "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )?;
        stream.flush()?;
        Ok(())
    }

    fn write_http_error(stream: &mut TcpStream, status: u16, message: &str) -> AppResult<()> {
        let body = json!({"error": message}).to_string();
        write_http_json(stream, status, &body)
    }

    fn tool_result(value: Value) -> Value {
        json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
            }],
            "structuredContent": value,
            "isError": false
        })
    }

    fn tool_definitions() -> Vec<Value> {
        vec![
            tool_schema(
                "remember",
                "Store an agent event and extracted memories.",
                json!({
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "namespace": {"type": "string"},
                        "actor": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object", "additionalProperties": {"type": "string"}}
                    }
                }),
            ),
            tool_schema(
                "search",
                "Search memories relevant to a query.",
                query_schema(),
            ),
            tool_schema(
                "build_context",
                "Build prompt-ready memory context for a query.",
                query_schema(),
            ),
            tool_schema(
                "delete_memory",
                "Delete a memory by id within a namespace.",
                json!({
                    "type": "object",
                    "required": ["memory_id"],
                    "properties": {
                        "namespace": {"type": "string"},
                        "memory_id": {"type": "string"}
                    }
                }),
            ),
        ]
    }

    fn tool_schema(name: &str, description: &str, input_schema: Value) -> Value {
        json!({
            "name": name,
            "description": description,
            "inputSchema": input_schema
        })
    }

    fn query_schema() -> Value {
        json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "namespace": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural", "reflection"]}
                }
            }
        })
    }

    fn memory_query(
        arguments: &serde_json::Map<String, Value>,
        default_namespace: &str,
        query_text: String,
    ) -> Result<MemoryQuery, (i32, String)> {
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(8)
            .clamp(1, 100) as usize;
        let mut query = MemoryQuery::new(query_text)
            .namespace(namespace(arguments, default_namespace))
            .limit(limit);
        if let Some(types) = arguments.get("memory_types").and_then(Value::as_array) {
            let memory_types = types
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .ok_or_else(|| invalid_params("memory_types must be strings"))
                        .and_then(|value| MemoryType::from_str(value).map_err(invalid_params))
                })
                .collect::<Result<Vec<_>, _>>()?;
            query = query.memory_types(memory_types);
        }
        Ok(query)
    }

    fn packet_json(packet: MemoryPacket) -> Value {
        json!({
            "id": packet.memory.id,
            "type": packet.memory.memory_type.to_string(),
            "content": packet.memory.content,
            "score": packet.score,
            "reasons": packet.reasons,
            "source_event_id": packet.memory.source_event_id,
            "importance": packet.memory.importance,
            "confidence": packet.memory.confidence,
            "metadata": packet.memory.metadata
        })
    }

    fn namespace(arguments: &serde_json::Map<String, Value>, default_namespace: &str) -> String {
        optional_string(arguments, "namespace").unwrap_or_else(|| default_namespace.to_string())
    }

    fn required_string(
        arguments: &serde_json::Map<String, Value>,
        key: &str,
    ) -> Result<String, (i32, String)> {
        optional_string(arguments, key)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid_params(format!("missing {key}")))
    }

    fn optional_string(arguments: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
        arguments
            .get(key)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    }

    fn invalid_params(message: impl Into<String>) -> (i32, String) {
        (-32602, message.into())
    }

    fn store_error(error: impl std::fmt::Display) -> (i32, String) {
        (-32000, error.to_string())
    }

    fn parse_content_length(line: &str) -> AppResult<usize> {
        let (_, value) = line
            .split_once(':')
            .ok_or_else(|| "invalid Content-Length header".to_string())?;
        Ok(value.trim().parse()?)
    }

    fn status_text(status: u16) -> &'static str {
        match status {
            200 => "OK",
            202 => "Accepted",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "OK",
        }
    }
}
