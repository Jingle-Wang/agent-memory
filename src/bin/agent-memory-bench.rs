use std::env;
use std::fs;
use std::path::PathBuf;

use agent_memory::benchmark::{
    Answerer, BasicExtractiveAnswerer, BenchmarkKind, BenchmarkMode, BenchmarkRunConfig,
    BenchmarkRunner, DateExtractiveAnswerer, EvidenceComposerAnswerer, ExactContainsJudge,
    HeuristicReranker, HybridLlmComposerAnswerer, Judge, LlmComposerAnswerer, LlmEvidenceComposer,
    LlmReranker, NormalizedContainsJudge, SpanExtractiveAnswerer, TypedExtractiveAnswerer,
    dataset_hash, load_dataset,
};
use agent_memory::extractor::{LlmMemoryExtractor, MemoryExtractor};
use agent_memory::llm::{ConfiguredLlmProvider, LlmProvider, LlmProviderConfig};
use agent_memory::{FileMemoryStore, MemoryEngine, VolatileMemoryStore};

/// Create a memory engine with the appropriate embedding provider.
/// Set AGENT_MEMORY_EMBEDDING_MODEL to opt into real embeddings (e.g. "all-minilm").
/// When unset or empty the deterministic hash fallback is used — fast, offline-safe.
#[allow(unused_mut)]
fn create_engine<S: agent_memory::MemoryStore>(store: S) -> MemoryEngine<S> {
    #[cfg(feature = "embed-ollama")]
    {
        let model = std::env::var("AGENT_MEMORY_EMBEDDING_MODEL").unwrap_or_default();
        if model.is_empty() {
            MemoryEngine::new(store)
        } else {
            MemoryEngine::new_with_embedding(
                store,
                Box::new(agent_memory::embedding::OllamaEmbeddingProvider::from_env()),
            )
        }
    }
    #[cfg(not(feature = "embed-ollama"))]
    {
        MemoryEngine::new(store)
    }
}

#[cfg(feature = "sqlite")]
use agent_memory::SqliteMemoryStore;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    let dataset_bytes = fs::read(&args.dataset)?;
    let dataset = load_dataset(&args.dataset, args.benchmark)?;
    let config = BenchmarkRunConfig {
        benchmark: args.benchmark.name().to_string(),
        mode: args.mode,
        top_k: args.top_k,
        output_dir: args.output,
        dataset_hash: dataset_hash(&dataset_bytes),
        store: args.store.clone(),
        answerer: args.answerer.clone(),
        extractor: args.extractor.clone(),
        judge: args.judge.clone(),
        evidence_pack: args.evidence_pack.clone(),
        llm_provider: None,
        max_questions: args.limit,
        question_offset: args.offset,
    };
    fs::create_dir_all(&config.output_dir)?;

    match args.store.as_str() {
        "file" => {
            let store_path = config.output_dir.join("memory.log");
            let _ = fs::remove_file(&store_path);
            let store = FileMemoryStore::open(store_path)?;
            let mut engine = create_engine(store);
            run_with_answerer(
                &args.answerer,
                &args.extractor,
                &args.judge,
                &mut engine,
                &dataset,
                &config,
            )?;
        }
        "memory" | "volatile" => {
            let store = VolatileMemoryStore::new();
            let mut engine = create_engine(store);
            run_with_answerer(
                &args.answerer,
                &args.extractor,
                &args.judge,
                &mut engine,
                &dataset,
                &config,
            )?;
        }
        "sqlite" => run_sqlite(
            dataset,
            config,
            &args.answerer,
            &args.extractor,
            &args.judge,
        )?,
        other => return Err(format!("unsupported store: {other}").into()),
    }

    Ok(())
}

#[cfg(feature = "sqlite")]
fn run_sqlite(
    dataset: agent_memory::benchmark::BenchmarkDataset,
    config: BenchmarkRunConfig,
    answerer: &str,
    extractor: &str,
    judge: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let store_path = config.output_dir.join("memory.db");
    let _ = fs::remove_file(&store_path);
    let store = SqliteMemoryStore::open(store_path)?;
    let mut engine = create_engine(store);
    run_with_answerer(answerer, extractor, judge, &mut engine, &dataset, &config)?;
    Ok(())
}

#[cfg(not(feature = "sqlite"))]
fn run_sqlite(
    _dataset: agent_memory::benchmark::BenchmarkDataset,
    _config: BenchmarkRunConfig,
    _answerer: &str,
    _extractor: &str,
    _judge: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("sqlite store requested, but binary was built without --features sqlite".into())
}

fn run_with_answerer<S: agent_memory::MemoryStore>(
    answerer: &str,
    extractor: &str,
    judge: &str,
    engine: &mut MemoryEngine<S>,
    dataset: &agent_memory::benchmark::BenchmarkDataset,
    config: &BenchmarkRunConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    match judge {
        "exact" => run_with_answerer_and_judge(
            answerer,
            extractor,
            ExactContainsJudge,
            engine,
            dataset,
            config,
        ),
        "normalized" | "norm" => run_with_answerer_and_judge(
            answerer,
            extractor,
            NormalizedContainsJudge,
            engine,
            dataset,
            config,
        ),
        other => return Err(format!("unsupported judge: {other}").into()),
    }
}

fn run_with_answerer_and_judge<S, J>(
    answerer: &str,
    extractor_name: &str,
    judge: J,
    engine: &mut MemoryEngine<S>,
    dataset: &agent_memory::benchmark::BenchmarkDataset,
    config: &BenchmarkRunConfig,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: agent_memory::MemoryStore,
    J: Judge + Clone,
{
    let extractor = match extractor_name {
        "rule" => None,
        "llm" => {
            let provider = ConfiguredLlmProvider::from_env()?;
            let llm_extractor =
                LlmMemoryExtractor::new(provider.clone(), provider.metadata().model);
            Some(llm_extractor)
        }
        other => return Err(format!("unsupported extractor: {other}").into()),
    };
    let extractor_ref: Option<&dyn MemoryExtractor> =
        extractor.as_ref().map(|e| e as &dyn MemoryExtractor);
    let reranker_provider = if config.mode == BenchmarkMode::Answer {
        configured_reranker_provider()?
    } else {
        None
    };
    let mut run_config = config.clone();
    if let Some(provider) = &reranker_provider {
        run_config.llm_provider.get_or_insert_with(|| {
            let mut metadata = provider.metadata();
            metadata.prompt_version = Some("benchmark-rerank-v1".to_string());
            metadata
        });
    }

    match answerer {
        "basic" | "extractive" => {
            let runner = with_configured_reranker(
                BenchmarkRunner::new(BasicExtractiveAnswerer, judge.clone()),
                reranker_provider.clone(),
                true,
            );
            let report = runner.run(engine, dataset, &run_config, extractor_ref)?;
            print_summary(&report.summary);
        }
        "date" => {
            let runner = with_configured_reranker(
                BenchmarkRunner::new(DateExtractiveAnswerer, judge.clone()),
                reranker_provider.clone(),
                true,
            );
            let report = runner.run(engine, dataset, &run_config, extractor_ref)?;
            print_summary(&report.summary);
        }
        "typed" => {
            let runner = with_configured_reranker(
                BenchmarkRunner::new(TypedExtractiveAnswerer, judge.clone()),
                reranker_provider.clone(),
                true,
            );
            let report = runner.run(engine, dataset, &run_config, extractor_ref)?;
            print_summary(&report.summary);
        }
        "span" => {
            let runner = with_configured_reranker(
                BenchmarkRunner::new(SpanExtractiveAnswerer, judge.clone()),
                reranker_provider.clone(),
                true,
            );
            let report = runner.run(engine, dataset, &run_config, extractor_ref)?;
            print_summary(&report.summary);
        }
        "composer" => {
            let runner = with_configured_reranker(
                BenchmarkRunner::new(EvidenceComposerAnswerer, judge.clone()),
                reranker_provider.clone(),
                true,
            );
            let report = runner.run(engine, dataset, &run_config, extractor_ref)?;
            print_summary(&report.summary);
        }
        "llm" | "llm-composer" => {
            let provider = ConfiguredLlmProvider::from_env()?;
            let composer = LlmEvidenceComposer::new(provider);
            let answerer = LlmComposerAnswerer::new(composer);
            let mut config = run_config.clone();
            config.llm_provider = Some(answerer.metadata());
            let runner = with_configured_reranker(
                BenchmarkRunner::new(answerer, judge.clone()),
                reranker_provider.clone(),
                false,
            );
            let report = runner.run(engine, dataset, &config, extractor_ref)?;
            print_summary(&report.summary);
        }
        "llm-hybrid" | "hybrid-llm" => {
            let provider = ConfiguredLlmProvider::from_env()?;
            let composer = LlmEvidenceComposer::new(provider);
            let answerer = HybridLlmComposerAnswerer::new(composer);
            let mut config = run_config.clone();
            config.llm_provider = Some(answerer.metadata());
            let runner = with_configured_reranker(
                BenchmarkRunner::new(answerer, judge.clone()),
                reranker_provider.clone(),
                false,
            );
            let report = runner.run(engine, dataset, &config, extractor_ref)?;
            print_summary(&report.summary);
        }
        other => return Err(format!("unsupported answerer: {other}").into()),
    }
    Ok(())
}

fn configured_reranker_provider()
-> Result<Option<ConfiguredLlmProvider>, Box<dyn std::error::Error>> {
    let model = match env::var("AGENT_MEMORY_RERANK_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        Some(model) => model,
        None => return Ok(None),
    };
    let config = LlmProviderConfig {
        provider: env::var("AGENT_MEMORY_LLM_PROVIDER")
            .unwrap_or_else(|_| "openai-compatible".to_string())
            .to_lowercase(),
        model,
        base_url: env::var("AGENT_MEMORY_LLM_BASE_URL").ok(),
        api_key: env::var("AGENT_MEMORY_LLM_API_KEY").ok(),
        fixture_path: env::var("AGENT_MEMORY_LLM_FIXTURE").ok().map(PathBuf::from),
        cache_path: env::var("AGENT_MEMORY_LLM_CACHE").ok().map(PathBuf::from),
        timeout_secs: env::var("AGENT_MEMORY_LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(60),
    };
    Ok(Some(ConfiguredLlmProvider::from_config(config)?))
}

fn with_configured_reranker<A, J>(
    runner: BenchmarkRunner<A, J>,
    provider: Option<ConfiguredLlmProvider>,
    use_heuristic_fallback: bool,
) -> BenchmarkRunner<A, J>
where
    A: Answerer,
    J: Judge,
{
    let runner = if let Some(provider) = provider.clone() {
        // Phase 4: use same LLM provider for both reranking and query expansion
        let runner = runner.with_reranker(LlmReranker::new(provider.clone()));
        runner.with_query_expansion(provider)
    } else if use_heuristic_fallback
        && env::var("AGENT_MEMORY_HEURISTIC_RERANK").as_deref() == Ok("1")
    {
        runner.with_reranker(HeuristicReranker::new())
    } else {
        runner
    };
    runner
}

fn print_summary(summary: &agent_memory::benchmark::BenchmarkSummary) {
    println!("benchmark: {}", summary.benchmark);
    println!("mode: {}", summary.mode);
    println!("questions: {}", summary.question_count);
    println!("accuracy: {:?}", summary.accuracy);
    println!("recall@1: {:.4}", summary.recall_at_1);
    println!("recall@3: {:.4}", summary.recall_at_3);
    println!("recall@5: {:.4}", summary.recall_at_5);
    println!("recall@10: {:.4}", summary.recall_at_10);
    println!("recall@20: {:.4}", summary.recall_at_20);
    println!("recall@50: {:.4}", summary.recall_at_50);
    println!("recall@100: {:.4}", summary.recall_at_100);
    println!("recall@200: {:.4}", summary.recall_at_200);
    println!("mrr: {:.4}", summary.mrr);
    println!(
        "retrieval_miss@10: {:.4}",
        summary.retrieval_miss_at_10_rate
    );
    println!(
        "hit@10_answer_wrong: {:.4}",
        summary.hit_at_10_answer_wrong_rate
    );
}

#[derive(Clone, Debug)]
struct Args {
    benchmark: BenchmarkKind,
    dataset: PathBuf,
    output: PathBuf,
    mode: BenchmarkMode,
    store: String,
    answerer: String,
    extractor: String,
    judge: String,
    evidence_pack: String,
    top_k: usize,
    limit: Option<usize>,
    offset: usize,
}

impl Args {
    fn parse(values: Vec<String>) -> Result<Self, String> {
        let mut benchmark = BenchmarkKind::Generic;
        let mut dataset = None;
        let mut output = PathBuf::from("runs/latest");
        let mut mode = BenchmarkMode::Retrieval;
        let mut store = "file".to_string();
        let mut answerer = "basic".to_string();
        let mut extractor = "llm".to_string();
        let mut judge = "exact".to_string();
        let mut evidence_pack = "two-stage".to_string();
        let mut top_k = 10_usize;
        let mut limit = None;
        let mut offset = 0_usize;

        let mut index = 0;
        while index < values.len() {
            let key = &values[index];
            let value = values
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {key}"))?;
            match key.as_str() {
                "--benchmark" => benchmark = BenchmarkKind::parse(value)?,
                "--dataset" => dataset = Some(PathBuf::from(value)),
                "--output" => output = PathBuf::from(value),
                "--mode" => mode = BenchmarkMode::parse(value)?,
                "--store" => store = value.to_lowercase(),
                "--answerer" => answerer = value.to_lowercase(),
                "--extractor" => extractor = value.to_lowercase(),
                "--judge" => judge = value.to_lowercase(),
                "--evidence-pack" => evidence_pack = value.to_lowercase(),
                "--top-k" => {
                    top_k = value
                        .parse()
                        .map_err(|_| format!("invalid --top-k value: {value}"))?
                }
                "--limit" => {
                    limit = Some(
                        value
                            .parse()
                            .map_err(|_| format!("invalid --limit value: {value}"))?,
                    )
                }
                "--offset" => {
                    offset = value
                        .parse()
                        .map_err(|_| format!("invalid --offset value: {value}"))?
                }
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown argument: {other}\n{}", usage())),
            }
            index += 2;
        }

        Ok(Self {
            benchmark,
            dataset: dataset.ok_or_else(usage)?,
            output,
            mode,
            store,
            answerer,
            extractor,
            judge,
            evidence_pack,
            top_k,
            limit,
            offset,
        })
    }
}

fn usage() -> String {
    "usage: agent-memory-bench --benchmark <generic|locomo|longmemeval> --dataset <path> [--output <dir>] [--mode <retrieval|answer>] [--store <memory|file|sqlite>] [--answerer <basic|date|typed|span|composer|llm|llm-hybrid>] [--extractor <rule|llm>] [--judge <exact|normalized>] [--evidence-pack <side-channel|primary|source|source-first|two-stage>] [--top-k <n>] [--limit <n>] [--offset <n>]".to_string()
}
