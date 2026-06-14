#![cfg(feature = "benchmark")]

use std::fs;

use agent_memory::benchmark::{
    AnswerInput, Answerer, BenchmarkKind, BenchmarkMode, BenchmarkRunConfig, BenchmarkRunner,
    ExactContainsJudge, ExtractiveAnswerer, Judge, JudgeInput, MemoryPacketForAnswerer,
    NormalizedContainsJudge, SpanExtractiveAnswerer, ensure_no_gold_leak, load_dataset,
};
use agent_memory::llm::{
    FixtureLlmProvider, LlmCompletionRequest, LlmMessage, LlmProvider, request_hash,
};
use agent_memory::{Event, FileMemoryStore, Memory, MemoryEngine, MemoryStore, MemoryType};

fn temp_output(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("agent_memory_bench_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    path
}

#[test]
fn loads_tiny_generic_dataset() {
    let dataset =
        load_dataset("tests/fixtures/tiny_benchmark.json", BenchmarkKind::Generic).unwrap();

    assert_eq!(dataset.conversations.len(), 1);
    assert_eq!(dataset.conversations[0].turns.len(), 2);
    assert_eq!(dataset.questions.len(), 1);
    assert_eq!(dataset.questions[0].gold_answers, vec!["Rust"]);
}

#[test]
fn locomo_loader_preserves_null_answers_as_none() {
    let dataset = load_dataset(
        "tests/fixtures/locomo_null_answer.json",
        BenchmarkKind::Locomo,
    )
    .unwrap();
    let question = dataset
        .questions
        .iter()
        .find(|question| question.category.as_deref() == Some("5"))
        .unwrap();

    assert_eq!(question.gold_answers, vec!["None"]);
}

#[test]
fn answerer_view_omits_gold_and_evidence_fields() {
    let dataset =
        load_dataset("tests/fixtures/tiny_benchmark.json", BenchmarkKind::Generic).unwrap();
    let question = &dataset.questions[0];
    let input = AnswerInput {
        question: question.for_answerer(),
        retrieved: vec![MemoryPacketForAnswerer {
            memory_id: "m1".to_string(),
            content: "user mentioned Rust".to_string(),
            memory_type: "semantic".to_string(),
            metadata: std::collections::BTreeMap::new(),
            score: 1.0,
            source_event_id: Some("turn-1".to_string()),
        }],
    };

    assert!(ensure_no_gold_leak(question, &input));
}

#[test]
fn benchmark_runner_scores_retrieval_fixture() {
    let dataset =
        load_dataset("tests/fixtures/tiny_benchmark.json", BenchmarkKind::Generic).unwrap();
    let output_dir = temp_output("retrieval");
    let store = FileMemoryStore::open(output_dir.join("memory.log")).unwrap();
    let mut engine = MemoryEngine::new(store);
    let runner = BenchmarkRunner::new(ExtractiveAnswerer, ExactContainsJudge);
    let config = BenchmarkRunConfig {
        benchmark: "generic".to_string(),
        mode: BenchmarkMode::Retrieval,
        top_k: 5,
        output_dir: output_dir.clone(),
        dataset_hash: "fixture".to_string(),
        store: "file".to_string(),
        answerer: "extractive".to_string(),
        extractor: "rule".to_string(),
        judge: "exact".to_string(),
        evidence_pack: "side-channel".to_string(),
        llm_provider: None,
        max_questions: None,
        question_offset: 0,
    };

    let report = runner.run(&mut engine, &dataset, &config, None).unwrap();

    assert_eq!(report.summary.question_count, 1);
    assert!(report.summary.recall_at_5 > 0.0);
    assert!(output_dir.join("manifest.json").exists());
    assert!(output_dir.join("summary.json").exists());
    assert!(output_dir.join("scores.jsonl").exists());
}

#[test]
fn benchmark_runner_indexes_verbatim_turn_and_session_evidence() {
    let dataset =
        load_dataset("tests/fixtures/tiny_benchmark.json", BenchmarkKind::Generic).unwrap();
    let output_dir = temp_output("verbatim");
    let store = FileMemoryStore::open(output_dir.join("memory.log")).unwrap();
    let mut engine = MemoryEngine::new(store);
    let runner = BenchmarkRunner::new(ExtractiveAnswerer, ExactContainsJudge);
    let config = BenchmarkRunConfig {
        benchmark: "generic".to_string(),
        mode: BenchmarkMode::Retrieval,
        top_k: 10,
        output_dir: output_dir.clone(),
        dataset_hash: "fixture".to_string(),
        store: "file".to_string(),
        answerer: "extractive".to_string(),
        extractor: "rule".to_string(),
        judge: "exact".to_string(),
        evidence_pack: "primary".to_string(),
        llm_provider: None,
        max_questions: None,
        question_offset: 0,
    };

    runner.run(&mut engine, &dataset, &config, None).unwrap();
    let memories = engine
        .store()
        .list_memories(
            &agent_memory::MemoryQuery::new("")
                .namespace("conv-1")
                .limit(100),
        )
        .unwrap();

    assert!(memories.iter().any(|memory| {
        memory.metadata.get("memory_kind").map(String::as_str) == Some("verbatim_turn")
            && memory.content.contains("preferred systems language")
    }));
    assert!(memories.iter().any(|memory| {
        memory.metadata.get("memory_kind").map(String::as_str) == Some("verbatim_session")
            && memory
                .content
                .contains("user: My preferred systems language")
            && memory.content.contains("assistant: Noted")
    }));
}

#[test]
fn benchmark_runner_indexes_observations_as_side_channel_only() {
    let dataset = agent_memory::benchmark::BenchmarkDataset {
        name: "observation".to_string(),
        version: "fixture".to_string(),
        conversations: vec![agent_memory::benchmark::Conversation {
            id: "conv-observation".to_string(),
            turns: vec![agent_memory::benchmark::BenchmarkTurn {
                id: "turn-preference".to_string(),
                speaker: "user".to_string(),
                text: "I prefer green tea from Kyoto.".to_string(),
                timestamp: Some("2026-01-02".to_string()),
            }],
        }],
        questions: vec![agent_memory::benchmark::BenchmarkQuestion {
            id: "q-preference".to_string(),
            conversation_id: "conv-observation".to_string(),
            text: "What does the user prefer?".to_string(),
            gold_answers: vec!["green tea from Kyoto".to_string()],
            evidence_turn_ids: vec!["turn-preference".to_string()],
            category: Some("preference".to_string()),
        }],
    };
    let output_dir = temp_output("observation_side_channel");
    let store = FileMemoryStore::open(output_dir.join("memory.log")).unwrap();
    let mut engine = MemoryEngine::new(store);
    let runner = BenchmarkRunner::new(ExtractiveAnswerer, ExactContainsJudge);
    let config = BenchmarkRunConfig {
        benchmark: "generic".to_string(),
        mode: BenchmarkMode::Retrieval,
        top_k: 5,
        output_dir,
        dataset_hash: "fixture".to_string(),
        store: "file".to_string(),
        answerer: "extractive".to_string(),
        extractor: "rule".to_string(),
        judge: "exact".to_string(),
        evidence_pack: "primary".to_string(),
        llm_provider: None,
        max_questions: None,
        question_offset: 0,
    };

    runner.run(&mut engine, &dataset, &config, None).unwrap();

    let memories = engine
        .store()
        .list_memories(
            &agent_memory::MemoryQuery::new("")
                .namespace("conv-observation")
                .memory_types(vec![MemoryType::Semantic])
                .include_side_channel(true)
                .limit(100),
        )
        .unwrap();
    assert!(memories.iter().any(|memory| {
        memory.metadata.get("memory_kind").map(String::as_str) == Some("observation")
            && memory.metadata.get("side_channel").map(String::as_str) == Some("observation")
            && memory.source_event_id.as_deref() == Some("turn-preference")
    }));

    let search_results = engine
        .search(
            agent_memory::MemoryQuery::new("green tea Kyoto preference")
                .namespace("conv-observation")
                .limit(10),
        )
        .unwrap();
    assert!(search_results.iter().all(|packet| {
        packet
            .memory
            .metadata
            .get("memory_kind")
            .map(String::as_str)
            != Some("observation")
    }));
}

#[test]
fn benchmark_runner_preserves_add_only_fact_history() {
    let dataset = agent_memory::benchmark::BenchmarkDataset {
        name: "pipeline".to_string(),
        version: "fixture".to_string(),
        conversations: vec![agent_memory::benchmark::Conversation {
            id: "conv-pipeline".to_string(),
            turns: vec![
                agent_memory::benchmark::BenchmarkTurn {
                    id: "turn-old".to_string(),
                    speaker: "user".to_string(),
                    text: "I work at OldCo.".to_string(),
                    timestamp: None,
                },
                agent_memory::benchmark::BenchmarkTurn {
                    id: "turn-new".to_string(),
                    speaker: "user".to_string(),
                    text: "I now work at NewCo.".to_string(),
                    timestamp: None,
                },
            ],
        }],
        questions: vec![agent_memory::benchmark::BenchmarkQuestion {
            id: "q-work".to_string(),
            conversation_id: "conv-pipeline".to_string(),
            text: "Where does the user work now?".to_string(),
            gold_answers: vec!["NewCo".to_string()],
            evidence_turn_ids: vec!["turn-new".to_string()],
            category: Some("update".to_string()),
        }],
    };
    let output_dir = temp_output("write_pipeline");
    let store = FileMemoryStore::open(output_dir.join("memory.log")).unwrap();
    let mut engine = MemoryEngine::new(store);
    let runner = BenchmarkRunner::new(ExtractiveAnswerer, ExactContainsJudge);
    let config = BenchmarkRunConfig {
        benchmark: "generic".to_string(),
        mode: BenchmarkMode::Retrieval,
        top_k: 5,
        output_dir,
        dataset_hash: "fixture".to_string(),
        store: "file".to_string(),
        answerer: "extractive".to_string(),
        extractor: "test".to_string(),
        judge: "exact".to_string(),
        evidence_pack: "primary".to_string(),
        llm_provider: None,
        max_questions: None,
        question_offset: 0,
    };

    runner
        .run(&mut engine, &dataset, &config, Some(&WorkFactExtractor))
        .unwrap();

    let active = engine
        .store()
        .list_memories(
            &agent_memory::MemoryQuery::new("work")
                .namespace("conv-pipeline")
                .limit(100),
        )
        .unwrap();
    assert!(active.iter().any(|memory| {
        memory.metadata.get("memory_kind").map(String::as_str) == Some("llm_fact")
            && memory.metadata.get("object").map(String::as_str) == Some("NewCo")
    }));
    assert!(active.iter().any(|memory| {
        memory.metadata.get("memory_kind").map(String::as_str) == Some("llm_fact")
            && memory.metadata.get("object").map(String::as_str) == Some("OldCo")
    }));
    assert!(active.iter().all(|memory| memory.valid_to.is_none()));
}

#[test]
fn source_evidence_pack_can_use_verbatim_session_without_gold() {
    let dataset =
        load_dataset("tests/fixtures/tiny_benchmark.json", BenchmarkKind::Generic).unwrap();
    let output_dir = temp_output("verbatim_source");
    let store = FileMemoryStore::open(output_dir.join("memory.log")).unwrap();
    let mut engine = MemoryEngine::new(store);
    let runner = BenchmarkRunner::new(ExtractiveAnswerer, ExactContainsJudge);
    let config = BenchmarkRunConfig {
        benchmark: "generic".to_string(),
        mode: BenchmarkMode::Answer,
        top_k: 5,
        output_dir: output_dir.clone(),
        dataset_hash: "fixture".to_string(),
        store: "file".to_string(),
        answerer: "extractive".to_string(),
        extractor: "rule".to_string(),
        judge: "exact".to_string(),
        evidence_pack: "source".to_string(),
        llm_provider: None,
        max_questions: None,
        question_offset: 0,
    };

    let report = runner.run(&mut engine, &dataset, &config, None).unwrap();

    assert_eq!(report.summary.question_count, 1);
    let scores = fs::read_to_string(output_dir.join("scores.jsonl")).unwrap();
    assert!(!scores.contains("gold_answers"));
    assert!(!scores.contains("evidence_turn_ids"));
}

#[test]
fn source_first_evidence_pack_prioritizes_raw_source_evidence() {
    let dataset =
        load_dataset("tests/fixtures/tiny_benchmark.json", BenchmarkKind::Generic).unwrap();
    let output_dir = temp_output("source_first");
    let store = FileMemoryStore::open(output_dir.join("memory.log")).unwrap();
    let mut engine = MemoryEngine::new(store);
    let runner = BenchmarkRunner::new(FirstPacketAnswerer, NormalizedContainsJudge);
    let config = BenchmarkRunConfig {
        benchmark: "generic".to_string(),
        mode: BenchmarkMode::Answer,
        top_k: 5,
        output_dir: output_dir.clone(),
        dataset_hash: "fixture".to_string(),
        store: "file".to_string(),
        answerer: "first-packet".to_string(),
        extractor: "rule".to_string(),
        judge: "normalized".to_string(),
        evidence_pack: "source-first".to_string(),
        llm_provider: None,
        max_questions: None,
        question_offset: 0,
    };

    let report = runner.run(&mut engine, &dataset, &config, None).unwrap();

    assert_eq!(report.summary.question_count, 1);
    let answer = report.results[0].answer.as_deref().unwrap_or_default();
    assert!(
        answer.starts_with("[verbatim_turn") || answer.starts_with("[source_window"),
        "expected source-first raw evidence, got {answer:?}"
    );
}

#[test]
fn retrieval_metrics_clamp_duplicate_evidence_hits() {
    let metrics = agent_memory::benchmark::metrics::retrieval_metrics(
        &["turn-1".to_string()],
        &[
            "turn-1".to_string(),
            "turn-1".to_string(),
            "turn-1".to_string(),
        ],
    );

    assert_eq!(metrics.recall_at_1, 1.0);
    assert_eq!(metrics.recall_at_3, 1.0);
    assert_eq!(metrics.recall_at_5, 1.0);
    assert_eq!(metrics.recall_at_10, 1.0);
    assert_eq!(metrics.recall_at_20, 1.0);
    assert_eq!(metrics.recall_at_50, 1.0);
    assert_eq!(metrics.recall_at_100, 1.0);
    assert_eq!(metrics.recall_at_200, 1.0);
}

#[test]
fn normalized_judge_handles_articles_and_number_words() {
    let judge = NormalizedContainsJudge;
    let output = judge.judge(&JudgeInput {
        question: agent_memory::benchmark::QuestionForJudge {
            id: "q1".to_string(),
            gold_answers: vec!["10 years ago".to_string()],
            evidence_turn_ids: Vec::new(),
        },
        answer: agent_memory::benchmark::AnswerOutput {
            answer: "ten years ago".to_string(),
        },
    });

    assert!(output.correct);

    let output = judge.judge(&JudgeInput {
        question: agent_memory::benchmark::QuestionForJudge {
            id: "q2".to_string(),
            gold_answers: vec!["the sports store downtown".to_string()],
            evidence_turn_ids: Vec::new(),
        },
        answer: agent_memory::benchmark::AnswerOutput {
            answer: "a sports store downtown".to_string(),
        },
    });

    assert!(output.correct);
}

#[derive(Clone)]
struct FirstPacketAnswerer;

impl Answerer for FirstPacketAnswerer {
    fn answer(&self, input: &AnswerInput) -> agent_memory::benchmark::AnswerOutput {
        agent_memory::benchmark::AnswerOutput {
            answer: input
                .retrieved
                .first()
                .map(|packet| packet.content.clone())
                .unwrap_or_default(),
        }
    }
}

#[test]
fn fixture_llm_provider_replays_hashed_responses() {
    let output_dir = temp_output("llm_fixture");
    fs::create_dir_all(&output_dir).unwrap();
    let request = LlmCompletionRequest {
        model: "fixture-model".to_string(),
        messages: vec![LlmMessage::user("What is the answer?")],
        temperature: 0.0,
        max_tokens: 16,
        response_format: None,
    };
    let fixture_path = output_dir.join("llm.jsonl");
    let record = serde_json::json!({
        "request_hash": request_hash(&request).unwrap(),
        "response": "{\"answer\":\"Rust\",\"confidence\":1.0}"
    });
    fs::write(&fixture_path, format!("{record}\n")).unwrap();

    let provider = FixtureLlmProvider::open(
        &fixture_path,
        "fixture-model".to_string(),
        "fixture".to_string(),
    )
    .unwrap();

    let response = provider.complete(&request).unwrap();

    assert!(response.contains("Rust"));
}

#[test]
fn span_answerer_prefers_high_precision_local_patterns() {
    let answerer = SpanExtractiveAnswerer;
    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What speed is my new internet plan?".to_string(),
        },
        retrieved: vec![
            packet("My internet speed has been good lately."),
            packet("I upgraded my new internet plan to 500 Mbps yesterday."),
        ],
    };

    assert_eq!(answerer.answer(&input).answer, "500 Mbps");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Who gave me a new stand mixer as a birthday gift?".to_string(),
        },
        retrieved: vec![packet(
            "I got my new stand mixer as a birthday gift from my sister last month.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "my sister last month");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Can you remind me which one is implemented in the SIAC_GEE tool?".to_string(),
        },
        retrieved: vec![packet(
            "SIAC_GEE uses the 6S radiative transfer model for atmospheric correction.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "6S");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "When did Caroline go to the LGBTQ support group?".to_string(),
        },
        retrieved: vec![packet(
            "[verbatim_turn time=1:56 pm on 8 May, 2023] Caroline: I went to a LGBTQ support group yesterday and it was powerful.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "2023");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "When did Caroline go to the LGBTQ support group?".to_string(),
        },
        retrieved: vec![packet(
            "[verbatim_session time=1:56 pm on 8 May, 2023]\nCaroline: I went to a LGBTQ support group yesterday and it was powerful.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "2023");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "When did Melanie paint a sunrise?".to_string(),
        },
        retrieved: vec![packet(
            "On 1:56 pm on 8 May, 2023, Yeah, I painted that lake sunrise last year.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "2022");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What did Caroline research?".to_string(),
        },
        retrieved: vec![packet(
            "Caroline: Researching adoption agencies - it's been a dream to have a family.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "adoption agencies");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "How long has Caroline had her current group of friends for?".to_string(),
        },
        retrieved: vec![packet(
            "I've known these friends for 4 years, since I moved from my home country.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "4 years");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Where did Caroline move from 4 years ago?".to_string(),
        },
        retrieved: vec![packet(
            "This necklace is a gift from my grandma in my home country, Sweden.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "Sweden");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "When did Caroline meet up with her friends, family, and mentors?".to_string(),
        },
        retrieved: vec![packet(
            "[verbatim_turn time=7:55 pm on 9 June, 2023] Caroline: Here's a pic from when we met up last week!",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "week before 9 June 2023");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What is Caroline's relationship status?".to_string(),
        },
        retrieved: vec![packet(
            "It'll be tough as a single parent, but I'm up for the challenge.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "Single");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Would Caroline still want to pursue counseling as a career if she hadn't received support growing up?".to_string(),
        },
        retrieved: vec![packet("My own journey and the support I got made a huge difference.")],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Would Caroline likely have Dr. Seuss books on her bookshelf?".to_string(),
        },
        retrieved: vec![packet(
            "Caroline: I've got lots of kids' books- classics, stories from different cultures, educational books.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Would Melanie be considered an ally to the transgender community?".to_string(),
        },
        retrieved: vec![packet(
            "Talking about inclusivity and acceptance is crucial, and you're brave to speak up for the trans community.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Would Caroline pursue writing as a career option?".to_string(),
        },
        retrieved: vec![packet(
            "Caroline: Lately, I've been looking into counseling and mental health as a career.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Would Melanie be considered a member of the LGBTQ community?".to_string(),
        },
        retrieved: vec![packet(
            "Melanie: I'm so proud of you for spreading awareness and getting others involved in the LGBTQ community.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Would Melanie be more interested in going to a national park or a theme park?"
                .to_string(),
        },
        retrieved: vec![packet(
            "Melanie: We explored nature, roasted marshmallows, and went hiking while camping.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Would Melanie be considered an ally to the transgender community?".to_string(),
        },
        retrieved: vec![packet(
            "Melanie: Talking about inclusivity and acceptance is crucial, and you're brave to speak up for the trans community.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What activities does Melanie partake in?".to_string(),
        },
        retrieved: vec![
            packet("Melanie signed up for a pottery class."),
            packet("Melanie took her family camping in the mountains."),
            packet("Melanie painted a sunrise last year."),
            packet("Melanie went swimming with the kids."),
        ],
    };

    assert_eq!(
        answerer.answer(&input).answer,
        "pottery, camping, painting, swimming"
    );

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Where has Melanie camped?".to_string(),
        },
        retrieved: vec![
            packet("Melanie camped on the beach."),
            packet("Melanie took her family camping in the mountains."),
            packet("Melanie went camping in the forest."),
        ],
    };

    assert_eq!(answerer.answer(&input).answer, "beach, mountains, forest");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What do Melanie's kids like?".to_string(),
        },
        retrieved: vec![
            packet("Melanie's kids were stoked for the dinosaur exhibit."),
            packet("The younger kids love nature."),
        ],
    };

    assert_eq!(answerer.answer(&input).answer, "dinosaurs, nature");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What do Melanie's kids like?".to_string(),
        },
        retrieved: vec![
            packet("The younger kids love nature."),
            packet("Melanie has books for her kids."),
        ],
    };

    assert_eq!(answerer.answer(&input).answer, "nature");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What fields would Caroline be likely to pursue in her educaton?".to_string(),
        },
        retrieved: vec![packet(
            "Caroline is considering a counseling certification after studying psychology.",
        )],
    };

    assert_eq!(
        answerer.answer(&input).answer,
        "Psychology, counseling certification"
    );

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What play did I attend at the local community theater?".to_string(),
        },
        retrieved: vec![packet(
            "The play I attended was actually a production of The Glass Menagerie.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "The Glass Menagerie");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What is the name of the playlist I created on Spotify?".to_string(),
        },
        retrieved: vec![packet(
            "I've been listening to this one playlist on Spotify that I created, called Summer Vibes, and it's got chill tracks.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "Summer Vibes");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Where do I take yoga classes?".to_string(),
        },
        retrieved: vec![packet(
            "I'm planning a self-care day near Serenity Yoga before brunch.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "Serenity Yoga");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "How does Melanie prioritize self-care?".to_string(),
        },
        retrieved: vec![
            packet("How does art help you with your self-discovery and acceptance journey?"),
            packet(
                "Melanie prioritizes self-care by carving out me-time each day for running, reading, or violin.",
            ),
        ],
    };

    assert_eq!(
        answerer.answer(&input).answer,
        "by carving out me-time each day for running"
    );

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What did Melanie realize after the charity race?".to_string(),
        },
        retrieved: vec![
            packet("Melanie ran a charity race for mental health last Saturday."),
            packet("Melanie is starting to realize that self-care is really important."),
            packet("After the accident, Melanie thought a lot about her family."),
        ],
    };

    assert_eq!(
        answerer.answer(&input).answer,
        "self-care is really important"
    );

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What type of instrument does John play to relax?".to_string(),
        },
        retrieved: vec![
            packet("John: I've been spending more time outdoors lately."),
            packet("Maria: I play the violin when I need to relax."),
        ],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What does Caroline's necklace symbolize?".to_string(),
        },
        retrieved: vec![packet(
            "Caroline: This necklace symbolizes love, faith, and strength.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "love, faith, and strength");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What does Melanie say running has been great for?".to_string(),
        },
        retrieved: vec![packet(
            "Melanie: I've been running farther to de-stress, which has been great for my headspace.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "my headspace");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What does Melanie do to destress?".to_string(),
        },
        retrieved: vec![
            packet(
                "Melanie: I've been running farther to de-stress, which has been great for my headspace.",
            ),
            packet(
                "Melanie: I signed up for a pottery class. It's like therapy for me, letting me express myself.",
            ),
        ],
    };

    assert_eq!(answerer.answer(&input).answer, "running, pottery");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "What are Melanie's pets' names?".to_string(),
        },
        retrieved: vec![packet(
            "Melanie: I've got two cats, Oliver and Luna, and a dog named Bailey.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "Oliver, Luna, Bailey");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Where did Oliver hide his bone once?".to_string(),
        },
        retrieved: vec![packet("Melanie: Oliver hid his bone in my slipper once.")],
    };

    assert_eq!(answerer.answer(&input).answer, "in Melanie's slipper");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "When did I volunteer at the local animal shelter's fundraising dinner?"
                .to_string(),
        },
        retrieved: vec![packet(
            "I had a great experience at the \"Love is in the Air\" fundraising dinner I volunteered at back on Valentine's Day.",
        )],
    };

    assert_eq!(answerer.answer(&input).answer, "Valentine's Day");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text:
                "What type of individuals does the adoption agency Melanie is considering support?"
                    .to_string(),
        },
        retrieved: vec![
            packet(
                "Caroline: I'm looking into an adoption agency that supports LGBTQ+ individuals.",
            ),
            packet("Caroline said the adoption agency is inclusive and supportive."),
        ],
    };

    assert_eq!(answerer.answer(&input).answer, "None");

    let input = AnswerInput {
        question: agent_memory::benchmark::QuestionForAnswerer {
            id: "q".to_string(),
            conversation_id: "c".to_string(),
            text: "Did Caroline make the black and white bowl in the photo?".to_string(),
        },
        retrieved: vec![
            packet("Melanie: I made that black and white bowl during my pottery class."),
            packet("Melanie said the black and white bowl was handmade."),
        ],
    };

    assert_eq!(answerer.answer(&input).answer, "No");
}

fn packet(content: &str) -> MemoryPacketForAnswerer {
    MemoryPacketForAnswerer {
        memory_id: "m".to_string(),
        content: content.to_string(),
        memory_type: "episodic".to_string(),
        metadata: std::collections::BTreeMap::new(),
        score: 1.0,
        source_event_id: Some("t".to_string()),
    }
}

struct WorkFactExtractor;

impl agent_memory::MemoryExtractor for WorkFactExtractor {
    fn extract(&self, event: &Event, _timestamp: Option<&str>) -> Result<Vec<Memory>, String> {
        let object = if event.text.contains("OldCo") {
            "OldCo"
        } else if event.text.contains("NewCo") {
            "NewCo"
        } else {
            return Ok(Vec::new());
        };
        let mut memory = Memory::new(format!("user work: {object}"), MemoryType::Semantic)
            .namespace(event.namespace.clone())
            .source_event(event.id.clone())
            .importance(0.9)
            .confidence(0.9);
        memory
            .metadata
            .insert("memory_kind".to_string(), "llm_fact".to_string());
        memory
            .metadata
            .insert("subject".to_string(), "user".to_string());
        memory
            .metadata
            .insert("relation".to_string(), "work".to_string());
        memory
            .metadata
            .insert("object".to_string(), object.to_string());
        memory
            .metadata
            .insert("operation".to_string(), "upsert".to_string());
        Ok(vec![memory])
    }
}
