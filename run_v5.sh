#!/bin/bash
set -a
source ~/.hermes/.env
set +a

export AGENT_MEMORY_LLM_PROVIDER=openai-compatible
export AGENT_MEMORY_LLM_MODEL=deepseek-v4-flash
export AGENT_MEMORY_LLM_API_KEY=$DEEPSEEK_API_KEY
export AGENT_MEMORY_LLM_BASE_URL=$DEEPSEEK_BASE_URL/v1
export AGENT_MEMORY_EMBEDDING_MODEL=all-minilm

cd /home/jingle/codex/agent-memory
rm -rf runs/goal80-v5

exec ./target/release/agent-memory-bench \
  --benchmark locomo \
  --dataset data/benchmarks/locomo/locomo10-conv26.json \
  --mode answer \
  --extractor llm \
  --answerer llm-hybrid \
  --evidence-pack source \
  --judge normalized \
  --store sqlite \
  --limit 20 \
  --output runs/goal80-v5
