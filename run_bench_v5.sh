#!/bin/bash
set -e

# Source the .env file
set -a
source <(grep -v '^#' ~/.hermes/.env | grep -v '^$')
set +a

# Override with benchmark-specific values
export AGENT_MEMORY_LLM_PROVIDER=openai-compatible
export AGENT_MEMORY_LLM_MODEL=deepseek-v4-pro
export AGENT_MEMORY_LLM_API_KEY="$DEEPSEEK_API_KEY"
export AGENT_MEMORY_LLM_BASE_URL="$DEEPSEEK_BASE_URL/v1"
export AGENT_MEMORY_EMBEDDING_MODEL=all-minilm

echo "=== Environment ==="
echo "DEEPSEEK_BASE_URL=$DEEPSEEK_BASE_URL"
echo "AGENT_MEMORY_LLM_BASE_URL=$AGENT_MEMORY_LLM_BASE_URL"
echo "AGENT_MEMORY_LLM_MODEL=$AGENT_MEMORY_LLM_MODEL"
echo "AGENT_MEMORY_EMBEDDING_MODEL=$AGENT_MEMORY_EMBEDDING_MODEL"
echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:0:20}..."
echo "===================="

rm -rf runs/goal80-v5
mkdir -p runs/goal80-v5

cd /home/jingle/codex/agent-memory

cargo run --release --features sqlite,llm-http,embed-ollama,benchmark \
  --bin agent-memory-bench -- \
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

echo "=== Benchmark Complete ==="
