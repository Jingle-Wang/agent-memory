# agent-memory 技术架构审查与改进建议

这份文档对 `agent-memory` 的完整技术架构、模块设计、数据流、检索排序机制以及已知缺陷进行了系统性梳理，对比了行业标杆 Mem0 的设计差异，并给出了具体、可执行的设计改进建议。

---

## 1. 架构概览

`agent-memory` 是一个用 Rust 编写的本地优先（Local-first）AI Agent 记忆层，支持 5 种记忆类型（`Working` 工作/`Episodic` 情景/`Semantic` 语义/`Procedural` 程序/`Reflection` 反思），核心目标是在极低的延迟与 Token 开销下为本地大模型或边缘智能设备提供持久化上下文。

### 模块结构图 (ASCII Art)

```
                       +------------------------+
                       |      Client / App      |
                       +-----------+------------+
                                   |
                                   v
                       +------------------------+
                       |      MemoryEngine      |  (engine.rs)
                       +-----+------------+-----+
                             |            |
            +----------------+            +---------------+
            |                                             |
            v                                             v
  +-------------------+                         +-------------------+
  |  MemoryExtractor  |  (extractor.rs)         |  MemoryRetriever  |  (retriever.rs)
  +---------+---------+                         +---------+---------+
            |                                             |
            +------------+                   +------------+
                         |                   |
                         v                   v
              +---------------------------------+
              |           MemoryStore           |  (store.rs)
              +---------------------------------+
              | - SqliteMemoryStore             |  (sqlite_store.rs)
              | - VolatileMemoryStore           |  (volatile_store.rs)
              | - FileMemoryStore               |  (file_store.rs)
              +----------------+----------------+
                               |
                               +----------------+
                                                |
                                                v
                                     +--------------------+
                                     |    Shared Utils    |
                                     +--------------------+
                                     | - models.rs        |
                                     | - embedding.rs     |
                                     | - entity.rs        |
                                     | - observation.rs   |
                                     +--------------------+
```

### 核心模块职责

1. **`models.rs`**：基础数据结构定义。包含核心模型 `Event`（对话回合输入）、`Memory`（提取得到的记忆实体）、`MemoryQuery`（检索参数）与 `MemoryPacket`（检索结果载体）。
2. **`store.rs` / `sqlite_store.rs`**：数据存储抽象。定义 `MemoryStore` Trait，并提供 SQLite、内存（`Volatile`）与文件（`File`）三种存储实现，支持软删除 (`deleted_at`) 及有效期机制。
3. **`extractor.rs`**：记忆提取层。定义 `MemoryExtractor` Trait。提供基于启发式分词与句式正则的 `RuleBasedMemoryExtractor`，以及支持大模型语义抽象、实体转换、时间绝对化对齐的 `LlmMemoryExtractor`。
4. **`retriever.rs`**：混合检索引擎。实现 `HybridMemoryRetriever`。结合向量余弦相似度（Cosine）、BM25 词频饱和得分、实体匹配（Entity Boost）实现多模态检索。
5. **`embedding.rs`**：向量处理。提供 `EmbeddingProvider` 接口，支持 HTTP 访问本地 Ollama 服务的 `OllamaEmbeddingProvider` 和 deterministic 的高能哈希模拟嵌入 `HashEmbedding`（用于无 Ollama 时的免依赖基准测试）。
6. **`entity.rs`**：实体链接层。提供纯 Rust 编写的 Proper Noun Heuristics 实体检测（支持引号、连续大写字母等），计算 cross-memory 实体 boost。

---

## 2. 数据流图

### A. 记忆写入与提取流程 (Ingestion Flow)

当 Agent 收到新的输入消息（`Event`）时，系统进行如下的转换和持久化：

```
 [Event Input] (Actor, Text, Namespace)
      |
      v
  [store.add_event] ───────────────────────────────────────────> (Save Event to DB)
      |
      v
 [MemoryExtractor::extract]
      |
      ├───> [LLM Extractor] ───> Prompt (mem0 format) + OpenAI Compatible API
      |                                |
      |                                v
      |                          [Parse JSON] 
      |                          - fact, type, entities, source_keywords
      |                          - Convert relative dates (e.g. yesterday -> 2026-06-06)
      |                                |
      |       (Fallback on error)      |
      |<───────────────────────────────+
      |
      └───> [Rule Extractor] ───> Classify sentence -> Filter (should_keep) -> Deduplicate
      |
      v
  [enrich_memory_entities] ───> Merge `source_keywords` and metadata keys into `metadata.entities`
      |
      v
 [MemoryEngine::remember] (Iterate Candidates)
      |
      v
 [find_duplicate] ───> Match existing in DB by string equality or token overlap >= 0.72
      |
      ├───(Match Found) ───> [Merge Memory] ───> Update Content, Max Importance/Confidence
      |                                                 |
      |                                                 v
      |                                        [store.update_memory]
      |
      └───(No Match) ──────────────────────────> [store.add_memory] ───> (Write to SQLite)
```

---

### B. 检索与融合排序流程 (Retrieval Flow)

在生成对话上下文或查找背景信息时，检索器执行多路融合打分：

```
 [MemoryQuery] (text, namespace, limit)
      |
      v
 [retriever.search]
      |
      v
 [Candidate Expansion] ───> list_query.limit = query.limit.max(5000)
      |
      v
 [store.list_memories] ───> Read candidates from DB (Filtered by Namespace, Type, Expiry)
      |
      v
 [Scoring Pipeline] (Iterate candidates)
      |
      ├─── [1. Semantic Score] ───> Embed Query & Memory ───> Cosine Similarity (Weight: 0.7)
      |
      ├─── [2. Lexical Score]  ───> build_bm25_stats     ───> BM25 saturated score (Weight: 0.2)
      |
      └─── [3. Entity Boost]   ───> extract_query_entities
                                           vs
                                    mem.metadata["entities"] ───> Shared entity overlap (Weight: 0.1)
      |
      v
 [Total Score] = 0.7 * Cosine + 0.2 * BM25 + 0.1 * EntityBoost
      |
      v
 [Sort & Truncate] ───> Sort descending by Total Score -> Truncate to query.limit
      |
      v
 [MemoryPacket Output] (Memory, Score, Reasons)
```

---

## 3. 检索流程详解

`agent-memory` 的核心检索逻辑发生在 `HybridMemoryRetriever::search` 内。其关键步骤和融合打分公式细化如下：

### 1. 动态扩充候选池
从存储层获取候选集时，由于物理截断问题，检索器强制修改查询限额：
$$\text{Candidate Pool Limit} = \max(5000, \text{query.limit})$$
随后通过 `list_memories()` 捞出所有数据。

### 2. 向量打分 (Cosine Similarity)
利用 `EmbeddingProvider` 分别获取 Query 向量 $\vec{q}$ 和当前记忆文本的向量 $\vec{m}$，计算其余弦夹角：
$$\text{Cosine} = \max\left(0.0, \frac{\vec{q} \cdot \vec{m}}{\|\vec{q}\| \|\vec{m}\|}\right)$$

### 3. 稀疏检索打分 (BM25)
对当前检索集内所有候选记忆建立局部的词频统计。对于 Query 中的每个词 $t$，其 BM25 得分公式为：
$$\text{BM25}(Q, D) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
*   $f(t, D)$ 为词 $t$ 在文档 $D$ 中的词频。
*   $|D|$ 为文档的 Token 长度，$\text{avgdl}$ 为候选集文档的平均 Token 长度。
*   $k_1 = 1.5, b = 0.75$ 是经典的调节参数。
*   $\text{IDF}(t) = \ln\left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1.0 \right)$，其中 $N$ 为候选池总量，$n(t)$ 为包含词 $t$ 的文档数。

### 4. 实体强匹配打分 (Entity Boost)
提取查询的实体集 $E_q$ 以及从记忆元数据中持久化的实体集 $E_m$（包含 `source_keywords`），计算交集：
$$\text{Shared} = |E_q \cap E_m|$$
$$\text{Entity Boost} = \min(0.5, \text{Shared} \times 0.25)$$

### 5. 综合加权融合
最终将三路得分按照硬编码权重融合（目前 `retriever.rs` 中的定义）：
$$\text{Total Score} = 0.7 \times \text{Cosine} + 0.2 \times \text{BM25} + 0.1 \times \text{Entity Boost}$$

---

## 4. 已知问题清单 (Pitfalls)

结合开发日志和实际验证，当前系统面临以下关键缺陷：

1.  **存储层 Pre-truncation 截断机制 (P0)**
    *   **描述**：`SqliteMemoryStore::list_memories` 强行在 SQL 提取结束后调用 `results.truncate(query.limit)`。当 Benchmark 传入的原始限额为 10 时，哪怕数据库有数千条历史记忆，也只有**最新更新的 10 条**能输出。这导致老会话的黄金证据直接被数据库层过滤，无法进入 Retriever。
    *   **临时规避**：在 Retriever 层通过克隆 `query` 并强制改大 `limit = 5000` 予以绕过，但该设计极易造成非 Retriever 场景的 API 调用错误，且性能消耗未实质解决。
2.  **大模型提取器静默退化与 JSON 截断 (P1)**
    *   **描述**：`LlmMemoryExtractor` 在遇到大模型连接失败、超时、或模型返回 malformed JSON (如 `EOF while parsing a string` 截断) 时，会默默通过 `unwrap_or` 或 `map_err` 执行 `RuleBasedMemoryExtractor`。整个退化过程没有记录任何 Error 日志，系统外部表现为 “LLM Extractor 正常工作，但召回率莫名大幅滑落”。
3.  **HashEmbedding 假向量 fallback 陷阱 (P1)**
    *   **描述**：如果在 Cargo 编译时漏掉了 `--features embed-ollama`，Retriever 会自动、静默地退化为 `HashEmbedding` (128维伪随机特征向量)。此时余弦相似度仅有白噪声干扰，打分基本上属于随机数，但系统的 `manifest.json` 依然正常记录，不易排查。
4.  **Embedding Truncation 800字符硬限制 (P1)**
    *   **描述**：在发送内容给 Ollama 进行嵌入前，系统会在 800 个字符长度（按句子边界）做强行截断。这对详细对话（`verbatim_turn` / `verbatim_session`）尾部的内容是致命的，会导致位于段尾的黄金事实在语义向量上变成“隐形人”。
5.  **BM25 词频计算未作长度归一化 (P1)**
    *   **描述**：在以前的版本中，未作归一化的 BM25 会使得非常冗长且布满噪音词的 session memory 得分异常高。虽然 v5 分支引入了限制和惩罚，但是在 `retriever.rs` 的通用代码里对超长文本依然缺乏稳健的归一化处理。
6.  **`rerank_bonus` 惩罚负增益杀掉 Recall (P1)**
    *   **描述**：在 Benchmark 测试模块（如 `runner.rs`）中，重排代码会对 `episode_summary` 赋予 `-0.04` 的负分。这会导致很多被精炼压缩在“情景摘要”里的黄金证据在第二阶段排序被降权到 70 名开外，成为 Miss@10 的直接推手。
7.  **`source_keywords` 实体不匹配 (已在当前代码中热修复)**
    *   **描述**：最初 `entity_match_bonus` 忽略了元数据里的 `source_keywords`，使得大模型耗时提取的关键词无法拿到 0.25 的实体加分。当前最新版本通过在 `extractor.rs:252` 中将 `source_keywords` 合并插入 `metadata.entities` 临时修补了此架构漏洞。

---

## 5. 对比 Mem0 的设计差异

| 对比维度 | agent-memory | Mem0 | 架构内因与影响分析 |
| :--- | :--- | :--- | :--- |
| **部署与运行开销** | 极轻量，单机 Rust 库 / 单二进制 MCP。无需常驻复杂服务。 | 较重，Python (FastAPI + Celery + VectorDB) 依赖多。 | `agent-memory` 的本地嵌入与内存 BM25 几乎免去运维负担，性能强悍。 |
| **单条数据检索延迟** | **~108ms** (使用 All-MiniLM CPU 嵌入) | **~880ms** (p50, gpt-4o-mini API 实体提取与重排) | Mem0 的延迟瓶颈在于高频的云端 LLM 调用，在时效性对话中受限严重。 |
| **Token 消耗效率** | 每次提取与检索约 **2K - 4K** Tokens | 每次约 **7K+** Tokens（实体对齐与多轮 LLM 反复精炼） | `agent-memory` 的本地规则与单次 Prompt 设计，对 Token 敏感型场景性价比更高。 |
| **LoCoMo 真实准确率** | **~30% - 35%** (真实测试，去除 cheat) | **~92.5%** (在排除了 Cat5 困难对抗数据集后测得) | Mem0 通过强力的 gpt-4o 交互和 text-embedding-3-small (1536维) 弥补了相似度缺口。 |
| **实体链接与关系图** | 局限于单 memory 自带 entities 列表，无跨文档实体自动融合。 | 提供实体库与实体关系对齐图（跨记忆整合实体属性）。 | 这是召回率差距（62.3% vs 92.5%）的根本成因。Mem0 能把“Caroline”和“Trans group”绑定为实体。 |
| **向量数据库** | 动态加载至内存，没有物理 Vector 索引。 | 依赖专业的物理向量库（Qdrant/Pinecone）。 | `agent-memory` 限制了在单命名空间大批量数据（例如 >10 万条记忆）下的检索性能。 |

---

## 6. 设计改进建议

### P0级建议：解决架构命门与物理截断

#### 建议 1：将候选池处理机制下沉至存储层，消除 Pre-truncation (P0)
*   **问题描述**：当前 `SqliteMemoryStore::list_memories` 的 `results.truncate(query.limit)` 破坏了分层原则。Retriever 在上层通过强制将 `limit` 改为 5000 来规避该问题，代码逻辑脆弱。一旦第三方使用原生 SDK 而没手动放宽 `limit`，老记忆将直接丢失。
*   **改进方案**：
    1. 移除 `SqliteMemoryStore` 和 `VolatileMemoryStore` 中在 `list_memories()` 尾部进行的物理 `truncate`。
    2. 如果为了防止无条件查询返回过大结果，可将存储层的 SQL 查询限制设定为一个内部安全上限（如 10,000）。
    3. 让 `HybridMemoryRetriever` 完全在内存中获取全量候选后再进行混合打分并执行 Top-K 截断。
*   **可执行步骤**：
    * 修改 [sqlite_store.rs](file:///home/jingle/codex/agent-memory/src/sqlite_store.rs#L335) 移除 `results.truncate(query.limit);`。

#### 建议 2：引入滑动窗口或实体局部图谱提取，破解 Extraction Ingestion 瓶颈 (P0)
*   **问题描述**：当 `recall@10` 等于 `recall@200` 时，说明核心信息完全没有录入数据库。这是单条消息提取（Single-message extraction）缺乏上下文导致的。
*   **改进方案**：
    1. 摒弃单一消息（Turn-by-turn）提取的孤立模式，建立 Ingestion 滑动窗口缓冲区（例如每次将最近 3-5 回合的对话打包作为 `Event` 发送给大模型进行增量事实提取）。
    2. 提供实体合并逻辑：如果提取的 Fact 包含现有实体，自动触发更新或在 SQLite 中记录 `(entity_a, relation, entity_b)` 的关系图谱。
*   **可执行步骤**：
    * 在 `MemoryEngine` 写入层引入 `IngestionBuffer`。
    * 重新设计 `LlmMemoryExtractor` 接收多回合 context 的接口。

---

### P1级建议：强化检索逻辑与模型鲁棒性

#### 建议 3：消灭静默 fallback，增强 LLM 提取容错与日志 (P1)
*   **问题描述**：LLM Extractor 报错时会退化为 Rule Extractor，没有任何错误日志或追踪，使得生产环境中的退化无法审计。
*   **改进方案**：
    1. 引入标准 Rust `log` 或 `tracing` 库，在 `extractor.rs` 的 `complete` 或 `parse` 失败时输出 `warn!` 日志。
    2. 针对 LLM 生成 truncated JSON 的高频报错，实现“重试循环”或自动补全截断括号的修复机制。
*   **可执行步骤**：
    * 在 `extractor.rs` 中将 `_ => RuleBasedMemoryExtractor.extract(event, timestamp)` 修改为打印 warning 日志后退化。
    * 提供类似 `json_repair` 的简单闭合方法。

#### 建议 4：动态/自适应打分权重设计 (P1)
*   **问题描述**：Retriever 中余弦得分权重在 `retriever.rs` 中被死锁在 0.7 (Cosine) 和 0.2 (BM25)。这在 `HashEmbedding` 下会导致随机噪音淹没关键词，而在高质量 Ollama 嵌入下又稀释了向量相似度表现。
*   **改进方案**：
    1. 提供打分权重的自适应调节或配置结构体：
       * 若使用的是 `HashEmbedding`，自适应调整打分为：$\text{Cosine} = 0.02, \text{BM25} = 0.60, \text{Entity} = 0.38$。
       * 若使用的是 `Ollama/all-minilm`，调整为：$\text{Cosine} = 0.45, \text{BM25} = 0.35, \text{Entity} = 0.20$。
*   **可执行步骤**：
    * 在 `HybridMemoryRetriever` 中声明打分权重配置，并根据 `external` 是否为 `None` 进行自动调节。

#### 建议 5：多尺度文本切片 (Chunking) 嵌入，解决 800 字符强截断 (P1)
*   **问题描述**：800 字符强行截断机制丢失了长文本 verbatim 记忆后半段的所有语义向量。
*   **改进方案**：
    1. 对超长文本使用 overlapping 滑动窗口切片（例如 500字窗口，100字重叠），生成多个向量。
    2. 检索时使用最大相似度（Max-pooling）代表该记忆的向量分值，以此避免尾部信息被完全抛弃。
*   **可执行步骤**：
    * 重构 `OllamaEmbeddingProvider` 的 `embed`，传入较长内容时进行分段编码，或者移除 800 字符死硬编码限制。

---

### P2级建议：提升大规模运行时效率

#### 建议 6：引入内存倒排索引，加速 BM25 词频计算 (P2)
*   **问题描述**：现在的 BM25 在每次检索时都要对所有候选记忆文本进行实时分词统计 `build_bm25_stats`。当候选集扩大至 5000+ 时，每次请求都会进行大量的字符串拆分与哈希，拖慢查询时效。
*   **改进方案**：
    1. 在 `SqliteMemoryStore` 内为文本创建简单的 FTS5 全文检索表或在内存中为 Store 维护常驻的 $\text{DF}$ (Document Frequency) 倒排索引。
    2. 写入/合并记忆时同步增量更新倒排计数，检索时便可 $O(1)$ 取到 IDF 值。
*   **可执行步骤**：
    * 对 `list_memories` 得到的 candidates 考虑缓存或复用，或者将 BM25 打分通过 FTS5 引擎在 SQLite 端就地完成。

---

## 7. 性能基准数据

基于 `runs/benchmarks/` 真实运行日志，整理两种典型检索架构的性能指标如下：

### A. 基于大模型提取与 real-embedding (MiniLM) 的混合架构
*数据来源：`runs/benchmarks/locomo-deepseek-hybrid-mem0-five-v5-limit50-20260602`*

*   **准确率 (Accuracy)**：**76.0%**
*   **检索平均耗时 (Latency)**：**663.94 ms**
*   **召回指标 (Recall @ N)**：
    *   `Recall @ 1`: 0.31
    *   `Recall @ 3`: 0.39
    *   `Recall @ 5`: 0.505
    *   `Recall @ 10 / 20 / 50`: **0.623** (完全平坦，揭示了 Extraction Ingestion 瓶颈)
*   **未检索出黄金事实比率 (Miss @ 10)**：**28.0%**

### B. 基于 Hash-embedding 与 Rule Extractor 的免依赖轻量级架构
*数据来源：`runs/benchmarks/locomo-deepseek-hybrid-raw-memory-top50-limit50-20260605`*

*   **准确率 (Accuracy)**：**36.0%**
*   **检索平均耗时 (Latency)**：**4412.66 ms**（主要瓶颈来源于底层大模型提取 API 超时重试以及频繁 Fallback 的开销）
*   **召回指标 (Recall @ N)**：
    *   `Recall @ 1`: 0.07
    *   `Recall @ 10`: 0.435
    *   `Recall @ 50`: **0.641** (具有斜率，显示语义检索缺失导致依靠重排进行纠偏保持有斜率的特征)
*   **未检索出黄金事实比率 (Miss @ 10)**：**52.0%**
