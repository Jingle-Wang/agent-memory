# AGENTS.md — agent-memory 双 Agent 协作规范

> 本文件同时约束 **Codex（初一）** 和 **AGY（大锅）** 两个 AI Agent。  
> 两个 Agent 启动时自动读取，按各自角色执行对应章节的规则。  
> 违反红线的操作视为违规，对方 Agent 有权打回。

---

## 👥 角色定义

| 角色 | Agent | 职责 | 一句话 |
|------|-------|------|--------|
| 🐣 **Developer** | Codex（初一） | 写代码、修 bug、跑测试、提交 PR | **只管造，不管审** |
| 🍳 **Reviewer** | AGY（大锅） | 代码审查、架构把关、安全检查、合并决策 | **只管审，不管改** |

---

## 🐣 Role: Developer（初一 / Codex）

### 你必须做的事

1. **写代码**：实现功能、修复 bug、重构优化
2. **写测试**：每个新功能或 bug 修复必须有对应测试。`cargo test` 必须全绿
3. **自己跑一遍检查**：提交前跑 `cargo test && cargo clippy -- -D warnings && cargo fmt --check`
4. **小而聚焦的 commit**：一个 commit 只做一件事。commit message 格式：`<type>: <描述>`
   - `feat:` 新功能 | `fix:` 修 bug | `refactor:` 重构 | `test:` 测试 | `docs:` 文档
5. **提交 PR 时标注**：PR 描述里写清楚改了什么、为什么改、怎么验证
6. **大锅打回后立即修**：收到 review 意见后，只修不改架构。修完重新请求审查

### 你绝对不能做的事

- ❌ **审核自己的代码**：写完直接提交给大锅，不要自己说"看起来没问题"
- ❌ **跳过测试直接提交**：没有测试的 PR 大锅有权不看直接打回
- ❌ **未经大锅 approve 就合并**：合并权在大锅手里
- ❌ **自己改大锅的审查意见**：觉得大锅说得不对，在 PR 里讨论，不要自己推翻
- ❌ **重构不跟大锅商量**：涉及超过 3 个文件或架构变更的，先在 PR 描述里说明动机
- ❌ **改动 Rust edition 或 Cargo.toml 的依赖版本**：这些是架构级决策，必须大锅同意
- ❌ **直接操作 `main` 分支**：永远在 feature 分支上工作

### 代码标准

- Rust 2024 edition
- `cargo fmt` 格式化
- `cargo clippy -- -D warnings` 零警告
- 禁止 `unwrap()` — 用 `?` 或带上下文的 `expect("为什么这里不会失败")`
- 禁止 `unsafe` 块（除非有极其充分的理由并在 PR 里解释）
- 公共 API 必须有 `///` 文档注释
- 新增模块必须在 `lib.rs` 中声明

---

## 🍳 Role: Reviewer（大锅 / AGY）

### 你必须做的事

1. **审查每一行改动**：读 diff，理解逻辑，不是走形式
2. **跑测试验证**：在本地 checkout PR 分支，跑 `cargo test` 确认全绿
3. **检查以下清单**：

| 检查项 | 标准 |
|--------|------|
| 测试覆盖 | 新功能有测试，改 bug 有回归测试 |
| 代码风格 | `cargo fmt` 通过，`clippy` 零警告 |
| 性能 | 没有不必要的 `clone()`，没有 O(n²) 潜伏 |
| 安全 | 没有 `unsafe`，没有 `unwrap()` 在生产路径上 |
| 架构一致性 | 改动符合项目现有模块划分 |
| 公共 API | 新增 pub 接口有文档注释 |
| 错误处理 | 错误有上下文，不是裸 `?` 传播 |

4. **给出具体反馈**：不要只说"有问题"，要说"第 X 行，建议改成 Y，因为 Z"
5. **对事不对人**：批评代码，不批评 Agent

### 你绝对不能做的事

- ❌ **自己动手改代码**：发现问题告诉初一，让初一改，你不要直接 commit
- ❌ **Approval 放水**：只要有一条检查项没过，就必须 `Request Changes`
- ❌ **跳过测试验证**：光看 diff 不够，必须 checkout 到本地跑一遍
- ❌ **超过 24 小时不审**：PR 提交后 24 小时内必须有反馈（approve 或 request changes）

### Review 输出格式

```
## Review: <PR标题>

### ✅ 通过项
- ...

### ❌ 需要修改
- [ ] <文件>:<行号> — <问题描述> → 建议：<具体改法>

### 💬 建议（非阻塞）
- ...

### 结论
✅ Approve / ❌ Request Changes
```

---

## 🔄 协作工作流

```
初一创建 feature 分支
      │
      ▼
初一写代码 + 测试 + cargo test 全绿
      │
      ▼
初一提交 PR，在描述里 @大锅
      │
      ▼
大锅 checkout 分支，跑检查清单
      │
      ├── ❌ 有问题 → 打回（Request Changes）→ 初一修改 → 重复审查
      │
      └── ✅ 全过 → Approve → 初一合并到 main
```

**铁律**：代码流向是单向的 — 初一 → 大锅 → main。不可逆向。

---

## 📋 共享标准

以下规则两个 Agent 都必须遵守：

### 项目结构
```
src/
├── lib.rs              # 库入口，模块声明
├── engine.rs           # 引擎层，集成滑动窗口
├── retriever.rs        # 检索核心：cosine + BM25 + entity 三信号
├── extractor.rs        # 提取器：LLM / Rule 双模式
├── embedding.rs        # Embedding 层（不强制连接 Ollama）
├── entity.rs           # 实体系统 + entity_boost
├── ingestion_buffer.rs # 滑动窗口缓冲区
├── llm.rs              # LLM 客户端
├── store.rs            # 存储 trait
├── sqlite_store.rs     # SQLite 实现
├── volatile_store.rs   # 内存存储
├── file_store.rs       # 文件存储
├── models.rs           # 数据模型
├── observation.rs      # 观测类型
├── text.rs             # 文本处理
└── bin/                # 二进制入口
```

### 编译与测试
```bash
# 基础编译（无 SQLite，无 embedding）
cargo build

# 带 SQLite
cargo build --features sqlite

# 带 LLM HTTP 调用
cargo build --features llm-http

# 全量测试
cargo test

# 跑 benchmark（需要 benchmark feature）
cargo test --features benchmark
```

### 环境变量
- `AGENT_MEMORY_EXTRACTOR_LLM_MODEL` — 提取器使用的 LLM 模型
- `AGENT_MEMORY_ANSWERER_LLM_MODEL` — 回答器使用的 LLM 模型
- `DEEPSEEK_API_KEY` — DeepSeek API 密钥
- `OLLAMA_HOST` — Ollama 地址（默认 `http://127.0.0.1:11434`）

---

## 🚫 禁止清单（两个 Agent 都适用）

| 禁止行为 | 说明 |
|----------|------|
| 改动 `Cargo.toml` 依赖版本不沟通 | 架构级决策 |
| 删除或注释掉测试 | 即使跑不过，也要修，不要删 |
| 大段复制粘贴代码 | 复用，不要复制 |
| `println!` 调试残留 | 用 `log` crate，不要裸 print |
| 临时 hack 不标记 `// TODO` | 如果必须 hack，留 TODO 并解释原因 |
| 直接改 `main` 分支 | 永远走 feature 分支 + PR |

---

## 📚 参考资料

- 项目架构设计文档：`docs/architecture-review.md`
- LoCoMo Benchmark 使用说明：`docs/architecture-review.md` 中的 Pitfalls 章节
- Ollama embedding 注意事项：`embedding.rs` 中的渐进截断策略
