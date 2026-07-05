# RAG 知识库

Open Agent 的 RAG（Retrieval-Augmented Generation）让 Agent 能基于你自己的文档回答问题。本页涵盖文档加载、切块、向量化、混合检索、重排序与多知识库路由。

## 整体链路

```text
文档文件 (.txt/.md/.pdf/.docx/.csv/.json/.html/.rst)
   │
   ▼
document_loaders.load_file   ── 解析为纯文本
   │
   ▼
Indexer                       ── 按 chunk_size / chunk_overlap 切块
   │
   ▼
FAISSStore / ChromaStore     ── 向量化 + 存储（L2 归一化嵌入）
   │
   ▼
HybridRetriever              ── 查询时：向量检索 + BM25 关键词检索 → RRF 融合
   │
   ▼
reranker (bge-reranker-v2-m3) ── 交叉编码器重排序，取 top_k
   │
   ▼
KnowledgeBaseTool            ── 拼成文本段落回灌给 Agent
```

## 文档加载

`src/open_agent/rag/document_loaders.py` 的 `load_file(path)` 根据扩展名自动选择解析器：

| 扩展名 | 解析方式 |
|---|---|
| `.txt` / `.md` / `.rst` | 直接 UTF-8 读取。 |
| `.pdf` | `pymupdf`（fitz）逐页提取文本。 |
| `.docx` | `python-docx` 提取段落。 |
| `.csv` | 按行读取，每行转为 `key: value` 文本。 |
| `.json` | 递归展平为路径式键值对。 |
| `.html` | 正则剥离标签，保留可见文本。 |

`SUPPORTED_EXTENSIONS` 集合是上述八种。上传时扩展名不在集合内会返回 `400 unsupported_file_type`。

!!! note "依赖"
    PDF / DOCX 解析依赖 `pymupdf` 与 `python-docx`，由 `[rag]` / `[docs]` / `[all]` extra 提供。最小安装（`pip install -e .`）不包含，调用时会 `ImportError`。

## 切块（Indexer）

`Indexer` 按 `chunk_size` 与 `chunk_overlap` 切分文本，单位由 `OPEN_AGENT_SPLIT_UNIT` 决定：

- `char`（默认）— 按字符数切。中英文混排友好，但可能切断句子。
- `paragraph` — 按段落（空行分隔）切；段落过长时回退到字符切。

约束：`chunk_overlap` **必须小于** `chunk_size`，否则 `Settings` 校验失败。建议重叠设为 chunk_size 的 10%~20%，保证边界信息不丢失。

## 向量化与存储

### 嵌入模型

默认 `BAAI/bge-small-zh-v1.5`（中文优化、体积小、速度快）。可通过 `OPEN_AGENT_EMBEDDING_MODEL` 切换为 `BAAI/bge-large-zh-v1.5`、`BAAI/bge-m3` 等任何 sentence-transformers 模型。

`embedding_cache.py` 提供磁盘缓存，避免重复嵌入相同文本（按文本哈希命中）。

### 向量后端

| 后端 | 类 | 特点 |
|---|---|---|
| FAISS | `FAISSStore` | 默认。L2 归一化 + 内积（等价余弦相似度）。支持 `save(path)` / `load(path)` 持久化。 |
| ChromaDB | `ChromaStore` | 可选。支持元数据过滤、持久化。需安装 `[rag]` extra。 |

`KBManager` 在 `storage_dir` 设置时，会为每个 KB 在 `<storage_dir>/<name>.faiss` 持久化索引；启动时若文件存在则加载，否则新建。

## 混合检索（HybridRetriever）

`src/open_agent/rag/hybrid_retriever.py` 同时跑两路检索并融合：

1. **向量检索** — 查询嵌入与文档嵌入做内积，取 top `rerank_k`（默认 20）。
2. **BM25 关键词检索** — `rank_bm25` 在所有 chunk 上跑，取 top `rerank_k`。
3. **RRF 融合** — Reciprocal Rank Fusion：`score = Σ 1/(k + rank_i)`，`k=60`。两路排名靠前的 chunk 得分更高。

RRF 比简单加权更适合异构分数：向量分与 BM25 分不在同一量纲，加权需要调参；RRF 只用排名，鲁棒性更好。

## 重排序（Reranker）

`src/open_agent/rag/reranker.py` 用交叉编码器对 RRF 融合后的 `rerank_k` 个候选重新打分，取前 `rag_top_k`（默认 5）返回。

默认模型 `BAAI/bge-reranker-v2-m3`，多语言、对中英文都友好。设 `OPEN_AGENT_RERANKER_MODEL` 为空字符串可禁用重排序（此时直接取 RRF 的前 `rag_top_k`）。

!!! tip "rerank_k 与 rag_top_k 的关系"
    - `rerank_k`（默认 20）— 送入重排序的候选数，应 ≥ `rag_top_k`。
    - `rag_top_k`（默认 5）— 最终返回给 Agent 的 chunk 数。
    - 重排序会显著提升精度但增加延迟（每个候选都要过一次交叉编码器）。生产环境若延迟敏感，可调小 `rerank_k` 或禁用重排序。

## 多知识库路由

`src/open_agent/rag/kb_router.py` 的 `KnowledgeBaseRouter` 支持多个知识库共存，按查询语义路由：

- 每个 `KnowledgeBase` 有 `name` 与 `description`（自然语言描述其内容）。
- `route(question)` 用查询嵌入与各 KB 描述嵌入做相似度匹配，返回最相关的若干 KB。
- `retrieve(question, top_k, routed)` 只在被路由到的 KB 内检索，避免无关 KB 干扰。

`KBManager.query` 的返回结构：

```python
{
    "routed_kbs": ["company", "policies"],   # 路由命中的 KB 名
    "chunks": [                              # 融合 + 重排序后的 chunk
        {
            "document": "...",
            "score": 0.87,
            "metadata": {"source": "handbook.md", "kb_name": "company"},
        },
        ...
    ],
    "context_text": "..."                     # chunks 用空行拼接，适合直接喂给 LLM
}
```

## KBManager API

`KBManager` 是 RAG 的高层入口，CLI、Server 与库用法共用同一套接口：

| 方法 | 说明 |
|---|---|
| `create_kb(name, description)` | 显式创建一个知识库。 |
| `get_kb(name)` | 按名取 KB，未找到返回 `None`。 |
| `list_kbs()` | 列出所有 KB 名。 |
| `index_file(path, kb_name)` | 索引单文件到 KB；返回新增 chunk 数。KB 不存在时自动创建。 |
| `index_directory(dir, kb_name, description)` | 索引目录顶层所有支持的文件。 |
| `list_documents(kb_name)` | 列出 KB 内所有文档（按 source 分组）。 |
| `delete_document(kb_name, source)` | 删除某 source 的所有 chunk；返回删除数。 |
| `query(question, top_k)` | 完整 RAG 查询：路由 → 检索 → 重排序 → 拼上下文。 |

!!! note "并发安全"
    `_get_or_create_kb` 用 `_kb_lock` 串行化「获取或创建」临界区，避免两个并发调用同时为同一个新 KB 名创建实例（会导致前一个 KB 的索引与文档被孤立）。

## 索引与查询示例

### CLI 索引

```bash
# 单文件
open-agent index ./docs/handbook.md --kb company

# 目录（顶层）
open-agent index ./docs/policies --kb policies --desc "公司规章制度与流程"
```

### API 上传

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@./docs/handbook.md" \
  -F "kb_name=company"
```

### 库内调用

```python
import asyncio
from open_agent.rag.kb_manager import KBManager

async def main():
    manager = KBManager(
        storage_dir=".open_agent_kb_indexes",   # 持久化目录
        embedding_model="BAAI/bge-small-zh-v1.5",
        chunk_size=500,
        chunk_overlap=50,
        split_unit="char",
        top_k=5,
        reranker_model="BAAI/bge-reranker-v2-m3",
        rerank_k=20,
    )
    await manager.index_file("./docs/handbook.md", "company")
    await manager.index_directory("./docs/policies", "policies", "公司规章制度")
    print("KBs:", manager.list_kbs())           # ['company', 'policies']
    result = await manager.query("年假政策", top_k=5)
    print("routed:", result["routed_kbs"])
    for c in result["chunks"]:
        print(c["metadata"]["kb_name"], c["score"], c["document"][:60])

asyncio.run(main())
```

### 列出与删除文档

```bash
# 列出 KB 内文档
curl http://127.0.0.1:8000/api/knowledge-bases/company/documents

# 删除某 source
curl -X DELETE "http://127.0.0.1:8000/api/knowledge-bases/company/documents?source=./docs/handbook.md"
```

## RAG 评估

`src/open_agent/rag/evaluation.py` 提供 RAG 质量评估，四个维度：

| 指标 | 含义 |
|---|---|
| `faithfulness` | 答案是否忠于检索到的上下文（不编造）。 |
| `answer_relevance` | 答案与问题的相关性。 |
| `context_recall` | 检索到的上下文覆盖 ground truth 的比例。 |
| `context_precision` | 检索到的上下文中有多少是相关的。 |

CLI 评估命令：

```bash
open-agent evaluate ./examples/evaluation/rag_eval_sample.json
# 用 --demo 走启发式指标，无需 LLM
open-agent evaluate ./examples/evaluation/rag_eval_sample.json --demo
```

测试用例 JSON 格式：

```json
{
  "test_cases": [
    {
      "question": "年假有几天？",
      "expected_answer": "10 天",
      "retrieved_contexts": ["..."],
      "generated_answer": "10 天",
      "ground_truth_contexts": ["..."]
    }
  ]
}
```

## 调参建议

| 场景 | 推荐 |
|---|---|
| 中文文档为主 | `EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`，`RERANKER_MODEL=BAAI/bge-reranker-v2-m3`。 |
| 长文档（>10k 字） | `CHUNK_SIZE=800`，`CHUNK_OVERLAP=100`，`SPLIT_UNIT=paragraph`。 |
| 高精度 | `RERANK_K=50`，`RAG_TOP_K=5`。 |
| 低延迟 | 关闭重排序（`RERANKER_MODEL=`），`RAG_TOP_K=3`。 |
| 多知识库 | 每个 KB 写清晰的 `description`（路由靠它），如「公司规章制度与流程」「产品技术文档」。 |

## 下一步

- [工具系统](tools.md) — `KnowledgeBaseTool` 如何被 Agent 调用。
- [API 参考](api.md#rag) — `/api/upload`、`/api/knowledge-bases/*` 端点。
- [配置参考](configuration.md#RAG--向量化) — RAG 相关环境变量。
