# Fashion Smart Customer Service · 服装智能客服 Agent

基于 **LangGraph + FastAPI + SQLite + Chroma** 的服装商城智能客服系统,覆盖商品查询、订单查询、人工退款三大功能,支持多轮会话、双角色(用户/客服)操作的完整 Web 应用。

## 核心能力

| 模块 | 实现 |
|---|---|
| 意图路由 | 关键词 L1 + 大模型 L2 两级分类,路由至商品查询 / 订单查询 / 退款人工审核三个分支 |
| ReAct 循环 | `agent_node ⇄ tool_node` 回边循环,标准 Function Calling 调用工具,3 步熔断兜底防死循环 |
| 商品检索 | **filter-then-rank**:SQLite 先按价格/品类/颜色/尺码硬过滤圈定候选集,候选集内 BM25 + 向量混合检索,RRF 融合排序 |
| 数据存储 | SQLite 商品表为元数据唯一事实源;向量库仅存商品 ID 与关键词文本,无双写 |
| 文本入库 | 大模型从非结构化文本抽取商品字段(few-shot prompt),缺失必填字段拒绝入库 |
| 人工退款 | 退款不注册为模型工具,强制 `interrupt` 转人工;客服 resume 后仅创建售后工单,绝不承诺"退款成功" |
| 历史压缩 | user_profile 固定插槽(订单号/浏览商品,工具事件确定性填充)+ 强约束摘要 + 滑动窗口(最近 4 条原文) |
| 多轮会话 | LangGraph checkpointer(SQLite)按 thread_id 持久化状态,断点续聊 |
| 鉴权 | 自建注册登录:PBKDF2-HMAC-SHA256(10 万迭代)+ Bearer Token,RBAC 区分 customer / operator |
| 可观测 | 全链路 trace 事件落盘,可按 trace_id 回放每个节点的输入输出排查问题 |

## 架构

```
Streamlit UI ──► FastAPI ──► LangGraph
                (Bearer     ├─ intent_node ──► product_search ─► rewrite ─► compress_history ─► agent ⇄ tool(ReAct)
                 鉴权)      ├─ order_query ────────────────────► compress_history ─► agent ⇄ tool
                           └─ refund_request ─► human_review(interrupt 转人工)
                                                        │
                                            客服工作台 resume ─► 创建售后工单(落库)

存储:SQLite(商品/订单会话/用户/工单/checkpoint)+ Chroma(商品向量)+ 外部订单服务(HTTP)
```

## 快速开始

```bash
# 1. 安装依赖(uv)
uv sync

# 2. 配置环境变量(.env)
#    DASHSCOPE_API_KEY=sk-...   # 或 OPENAI_API_KEY,走 OpenAI 兼容端点

# 3. 灌入示例商品(双写 SQLite + 向量库)
uv run python scripts/import_products.py

# 4. 启动模拟订单服务(端口 9000)
uv run python scripts/mock_order_service.py

# 5. 启动后端 API(端口 8000)
uv run uvicorn app.main:app --port 8000

# 6. 启动前端 UI(端口 8501)
uv run streamlit run scripts/chat_ui.py
```

打开 http://localhost:8501 注册两个账号:第一个注册的用户(user-001)可查示例订单;在登录页勾选"注册为客服"创建的账号可进入客服工作台处理退款工单。

## 测试

```bash
uv run pytest          # 122 项测试:意图/检索/过滤/压缩/鉴权/工单/双角色端到端
uv run ruff check app tests scripts
```

## 项目结构

```
app/
├── main.py              # FastAPI:/chat、/chats、/auth、/reviews、/traces
├── auth.py              # 用户体系:PBKDF2 密码哈希、Bearer Token、RBAC
├── graph/               # LangGraph 状态定义、图构建、checkpointer
├── nodes/               # 意图/改写/Agent/工具/历史压缩/人工审核/兜底节点
├── tools/               # Function Calling 工具注册表 + 订单服务客户端
├── retrieval/           # SQLite 商品库、Chroma 向量库、BM25+RRF 混合检索、文本入库抽取
├── chat/                # 会话注册表(多轮会话归属与列表)
├── review/              # 人工审核:待审任务与售后工单存储
└── trace/               # 链路追踪:节点事件记录与查询
scripts/                 # 商品灌库、文本入库、模拟订单服务、Streamlit UI
tests/                   # 122 项 pytest 测试
```
