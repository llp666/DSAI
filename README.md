# DSAI · 电商数据分析 Agent

一个基于五层架构的电商数据分析 Agent：自然语言提问 → 检索定位相关表 → 生成结构化语义查询 → 确定性编译为 SQL 只读执行 → 组装诊断洞察与图表。

- 研发方案：`docs/电商数据分析Agent研发方案.docx`
- 五层架构说明：`docs/电商数据分析Agent五层架构说明.docx`
- 数仓生成规格：`docs/DATAGEN_SPEC.md`

## 五层架构与目录

| 层 | 目录 | 职责 |
|---|---|---|
| L1 数据与元数据 | `data/` | 语义层 YAML 配置加载、表文档与场景卡语料（`data/semantic_layer/` 是语义层配置源） |
| L2 检索 | `retrieval/` | 问题改写、混合检索、嵌入、Token 预算裁剪 |
| L3 编排 | `orchestration/` | LangGraph 状态机、意图解析、双轨纠错、三层熔断 |
| L4 语义编译与执行 | `compile/` | 语义查询 DSL、SQL 编译器、只读执行 |
| L5 工具与呈现 | `tools/` | 库存/营销诊断工具、答案模板；`app.py` 是 Streamlit 前端入口 |

```
├── data/
├── retrieval/
├── orchestration/
├── compile/
├── tools/
├── app.py
├── config.py
├── tracing.py
├── datagen/
├── eval/
├── tests/
├── warehouse/ meta/
└── docs/
```

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env            # 填入 LLM / EMBEDDING 密钥
# 实时性问题（天气/新闻/汇率…）由 LLM 自主决定调用 search_tool 联网回答，可选：SEARCH_API_KEY=keen_xxx（见 docs/model.md）

streamlit run app.py            # 启动前端
python -m eval.run_demo         # 三场景 Demo（--tool-demo 工具四场景 / --empty-eval 空结果命中率）
python -m eval.run_eval         # 33 问端到端评测
python -m pytest tests datagen/tests -q   # 测试
```

数仓数据由 `datagen/` 生成（`python -m datagen.cli --config datagen/config/scale_full.yaml`），详见 `docs/DATAGEN_SPEC.md`。