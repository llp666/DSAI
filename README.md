# DSAI · 电商数据分析 Agent

**一句话问数：自然语言 → 语义层 → 确定性 SQL → 口径可解释的洞察与图表。**

电商运营每天都在问「上个月哪个商品卖得最好」「哪些 SKU 该补货」「这笔投放还要不要加」。
本项目把「业务提需求 → 分析师写 SQL → 回解读」的小时级链路，压缩成一次对话——
答案里带着口径、命中的表和可溯源的 SQL，而不是一个不知道从哪来的数字。

🔗 **产品介绍页**：[dsagent.netlify.app](https://dsagent.netlify.app/) —— 产品视角的介绍、架构流程图与实拍截图

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env            # 填入 LLM / EMBEDDING 密钥（搜索、Langfuse 可选）

python -m datagen.cli --config datagen/config/scale_small.yaml   # 生成模拟数仓
streamlit run app.py                                             # 启动
```

数仓数据由 `datagen/` 合成生成（规模、脏数据比例、随机种子均在 `datagen/config/` 配置），
规避真实数据的合规风险，同时保留多数据域、多表关联与真实数仓的分布特征。

## 目录结构

按五层架构组织，层间以契约接口交互，任一层可替换而不动其他层——LLM 可换（适配层）、
向量库可换（仓储接口）、数仓可换（连接器）。

| 层 | 目录 | 职责 |
|---|---|---|
| L1 数据与元数据 | `data/` | 语义层 YAML（指标口径、关系、维度值域）、表文档与场景卡语料 |
| L2 检索 | `retrieval/` | 问题改写、向量 + 关键词混合检索、Top-3 注入与 Token 预算裁剪 |
| L3 编排 | `orchestration/` | LangGraph 状态机、意图解析、双轨纠错路由、三层熔断 |
| L4 语义编译与执行 | `compile/` | 语义查询 DSL、sqlglot 编译器、干跑校验、只读执行 |
| L5 工具与呈现 | `tools/` | 库存/营销诊断工具、看板与图表；`app.py` 是 Streamlit 前端入口 |

`datagen/` 是模拟数仓生成器，只在准备数据时运行，不在提问链路上。

## 部署

仓库不携带数据产物：`warehouse/`（数仓 + 向量索引）与 `meta/`（表文档）都由 `datagen` 生成。
部署到 Streamlit Community Cloud 时，`bootstrap.py` 会在首次启动自动补齐——按
`datagen/config/scale_demo.yaml` 生成演示规模数仓，再构建检索索引（约 1~2 分钟，页面显示进度）；
密钥经 `bootstrap.bridge_secrets()` 从 `.streamlit/secrets.toml` 桥接为环境变量
（社区云的 `st.secrets` 不会进 `os.environ`）。产品介绍页部署在 Netlify，源码见 `site/`。

**技术栈**：LangGraph · ChromaDB · sqlglot · DuckDB · Pandas / Plotly · Streamlit · Langfuse
