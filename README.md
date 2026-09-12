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

| 层 | 目录 | 职责 |
|---|---|---|
| L1 数据与元数据 | `data/` | 语义层、检索语料 |
| L2 检索 | `retrieval/` | 混合检索、Token 预算 |
| L3 编排 | `orchestration/` | LangGraph 状态机、双轨纠错 |
| L4 语义编译与执行 | `compile/` | 语义查询 → SQL → 只读执行 |
| L5 工具与呈现 | `tools/` | 诊断工具、看板与图表 |

`app.py` 为 Streamlit 入口；`datagen/` 生成模拟数仓。

## 部署

数据产物（`warehouse/`、`meta/`）不入仓，均由 `datagen` 生成；部署到 Streamlit Community Cloud 时，
`bootstrap.py` 在首次启动自动补齐（约 1~2 分钟）。介绍页在 Netlify，源码见 `site/`。

**技术栈**：LangGraph · ChromaDB · sqlglot · DuckDB · Pandas / Plotly · Streamlit · Langfuse
