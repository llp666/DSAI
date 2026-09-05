PYTHON ?= python

# ---- 数仓生成与校验（datagen） ----
data-small:   ; $(PYTHON) -m datagen.cli --config datagen/config/scale_small.yaml
data-full:    ; $(PYTHON) -m datagen.cli --config datagen/config/scale_full.yaml
validate:     ; $(PYTHON) -m datagen.validate --db warehouse/ecommerce.duckdb --config datagen/config/scale_full.yaml

# ---- 阶段一/二：语义层与编译器 ----
validate-schema: ; $(PYTHON) -m agent.semantic_layer
test:            ; $(PYTHON) -m pytest tests datagen/tests -q
eval:            ; $(PYTHON) -m eval.run_eval

# ---- 可观测 ----
langfuse-up:   ; docker compose -f docker-compose.langfuse.yml up -d
langfuse-down: ; docker compose -f docker-compose.langfuse.yml down
