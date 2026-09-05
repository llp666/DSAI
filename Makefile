PYTHON ?= python

data-small:   ; $(PYTHON) -m datagen.cli --config datagen/config/scale_small.yaml
data-full:    ; $(PYTHON) -m datagen.cli --config datagen/config/scale_full.yaml
validate:     ; $(PYTHON) -m datagen.validate --db warehouse/ecommerce.duckdb
test:         ; $(PYTHON) -m pytest datagen/tests -q
