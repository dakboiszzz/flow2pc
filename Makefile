PY ?= python
FLOW_MATCHER ?= ot

lint:
	$(PY) -m ruff check src/ tests/ main.py train.py eval.py \
		train_endpoint_hclt.py train_mnist_classifier.py \
		summarize_benchmark.py summarize_dense_tuning.py

test:
	$(PY) -m pytest -q

train-flow:
	$(PY) train.py flow --flow-matcher $(FLOW_MATCHER)

distill:
	@test -n "$(FLOW_CKPT)" || (echo "FLOW_CKPT is required" && exit 1)
	$(PY) train.py distill --teacher-checkpoint "$(FLOW_CKPT)" \
		--pc-structure hclt --pc-layer dense \
		--capacity-profile standard \
		--endpoint-pretraining none --steps 20000 --batch-size 64

train-eval-classifier:
	$(PY) train_mnist_classifier.py

evaluate:
	@test -n "$(RUN_DIR)" || (echo "RUN_DIR is required" && exit 1)
	$(PY) eval.py --run-dir "$(RUN_DIR)"

.PHONY: lint test train-flow distill train-eval-classifier evaluate
