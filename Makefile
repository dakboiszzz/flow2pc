PY ?= python
FLOW_MATCHER ?= ot

lint:
	$(PY) -m ruff check src/ tests/ main.py train.py eval.py \
		train_endpoint_hclt.py train_mnist_classifier.py \
		synthetic_experiment.py summarize_benchmark.py summarize_dense_tuning.py \
		summarize_synthetic_tuning.py summarize_synthetic_confirmation.py \
		summarize_synthetic_timestep_significance.py \
		summarize_synthetic_density_weight_trend.py

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

synthetic-tune-density-time:
	FLOW2PC_PYTHON=$(PY) bash scripts/run_synthetic_density_timestep_tuning.sh

synthetic-refine-density-time:
	FLOW2PC_PYTHON=$(PY) bash scripts/run_synthetic_density_timestep_refinement.sh

synthetic-confirm-density-time:
	FLOW2PC_PYTHON=$(PY) bash scripts/run_synthetic_density_timestep_confirmation.sh

synthetic-test-timestep-significance:
	FLOW2PC_PYTHON=$(PY) bash scripts/run_synthetic_timestep_significance.sh

synthetic-tune-density-weight:
	FLOW2PC_PYTHON=$(PY) bash scripts/run_synthetic_density_weight_trend.sh

.PHONY: lint test train-flow distill train-eval-classifier evaluate \
	synthetic-tune-density-time synthetic-refine-density-time \
	synthetic-confirm-density-time synthetic-test-timestep-significance \
	synthetic-tune-density-weight
