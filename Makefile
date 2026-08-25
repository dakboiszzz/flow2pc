PY ?= /mnt/disk1/aiotlab/envs/uffm/bin/python

lint:
	$(PY) -m ruff check repc/ train.py eval.py

train-flow:
	$(PY) train.py flow

distill:
	@test -n "$(FLOW_CKPT)" || (echo "FLOW_CKPT is required" && exit 1)
	$(PY) train.py distill --teacher-checkpoint "$(FLOW_CKPT)" --student region

evaluate:
	@test -n "$(RUN_DIR)" || (echo "RUN_DIR is required" && exit 1)
	$(PY) eval.py --run-dir "$(RUN_DIR)"

.PHONY: lint train-flow distill evaluate
