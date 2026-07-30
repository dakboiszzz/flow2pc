PY ?= /mnt/disk1/aiotlab/envs/uffm/bin/python

test:            ## verification battery for all circuit families
	$(PY) -m pytest tests/ -q

lint:            ## ruff over the package and entry points
	ruff check repc/ train.py eval.py tests/

flow-mnist:      ## pretrain a flow teacher on MNIST
	$(PY) train.py task=flow data=mnist model=flow_unet

distill-mnist:   ## recompile: FLOW_CKPT=path/to/last.ckpt
	$(PY) train.py task=distill data=mnist model=pc_deep task.flow_ckpt=$(FLOW_CKPT)

distill-ddpm:    ## recompile a public DDPM teacher (no teacher training)
	$(PY) train.py task=distill data=cifar10 model=pc_shallow model.K=1024 \
	  task.flow_ckpt=hub 'task.flow_model._target_=repc.pretrained.DDPMTeacher' \
	  '~task.flow_model.base' +task.data_init=true

clean-logs:      ## sweep stray root logs into outputs/logs/
	-mv outputs_*.log outputs/logs/ 2>/dev/null

help:
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ —/'
.PHONY: test lint flow-mnist distill-mnist distill-ddpm clean-logs help
