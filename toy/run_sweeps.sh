set -u; cd /mnt/disk1/aiotlab/pqhung/Replica-Flow2PC
set -a; source .env; set +a
run() { echo "=== $* ==="; env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n uffm python train.py "$@" trainer.max_steps=20000 2>&1 | grep -aE "^FINAL|Error executing"; }
for R in 0.1 0.25 0.75 0.9; do run world=A model_variant=pair_path data.rho_e=$R logger.wandb.name=b0-rhoe$R; done
for C in 0.1 0.25 0.5; do run world=A model_variant=pair_path data.rho_ci=$C logger.wandb.name=b0-rhoci$C; done
echo "SWEEPS COMPLETE"
