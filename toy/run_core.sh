set -u; cd /mnt/disk1/aiotlab/pqhung/Replica-Flow2PC
set -a; source .env; set +a
run() { echo "=== $* ==="; env CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n uffm python train.py "$@" trainer.max_steps=20000 2>&1 | grep -aE "^FINAL|Error executing"; }
for W in A B; do
  for V in endpoint marginal_path marginal_path_vel pair_endpoint pair_path; do
    run world=$W model_variant=$V logger.wandb.name=b0-$W-$V
  done
done
echo "CORE COMPLETE"
