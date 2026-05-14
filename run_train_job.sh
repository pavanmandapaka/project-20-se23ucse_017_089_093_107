#!/bin/bash
# run_train_job.sh
# Submit with: sbatch run_train_job.sh

#SBATCH --job-name=vlm_train_smoke
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16000
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/vlm_train_%j.out
#SBATCH --error=logs/vlm_train_%j.err

mkdir -p logs checkpoints

# ── CUDA paths ────────────────────────────────────────────────────────────────
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ── Environment ───────────────────────────────────────────────────────────────
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:$(python3 -m site --user-site)
export TORCH_ALLOW_TF32=1

echo "============================================================"
echo "  VLM Smoke-Test Training Job"
echo "  Date      : $(date)"
echo "  Node      : $SLURMD_NODENAME"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Partition : $SLURM_JOB_PARTITION"
echo "============================================================"

# ── Full CUDA diagnostics ─────────────────────────────────────────────────────
echo ""
echo "--- SLURM GPU env vars ---"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS       : $SLURM_JOB_GPUS"
echo "SLURM_GPUS_ON_NODE   : $SLURM_GPUS_ON_NODE"

echo ""
echo "--- nvidia-smi ---"
which nvidia-smi && nvidia-smi -L || echo "nvidia-smi not found"

echo ""
echo "--- libcuda location ---"
find /usr -name "libcuda.so*" 2>/dev/null || echo "libcuda not found under /usr"

echo ""
echo "--- PyTorch CUDA check ---"
python3 -c "
import torch
print('torch version       :', torch.__version__)
print('torch.version.cuda  :', torch.version.cuda)
print('cuda.is_available() :', torch.cuda.is_available())
print('device_count        :', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name         :', torch.cuda.get_device_name(0))
    print('total memory        :', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"

echo ""
echo "--- LD_LIBRARY_PATH ---"
echo $LD_LIBRARY_PATH
echo ""

echo "Storage used in working directory:"
echo -e "\nCurrent utilisation of Storage(U.2 NVMe):  $(du -sh "$PWD" | awk '{print $1}')B\n"

# ── Run training ──────────────────────────────────────────────────────────────
echo "Starting smoke-test training ..."
python3 src/train.py

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job FAILED with exit code $EXIT_CODE. Check logs/vlm_train_${SLURM_JOB_ID}.err"
fi

echo "============================================================"
echo "  End time: $(date)"
echo "============================================================"