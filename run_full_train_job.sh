#!/bin/bash
# run_full_train_job.sh
# Primary 24-hour fine-tuning run — May 9 (Anirudh)
# Submit with: sbatch run_full_train_job.sh

#SBATCH --job-name=vlm_full_train
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16000
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/vlm_full_train_%j.out
#SBATCH --error=logs/vlm_full_train_%j.err

mkdir -p logs checkpoints/full_train results

# ── CUDA paths ────────────────────────────────────────────────────────────────
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ── Environment ───────────────────────────────────────────────────────────────
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:$(python3 -m site --user-site)
export TORCH_ALLOW_TF32=1

# Disable tokenizer parallelism warning (single GPU, no benefit)
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "  VLM Full Fine-Tuning Job — 24-hour run"
echo "  Date      : $(date)"
echo "  Node      : $SLURMD_NODENAME"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Partition : $SLURM_JOB_PARTITION"
echo "============================================================"

# ── CUDA diagnostics ──────────────────────────────────────────────────────────
echo ""
echo "--- SLURM GPU env vars ---"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS       : $SLURM_JOB_GPUS"
echo "SLURM_GPUS_ON_NODE   : $SLURM_GPUS_ON_NODE"

echo ""
echo "--- nvidia-smi ---"
which nvidia-smi && nvidia-smi -L || echo "nvidia-smi not found"

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
    print('total memory        :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
"

echo ""
echo "--- Storage utilisation ---"
echo "Working directory: $PWD"
echo "Disk usage: $(du -sh "$PWD" | awk '{print $1}')B"

# ── Resume detection ──────────────────────────────────────────────────────────
echo ""
if [ -d "checkpoints/full_train" ]; then
    LATEST=$(ls -d checkpoints/full_train/epoch_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST" ]; then
        echo ">>> Resume detected: $LATEST — training will continue from next epoch."
    else
        echo ">>> No prior epoch checkpoints found — starting fresh."
    fi
else
    echo ">>> No checkpoint directory found — starting fresh."
fi

# ── Launch full training ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Launching full fine-tuning run ..."
echo "============================================================"
python3 src/train_full.py

EXIT_CODE=$?
echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Job completed successfully."
    echo "  Check results/full_training_manifest.json for summary."
    echo "  Loss curve: logs/full_train_loss.csv"
else
    echo "  Job FAILED with exit code $EXIT_CODE."
    echo "  Check logs/vlm_full_train_${SLURM_JOB_ID}.err for details."
    echo "  The run is resumable — resubmit this script and it will"
    echo "  continue from the last saved epoch checkpoint."
fi
echo "  End time: $(date)"
echo "============================================================"