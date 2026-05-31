#!/bin/bash
# run_inference_finetuned_job.sh
# Submit with: sbatch run_inference_finetuned_job.sh
#
# May 11 — Anirudh: Run fine-tuned VLM inference on full test dataset

#SBATCH --job-name=vlm_ft_infer
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16000
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/vlm_ft_infer_%j.out
#SBATCH --error=logs/vlm_ft_infer_%j.err

mkdir -p logs results

# ── CUDA paths ────────────────────────────────────────────────────────────────
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ── Environment ───────────────────────────────────────────────────────────────
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PYTHONPATH:$(python3 -m site --user-site)

echo "============================================================"
echo "  Fine-Tuned Inference Job — May 11"
echo "  Date      : $(date)"
echo "  Node      : $SLURMD_NODENAME"
echo "  Job ID    : $SLURM_JOB_ID"
echo "============================================================"

echo ""
echo "--- GPU check ---"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not found"

echo ""
echo "--- Checkpoint check ---"
ls checkpoints/full_run/ 2>/dev/null || echo "No checkpoints found!"

echo ""
echo "Starting fine-tuned inference ..."
python3 src/inference_finetuned.py

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo "  JOB COMPLETED SUCCESSFULLY"
    echo "  Results → results/fine_tuned_results.txt"
    echo "  End time: $(date)"
    echo "============================================================"
else
    echo "============================================================"
    echo "  JOB FAILED — exit code $EXIT_CODE"
    echo "  Check: logs/vlm_ft_infer_${SLURM_JOB_ID}.err"
    echo "  End time: $(date)"
    echo "============================================================"
fi