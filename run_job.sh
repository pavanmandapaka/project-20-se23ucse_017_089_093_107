#!/bin/bash
#SBATCH --job-name=vlm_inference
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16000
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/vlm_inference_%j.out
#SBATCH --error=logs/vlm_inference_%j.err

# Environment Setup
export CUDA_VISIBLE_DEVICES="" # Stick to CPU to avoid driver conflicts
export TRANSFORMERS_OFFLINE=1  # Assumes you did the pre-download on the login node
export PYTHONPATH=$PYTHONPATH:$(python3 -m site --user-site)

python3 src/inference.py