# VLM Fine-Tuning & Inference Pipeline

A Vision-Language Model (VLM) fine-tuning and inference pipeline built with PyTorch and HuggingFace Transformers, designed to run on SLURM-managed GPU clusters. The project includes a full training workflow, resumable fine-tuning, and a web-based frontend for interacting with the model.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [Inference (Base Model)](#inference-base-model)
  - [Fine-Tuning](#fine-tuning)
  - [Inference (Fine-Tuned Model)](#inference-fine-tuned-model)
  - [Web Frontend](#web-frontend)
- [SLURM Job Scripts](#slurm-job-scripts)
- [Results](#results)

---

## Overview

This project fine-tunes a Vision-Language Model on a custom dataset and exposes the model through a FastAPI backend and a JavaScript/CSS frontend. Training is designed to run on GPU clusters via SLURM batch jobs, with support for checkpoint resumption across job boundaries.

Key capabilities:
- Fine-tune a pre-trained VLM on domain-specific image-text data
- Run inference with both the base and fine-tuned model
- Serve predictions through a REST API (FastAPI + Uvicorn)
- Interactive frontend for submitting images and viewing model responses

---

## Project Structure

```
.
├── data/                        # Dataset files (images + annotations)
├── frontend/                    # Web UI (Vite + JS/CSS)
│   └── .env                     # Frontend environment config (VITE_API_BASE)
├── results/                     # Inference outputs and training manifests
├── server/                      # FastAPI server code
├── src/                         # Core Python source code
│   ├── inference.py             # Base model inference
│   ├── inference_finetuned.py   # Fine-tuned model inference
│   └── train_full.py            # Full fine-tuning training loop
├── logs/                        # SLURM job logs (auto-created)
├── checkpoints/                 # Saved model checkpoints (auto-created)
├── run_job.sh                   # SLURM job: base model inference
├── run_train_job.sh             # SLURM job: training run
├── run_full_train_job.sh        # SLURM job: 24-hour full fine-tuning
├── run_inference_finetuned_job.sh  # SLURM job: fine-tuned inference
├── requirements.txt             # Python dependencies
└── Domain Research note.pdf     # Background research document
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | HuggingFace Transformers (VLM) |
| Deep Learning | PyTorch + CUDA 12.4 |
| Training | `accelerate`, `multiversity` |
| API Server | FastAPI + Uvicorn |
| Frontend | Vite + JavaScript / CSS |
| Cluster | SLURM (A100 GPU partition) |
| Data | Pandas, Pillow |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Node.js 18+ and npm (for the frontend)
- CUDA-compatible GPU (training) or CPU (inference)
- Access to a SLURM cluster (for HPC runs)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Install frontend dependencies:**
```bash
cd frontend
npm install
```

**requirements.txt includes:**
```
multiversity
pandas
torch
transformers
accelerate
pillow
fastapi
uvicorn
python-multipart
```

### Pre-download Model Weights (for offline/SLURM use)

Before submitting SLURM jobs, download the model weights on the login node:

```bash
# Run once on the login node — jobs use TRANSFORMERS_OFFLINE=1
python3 -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('<model-name>'); AutoTokenizer.from_pretrained('<model-name>')"
```

---

## Usage

### Inference (Base Model)

Run locally:
```bash
python3 src/inference.py
```

Or via SLURM:
```bash
sbatch run_job.sh
```

### Fine-Tuning

Standard training run:
```bash
sbatch run_train_job.sh
```

Full 24-hour fine-tuning run (with automatic resume support):
```bash
sbatch run_full_train_job.sh
```

The full training job saves epoch-level checkpoints under `checkpoints/full_train/`. If a job times out or fails, simply resubmit the same script and it will resume from the last saved checkpoint automatically.

### Inference (Fine-Tuned Model)

After training completes, run inference on the full test dataset:
```bash
sbatch run_inference_finetuned_job.sh
```

Results are saved to `results/fine_tuned_results.txt`.

### Web Frontend 

**1. Configure environment variables**

Create a `.env` file in the `frontend/` directory:
```env
VITE_API_BASE=http://127.0.0.1:8000/api/v1
```

**2. Start the API server** (from the project root):
```bash
python -m uvicorn server.app:app --reload
```

**3. Start the frontend dev server** (in a separate terminal):
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in your terminal). Make sure the backend server is running before using the UI.

---

## SLURM Job Scripts

| Script | Purpose | Time Limit | GPU |
|---|---|---|---|
| `run_job.sh` | Base model inference | 4 hours | A100 1g.5gb |
| `run_train_job.sh` | Training run | — | A100 1g.5gb |
| `run_full_train_job.sh` | Full fine-tuning (resumable) | 24 hours | A100 1g.5gb |
| `run_inference_finetuned_job.sh` | Fine-tuned model inference | 1 hour | A100 1g.5gb |

All jobs write logs to `logs/` and use the `gpu_student` partition.

---

## Results

Training outputs and inference results are stored in the `results/` directory:

- `results/full_training_manifest.json` — Training summary (epochs, loss, config)
- `results/fine_tuned_results.txt` — Per-sample inference predictions
- `logs/full_train_loss.csv` — Epoch-by-epoch loss curve

---

## License

This project is for academic purposes. See individual library licenses for dependency terms.