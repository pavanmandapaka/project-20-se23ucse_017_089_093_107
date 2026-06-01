# Experiments & Results

## 4.1 Experimental Setup

### 4.1.1 Model

We use **BLIP** (Bootstrapping Language-Image Pre-training) in its `blip-image-captioning-base` variant (Salesforce), a vision-language model pre-trained on large-scale web image-text pairs. BLIP employs a Vision Transformer (ViT) encoder and an autoregressive text decoder, making it well-suited for image captioning tasks. The base model checkpoint was obtained from HuggingFace (`Salesforce/blip-image-captioning-base`) and stored locally for offline training.

### 4.1.2 Dataset

All experiments use the **MultiCaRe MRI subset**, a curated collection of 1,703 MRI image–caption pairs drawn from the MultiCaRe open-access medical imaging dataset. Each record follows the LLaVA-style conversational format:

```json
{
  "id": "...",
  "image": "data/whole_multicare_dataset/vlm_mri_subset/images/...",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this medical image..."},
    {"from": "gpt",   "value": "<ground-truth clinical caption>"}
  ]
}
```

The dataset was prepared using `prepare_vlm_dataset.py`, which filters the broader MultiCaRe corpus for MRI modality and validates that all referenced image files exist and are readable. A 90/10 train–validation split was applied deterministically (seed = 42), yielding approximately **1,533 training** and **170 validation** samples.

### 4.1.3 Hardware & Infrastructure

Training was conducted on the university HPC cluster using a single **NVIDIA A100 GPU** (MIG partition: 1g.5gb — a 5 GB memory slice). Jobs were submitted via SLURM (`sbatch run_full_train_job.sh`) with CUDA 12.4. The constrained GPU memory necessitated aggressive memory optimisation:

- **Mixed precision**: `torch.amp.autocast` with `bfloat16`
- **Gradient checkpointing**: enabled to trade compute for memory
- **Micro-batching**: batch size of 1, with gradient accumulation over 8 steps (effective batch size = 8)

### 4.1.4 Training Hyperparameters

| Hyperparameter              | Value                    |
|-----------------------------|--------------------------|
| Base model                  | BLIP-base (ViT + decoder)|
| Number of epochs            | 10                       |
| Batch size                  | 1                        |
| Gradient accumulation steps | 8                        |
| Effective batch size        | 8                        |
| Learning rate               | 2 × 10⁻⁵                |
| LR schedule                 | Cosine with linear warmup|
| Warmup ratio                | 5% of total steps        |
| Max gradient norm           | 1.0                      |
| Weight decay                | 0.01                     |
| Max text length             | 128 tokens               |
| Precision                   | bfloat16 (mixed)         |
| Random seed                 | 42                       |
| Validation split            | 10%                      |

---

## 4.2 Training Dynamics

The full fine-tuning run completed all 10 epochs successfully in approximately **53 minutes** (≈ 318 seconds per epoch). Training loss converged rapidly over the first 5 epochs and then plateaued, as shown in the per-epoch loss summary:

| Epoch | Avg. Train Loss | Duration (s) | Global Step |
|:-----:|:---------------:|:------------:|:-----------:|
|   1   |     8.3820      |    318.8     |     213     |
|   2   |     4.6886      |    318.0     |     426     |
|   3   |     3.3003      |    318.3     |     639     |
|   4   |     2.4415      |    317.9     |     852     |
|   5   |     2.0751      |    318.5     |    1065     |
|   6   |     2.0334      |    318.0     |    1278     |
|   7   |     2.0182      |    318.3     |    1491     |
|   8   |     2.0135      |    317.9     |    1704     |
|   9   |     2.0124      |    318.1     |    1917     |
|  10   |     2.0123      |    317.8     |    2130     |

**Key observations:**

- **Rapid early convergence**: Loss drops dramatically from 8.38 (epoch 1) to 2.08 (epoch 5), a **75% reduction** in the first half of training.
- **Loss plateau**: After epoch 5, losses stabilise in the narrow range 2.01–2.03. The marginal gain from epoch 5 → 10 is only 0.063 (3%), suggesting diminishing returns. This near-flat plateau raises the possibility that additional epochs would not yield meaningful improvement, and that the model may be approaching the information-theoretic floor for this dataset size and architecture.
- **No training instability**: No collapse events, gradient explosions, or NaN losses were detected. The collapse-detection monitor (with conservative thresholds for loss explosion and gradient norm) never triggered, indicating numerically stable training throughout.

---

## 4.3 Quantitative Results

We evaluate both the **zero-shot** (pre-trained, unmodified BLIP) and **fine-tuned** (epoch 10 checkpoint) models on the full set of 1,703 image–caption pairs using standard machine translation and summarisation metrics: **BLEU** (1 through 4) and **ROUGE** (1, 2, L). All metrics are computed at the sentence level using NLTK and `rouge-score`, then averaged across the corpus.

### 4.3.1 Ablation Table — Zero-Shot vs. Fine-Tuned

| Metric      | Zero-Shot | Fine-Tuned | Δ (Absolute) | Δ (Relative) |
|-------------|:---------:|:----------:|:------------:|:------------:|
| BLEU-1      |   0.0350  |   0.1826   |   +0.1476    |   +421.7%    |
| BLEU-2      |   0.0104  |   0.0884   |   +0.0780    |   +750.0%    |
| BLEU-3      |   0.0061  |   0.0440   |   +0.0380    |   +623.0%    |
| BLEU-4      |   0.0048  |   0.0228   |   +0.0180    |   +375.0%    |
| BLEU-avg    |   0.0100  |   0.0611   |   +0.0512    |   +512.0%    |
| ROUGE-1 F   |   0.0878  |   0.2726   |   +0.1849    |   +210.6%    |
| ROUGE-2 F   |   0.0073  |   0.0793   |   +0.0719    |   +984.9%    |
| ROUGE-L F   |   0.0716  |   0.2013   |   +0.1297    |   +181.1%    |

> **Summary**: Fine-tuning yields consistent and substantial improvements across all metrics. The largest absolute gain is in **ROUGE-1 F** (+0.185), while the largest relative gain is in **ROUGE-2 F** (+985%), reflecting that the zero-shot model produces almost no correct bigram overlap with the medical reference captions.

### 4.3.2 Analysis of Metric Improvements

- **BLEU scores**: All BLEU n-gram precisions improve by 4–8× after fine-tuning. However, BLEU-4 remains low at 0.023, indicating the model still struggles to reproduce 4-gram sequences from the reference captions. This is expected for medical text where captions are highly specialised and verbatim n-gram reproduction is inherently difficult.
- **ROUGE-1 F (0.273)**: The fine-tuned model achieves unigram overlap that captures approximately 27% of the reference vocabulary, suggesting it has learned domain-relevant medical terms (e.g., "T1-weighted", "lesion", "mass", "parietal").
- **ROUGE-L F (0.201)**: The longest common subsequence-based metric shows moderate structural similarity between generated and reference captions, indicating partial alignment in sentence-level word ordering.

---

## 4.4 Qualitative Analysis

### 4.4.1 Zero-Shot Baseline Outputs

The pre-trained BLIP model, having never seen medical images, produces outputs that are largely **non-informative and pathologically repetitive**. Representative examples:

| Image File | Zero-Shot Caption |
|------------|-------------------|
| PMC3162801...g003_c | `mri mri mri mri mri mri mri mri mri mri mri mri mri mri mri mri mri mri mri mri` |
| PMC3162802...g003_d | `the brain is shown in this image` |
| PMC3144596...g002 | `a black and white image of a person with a radio camera` |
| PMC2815846...g002_B | `the image shows a small, white dog with a black nose` |
| PMC2684223...g001 | `the image shows a large, circular, circular, circular, circular, circular, circular...` |

**Failure mode analysis (Zero-Shot):**
- **Token repetition collapse**: The dominant failure is degenerate repetition (e.g., "mri mri mri..." repeated 20 times). This occurs in an estimated >50% of all zero-shot outputs, where the decoder locks onto a single token and repeats it until max length.
- **Generic/irrelevant descriptions**: When not repeating, the model produces captions like "a black and white image of a dog's head" or "the cat is lying on the ground" — descriptions appropriate for natural images but entirely incorrect for medical scans.
- **No clinical vocabulary**: Zero-shot outputs contain no medical terminology such as lesion locations, imaging sequences (T1, T2, FLAIR), or anatomical structures.

### 4.4.2 Fine-Tuned Outputs

After fine-tuning, the model generates captions that contain medical terminology and attempt to describe clinical findings. Representative examples:

| Image File | Fine-Tuned Caption (truncated) |
|------------|-------------------------------|
| PMC10018421...f1_a | `sagittal t1-weighted magnetic resonance imaging of the right parietal ventricle, showing a mass in the left temporal lobe and midline lesional part of the cerebellum...` |
| PMC10070255...g001 | `preoperative magnetic resonance imaging of the patient's brain showing a large mass in the right temporal lobe, and an extra-axial t1-weighted image shows that the tumor has been removed...` |
| PMC10073555...g0001_B | `preoperative t1-weighted magnetic resonance imaging showing a mass in the right frontal parietal lesion of the cerebellum and the left temporal lobe, with an intracranial ventricular sinus...` |

**Improvements observed:**
- Captions now use appropriate **medical terminology**: "T1-weighted", "magnetic resonance imaging", "preoperative", "parietal", "ventricle", "cerebellum", "lesion".
- The model has learned the **structural pattern** of radiology captions (modality → plane → anatomical location → findings).
- Outputs are syntactically well-formed sentences rather than repeated tokens.

---

## 4.5 Honest Reporting of Limitations & Failures

Despite the quantitative improvements, several significant limitations must be acknowledged:

### 4.5.1 Hallucinated and Inaccurate Medical Content

> [!CAUTION]
> The fine-tuned model frequently generates plausible-sounding but **factually incorrect** clinical descriptions. This is a critical safety concern for any downstream medical application.

The model produces non-existent medical terms (e.g., "cranistic", "edemastrens", "edemastrub", "ventriclectomial", "craniteous", "peritum") that appear to be garbled combinations of real terminology. These are **hallucinations** — the model has learned the statistical distribution of medical language but not its semantics. Examples:

- `"...with an intracranial cranistic sinus attenous edemastrens on"` — nonsensical
- `"...ventriclectomial portion of the cerebellum"` — no such medical term
- `"...lesional capillal edemastrub"` — fabricated terminology

### 4.5.2 Template-Like Repetition

While the degenerate token-level repetition of the zero-shot model is eliminated, the fine-tuned model exhibits a different failure mode: **template-level repetition**. A large proportion of outputs follow a near-identical template:

> *"preoperative t1-weighted magnetic resonance imaging of the right parietal [ventricle/lesion], showing a [mass/large mass] in the left [temporal lobe/ventricle]..."*

This suggests the model has overfit to the dominant caption patterns in the training set rather than learning to discriminate between individual images. The captions are likely insensitive to the actual visual content of different MRI scans.

### 4.5.3 Low Absolute BLEU-4 Score

The BLEU-4 score of **0.023** is well below the typical threshold for useful machine translation output (BLEU-4 ≥ 0.30). While BLEU-4 is known to be a harsh metric for caption generation (especially with single references), the low score confirms that the model rarely produces exact 4-gram matches with the reference captions. This limits the clinical utility of the generated text.

### 4.5.4 Potential Overfitting

The training loss plateau from epoch 5 onward (2.08 → 2.01) with only a 3% marginal improvement, combined with the template-like outputs, suggests possible **overfitting to the training distribution**. Contributing factors include:

- **Small dataset size**: 1,703 samples is small for fine-tuning a vision-language model
- **No data augmentation**: No image-level or text-level augmentations were applied
- **Full model fine-tuning**: All parameters were unfrozen; parameter-efficient fine-tuning (LoRA/QLoRA) was not explored
- **Single dataset**: No cross-domain evaluation was performed (e.g., testing on non-MRI or non-MultiCaRe images)

### 4.5.5 Evaluation Limitations

- **Single-reference evaluation**: Each image has exactly one reference caption. BLEU and ROUGE scores are inherently depressed with single references since there are many valid ways to describe a medical image.
- **No clinical evaluation**: We did not conduct a human evaluation with radiologists, which would be essential to assess the clinical accuracy and utility of the generated captions.
- **Train/test overlap**: Metrics were computed on the full dataset (n=1,703) rather than on a held-out test set, which may inflate reported scores.

---

## 4.6 Summary of Findings

| Aspect | Result |
|--------|--------|
| Fine-tuning impact | Consistent improvement across all metrics (BLEU, ROUGE) |
| Best metric gain | ROUGE-2 F: +985% relative improvement |
| Medical vocabulary | Successfully acquired through fine-tuning |
| Caption structure | Learned radiology-style sentence patterns |
| Key failure | Hallucinated/fabricated medical terms |
| Second failure | Template-like outputs lacking image specificity |
| BLEU-4 (absolute) | 0.023 — still far below clinical utility threshold |
| Overfitting risk | Moderate — loss plateau + repetitive outputs |
| Clinical readiness | **Not ready** — requires human evaluation and hallucination mitigation |

Fine-tuning BLIP on the MultiCaRe MRI subset demonstrates that domain adaptation improves medical image captioning substantially over zero-shot baselines. However, the model's tendency to hallucinate medical terminology and produce template-like descriptions highlights the **significant gap between metric improvement and clinical utility**. Future work should explore parameter-efficient fine-tuning, larger and more diverse datasets, retrieval-augmented generation, and rigorous clinical evaluation.
