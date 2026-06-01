"""
src/compile_results.py
Compile Quantitative Results — Ablation Table (Kolla | May 11)

Reads:
  - results/metrics_zero_shot.json          (Pranav's May 10 baseline scores)
  - results/fine_tuned_results.txt           (Anirudh's May 11 inference output)
  - data/vlm_conversational_dataset.json     (ground-truth captions)

Produces:
  - results/metrics_fine_tuned.json          (computed BLEU + ROUGE for fine-tuned)
  - results/quantitative_comparison.json     (both models side-by-side, TBDs filled)
  - results/ablation_table.txt               (plain-text table ready to paste into report)

Usage:
  python3 src/compile_results.py

No extra args needed — paths are resolved relative to the repo root.
"""

import os
import re
import json
import math
import sys
from collections import Counter
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
DATASET_PATH    = os.path.join(BASE_DIR, "data", "vlm_conversational_dataset.json")

ZERO_SHOT_METRICS  = os.path.join(RESULTS_DIR, "metrics_zero_shot.json")
FT_RESULTS_FILE    = os.path.join(RESULTS_DIR, "fine_tuned_results.txt")
FT_METRICS_OUT     = os.path.join(RESULTS_DIR, "metrics_fine_tuned.json")
COMPARISON_OUT     = os.path.join(RESULTS_DIR, "quantitative_comparison.json")
TABLE_OUT          = os.path.join(RESULTS_DIR, "ablation_table.txt")

# ── Optional library imports (same fallback strategy as evaluate.py) ──────────
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    USE_NLTK = True
except ImportError:
    USE_NLTK = False

try:
    from rouge_score import rouge_scorer as rouge_lib
    USE_ROUGE_LIB = True
except ImportError:
    USE_ROUGE_LIB = False


# ══════════════════════════════════════════════════════════════════════════════
#  TOKENISATION
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower().strip())

def _get_tokens(text: str) -> list:
    if USE_NLTK:
        try:
            return word_tokenize(text.lower())
        except Exception:
            pass
    return _tokenize(text)


# ══════════════════════════════════════════════════════════════════════════════
#  BLEU
# ══════════════════════════════════════════════════════════════════════════════

def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def _clipped_precision(ref_tokens, hyp_tokens, n):
    ref_ng = _ngrams(ref_tokens, n)
    hyp_ng = _ngrams(hyp_tokens, n)
    clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
    return clipped, max(sum(hyp_ng.values()), 1)

def compute_bleu(reference: str, hypothesis: str) -> dict:
    if USE_NLTK:
        ref_tok = _get_tokens(reference)
        hyp_tok = _get_tokens(hypothesis)
        smoothie = SmoothingFunction().method1
        scores = {}
        for n in range(1, 5):
            weights = tuple(1.0/n if i < n else 0.0 for i in range(4))
            scores[f"bleu_{n}"] = sentence_bleu(
                [ref_tok], hyp_tok, weights=weights, smoothing_function=smoothie
            )
        scores["bleu_avg"] = (
            scores["bleu_1"] * scores["bleu_2"] *
            scores["bleu_3"] * scores["bleu_4"]
        ) ** 0.25
        return {k: round(v, 6) for k, v in scores.items()}

    ref_tok = _tokenize(reference)
    hyp_tok = _tokenize(hypothesis)
    if not hyp_tok:
        return {f"bleu_{n}": 0.0 for n in range(1, 5)} | {"bleu_avg": 0.0}

    bp = 1.0 if len(hyp_tok) >= len(ref_tok) else math.exp(
        1 - len(ref_tok) / max(len(hyp_tok), 1)
    )
    log_p = []
    for n in range(1, 5):
        c, t = _clipped_precision(ref_tok, hyp_tok, n)
        log_p.append(math.log((c + 1) / (t + 1)))

    scores = {f"bleu_{n}": round(bp * math.exp(sum(log_p[:n]) / n), 6) for n in range(1, 5)}
    scores["bleu_avg"] = round(
        (scores["bleu_1"] * scores["bleu_2"] * scores["bleu_3"] * scores["bleu_4"]) ** 0.25, 6
    )
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  ROUGE
# ══════════════════════════════════════════════════════════════════════════════

def _lcs(x, y):
    m, n = len(x), len(y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if x[i-1]==y[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def _f1(p, r):
    return (2*p*r/(p+r)) if (p+r) > 0 else 0.0

def compute_rouge(reference: str, hypothesis: str) -> dict:
    if USE_ROUGE_LIB:
        scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        res = scorer.score(reference, hypothesis)
        return {
            "rouge_1_p": round(res["rouge1"].precision, 6),
            "rouge_1_r": round(res["rouge1"].recall,    6),
            "rouge_1_f": round(res["rouge1"].fmeasure,  6),
            "rouge_2_p": round(res["rouge2"].precision, 6),
            "rouge_2_r": round(res["rouge2"].recall,    6),
            "rouge_2_f": round(res["rouge2"].fmeasure,  6),
            "rouge_l_p": round(res["rougeL"].precision, 6),
            "rouge_l_r": round(res["rougeL"].recall,    6),
            "rouge_l_f": round(res["rougeL"].fmeasure,  6),
        }

    ref_tok = _tokenize(reference)
    hyp_tok = _tokenize(hypothesis)
    scores = {}
    for n, key in [(1, "rouge_1"), (2, "rouge_2")]:
        ref_ng = _ngrams(ref_tok, n)
        hyp_ng = _ngrams(hyp_tok, n)
        overlap = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        p = overlap / max(sum(hyp_ng.values()), 1)
        r = overlap / max(sum(ref_ng.values()), 1)
        scores[f"{key}_p"] = round(p, 6)
        scores[f"{key}_r"] = round(r, 6)
        scores[f"{key}_f"] = round(_f1(p, r), 6)
    lcs = _lcs(ref_tok, hyp_tok)
    p = lcs / max(len(hyp_tok), 1)
    r = lcs / max(len(ref_tok),  1)
    scores["rouge_l_p"] = round(p, 6)
    scores["rouge_l_r"] = round(r, 6)
    scores["rouge_l_f"] = round(_f1(p, r), 6)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD RESULTS + GROUND-TRUTH PAIRS
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(dataset_path: str) -> dict:
    """Returns {filename: ground_truth_caption} from the conversational dataset."""
    if not os.path.exists(dataset_path):
        print(f"  [WARN] Dataset not found at {dataset_path} — BLEU/ROUGE will be 0.")
        return {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return {os.path.basename(rec["image"]): rec["conversations"][1]["value"]
            for rec in dataset}

def load_results_txt(results_file: str) -> list:
    """
    Parse:   File: <filename> | Caption: <generated text>
    Returns: list of (filename, generated_caption)
    """
    pairs = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " | Caption: " not in line:
                continue
            parts = line.split(" | Caption: ", 1)
            if len(parts) != 2:
                continue
            fname = parts[0].replace("File: ", "").strip()
            cap   = parts[1].strip()
            pairs.append((fname, cap))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATE
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(all_bleu: list, all_rouge: list) -> dict:
    n = len(all_bleu)
    corpus = {}
    for k in all_bleu[0]:
        corpus[k] = round(sum(d[k] for d in all_bleu) / n, 6)
    for k in all_rouge[0]:
        corpus[k] = round(sum(d[k] for d in all_rouge) / n, 6)
    return corpus


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

METRIC_ROWS = [
    ("BLEU-1",    "bleu_1"),
    ("BLEU-2",    "bleu_2"),
    ("BLEU-3",    "bleu_3"),
    ("BLEU-4",    "bleu_4"),
    ("BLEU-avg",  "bleu_avg"),
    ("ROUGE-1 F", "rouge_1_f"),
    ("ROUGE-2 F", "rouge_2_f"),
    ("ROUGE-L F", "rouge_l_f"),
]

def build_comparison_json(zs: dict, ft: dict) -> dict:
    """Build the quantitative_comparison.json structure."""
    return {
        "zero_shot": zs,
        "fine_tuned": ft,
        "comparison_table": {
            "metrics":    [label for label, _ in METRIC_ROWS],
            "zero_shot":  [zs.get(key, None) for _, key in METRIC_ROWS],
            "fine_tuned": [ft.get(key, None) for _, key in METRIC_ROWS],
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def build_ablation_table(zs: dict, ft: dict) -> str:
    """
    Plain-text ablation table for the report.

    Example output:
    ┌──────────────┬────────────┬────────────┬───────────────┐
    │ Metric       │ Zero-Shot  │ Fine-Tuned │ Δ (abs)       │
    ├──────────────┼────────────┼────────────┼───────────────┤
    │ BLEU-1       │  0.0350    │  0.1234    │ +0.0884 ↑     │
    ...
    """
    col_label = 14
    col_val   = 12
    col_delta = 15

    def hline(l, m, r, lw, mw, rw):
        return l + "─"*(lw+2) + m + "─"*(mw+2) + m + "─"*(rw+2) + m + "─"*(col_delta+2) + r

    top    = hline("┌", "┬", "┐", col_label, col_val, col_val)
    mid    = hline("├", "┼", "┤", col_label, col_val, col_val)
    bot    = hline("└", "┴", "┘", col_label, col_val, col_val)

    def row(label, zs_val, ft_val):
        if zs_val is None or ft_val is None:
            zs_str = ft_str = delta_str = "  N/A"
        else:
            delta  = ft_val - zs_val
            arrow  = "↑" if delta >= 0 else "↓"
            sign   = "+" if delta >= 0 else ""
            zs_str    = f"{zs_val:.4f}"
            ft_str    = f"{ft_val:.4f}"
            delta_str = f"{sign}{delta:.4f} {arrow}"
        return (f"│ {label:<{col_label}} │ {zs_str:>{col_val}} │"
                f" {ft_str:>{col_val}} │ {delta_str:>{col_delta}} │")

    header = (f"│ {'Metric':<{col_label}} │ {'Zero-Shot':>{col_val}} │"
              f" {'Fine-Tuned':>{col_val}} │ {'Δ (abs)':>{col_delta}} │")

    lines = [
        "",
        "=" * (col_label + col_val*2 + col_delta + 13),
        "  ABLATION TABLE — Zero-Shot vs Fine-Tuned BLIP",
        f"  Zero-shot  n={zs.get('n_samples','?')}   evaluated {zs.get('evaluated_at','')}",
        f"  Fine-tuned n={ft.get('n_samples','?')}   evaluated {ft.get('evaluated_at','')}",
        "=" * (col_label + col_val*2 + col_delta + 13),
        top,
        header,
        mid,
    ]
    for label, key in METRIC_ROWS:
        lines.append(row(label, zs.get(key), ft.get(key)))
    lines += [bot, ""]

    # Append interpretation notes
    r1_delta  = ft.get("rouge_1_f", 0) - zs.get("rouge_1_f", 0)
    b4_delta  = ft.get("bleu_4",    0) - zs.get("bleu_4",    0)
    rel_r1    = (r1_delta / max(zs.get("rouge_1_f", 1e-9), 1e-9)) * 100
    rel_b4    = (b4_delta / max(zs.get("bleu_4",    1e-9), 1e-9)) * 100

    lines += [
        "  Notes:",
        f"  • ROUGE-1 F improved by {r1_delta:+.4f} ({rel_r1:+.1f}% relative)",
        f"  • BLEU-4   improved by {b4_delta:+.4f} ({rel_b4:+.1f}% relative)",
        "  • Fine-tuned model: BLIP-base, 10 epochs, lr=2e-5, seed=42",
        "  • Checkpoint: checkpoints/full_train/epoch_10/model.safetensors",
        "  • Dataset: MultiCaRe MRI subset, 1703 samples",
        "",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 62)
    print("  Compile Quantitative Results — May 11 (Kolla)")
    print("=" * 62)

    # ── Step 1: Load zero-shot scores (already computed by Pranav, May 10) ──
    print("\n[Step 1] Loading zero-shot metrics ...")
    if not os.path.exists(ZERO_SHOT_METRICS):
        sys.exit(f"  [ERROR] Not found: {ZERO_SHOT_METRICS}\n"
                 "  Run:  python3 src/evaluate.py --model zero_shot "
                 "--results_file results/preliminary_results.txt")
    with open(ZERO_SHOT_METRICS) as f:
        zs_scores = json.load(f)
    print(f"  Zero-shot n_samples : {zs_scores.get('n_samples')}")
    print(f"  ROUGE-1 F           : {zs_scores.get('rouge_1_f')}")
    print(f"  BLEU-4              : {zs_scores.get('bleu_4')}")

    # ── Step 2: Evaluate fine-tuned model ───────────────────────────────────
    print(f"\n[Step 2] Evaluating fine-tuned captions from {FT_RESULTS_FILE} ...")
    if not os.path.exists(FT_RESULTS_FILE):
        sys.exit(f"  [ERROR] Not found: {FT_RESULTS_FILE}\n"
                 "  Run inference_finetuned.py first (Anirudh, May 11).")

    print("  Loading ground-truth dataset ...")
    gt_lookup = load_ground_truth(DATASET_PATH)
    if not gt_lookup:
        print("  [WARN] No ground truth — scores will be 0. "
              "Ensure vlm_conversational_dataset.json exists.")

    print("  Parsing fine-tuned results file ...")
    raw_pairs = load_results_txt(FT_RESULTS_FILE)
    print(f"  Parsed {len(raw_pairs)} captions")

    # Match captions to ground truth
    matched_pairs = []
    skipped = 0
    for fname, generated in raw_pairs:
        gt = gt_lookup.get(fname, "")
        if gt:
            matched_pairs.append({"image_path": fname,
                                   "ground_truth": gt,
                                   "generated": generated})
        else:
            skipped += 1

    print(f"  Matched to ground truth : {len(matched_pairs)}")
    if skipped:
        print(f"  Skipped (no GT found)   : {skipped}")

    if not matched_pairs:
        sys.exit("  [ERROR] 0 matched pairs — cannot compute scores.\n"
                 "  Check that vlm_conversational_dataset.json is populated.")

    print(f"  Computing BLEU + ROUGE for {len(matched_pairs)} samples ...")
    print(f"  (Using {'NLTK' if USE_NLTK else 'manual'} BLEU, "
          f"{'rouge-score lib' if USE_ROUGE_LIB else 'manual'} ROUGE)")

    all_bleu, all_rouge = [], []
    log_every = max(1, len(matched_pairs) // 10)

    for i, pair in enumerate(matched_pairs, start=1):
        all_bleu.append(compute_bleu(pair["ground_truth"], pair["generated"]))
        all_rouge.append(compute_rouge(pair["ground_truth"], pair["generated"]))
        if i % log_every == 0 or i == len(matched_pairs):
            pct = i / len(matched_pairs) * 100
            print(f"  Progress: {i}/{len(matched_pairs)} ({pct:.0f}%)")

    ft_scores = aggregate(all_bleu, all_rouge)
    ft_scores["model_version"] = "fine_tuned"
    ft_scores["n_samples"]     = len(matched_pairs)
    ft_scores["evaluated_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ft_scores["checkpoint"]    = "checkpoints/full_train/epoch_10/model.safetensors"
    ft_scores["note"]          = "Preliminary fine-tuned scores — epoch 10"

    with open(FT_METRICS_OUT, "w") as f:
        json.dump(ft_scores, f, indent=2)
    print(f"\n  Fine-tuned metrics saved → {FT_METRICS_OUT}")
    print(f"  ROUGE-1 F : {ft_scores.get('rouge_1_f')}")
    print(f"  BLEU-4    : {ft_scores.get('bleu_4')}")

    # ── Step 3: Build comparison JSON ───────────────────────────────────────
    print("\n[Step 3] Building quantitative_comparison.json ...")
    comparison = build_comparison_json(zs_scores, ft_scores)
    with open(COMPARISON_OUT, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved → {COMPARISON_OUT}")

    # ── Step 4: Build ablation table ────────────────────────────────────────
    print("\n[Step 4] Building ablation_table.txt ...")
    table_str = build_ablation_table(zs_scores, ft_scores)
    with open(TABLE_OUT, "w") as f:
        f.write(table_str)
    print(f"  Saved → {TABLE_OUT}")

    # ── Print table to stdout ────────────────────────────────────────────────
    print(table_str)

    print("=" * 62)
    print("  DONE — all 3 output files written to results/")
    print("=" * 62)


if __name__ == "__main__":
    main()