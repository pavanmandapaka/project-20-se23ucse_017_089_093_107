"""
src/evaluate.py
Evaluation Metrics — ROUGE & BLEU (Kolla | May 9)

Provides:
  - compute_bleu(reference, hypothesis)       → BLEU-1/2/3/4 scores
  - compute_rouge(reference, hypothesis)      → ROUGE-1/2/L scores
  - evaluate_model(model_version, results_file) → runs both metrics
      over every (ground_truth, generated) pair and logs to the DB.
  - print_summary(scores)                     → pretty comparison table

Usage:
  python3 src/evaluate.py --model zero_shot
  python3 src/evaluate.py --model fine_tuned
  python3 src/evaluate.py --model zero_shot --model fine_tuned  (ablation table)
"""

import os
import re
import json
import math
import argparse
import sqlite3
from collections import Counter
from datetime import datetime

# ─── Try to import optional libraries (fallback to manual impl if absent) ─────
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    USE_NLTK = True
except ImportError:
    USE_NLTK = False

try:
    from rouge_score import rouge_scorer as rouge_lib
    USE_ROUGE_LIB = True
except ImportError:
    USE_ROUGE_LIB = False

# Lazy-import DB helpers (same package)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import get_connection, log_metric, init_db

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH    = os.path.join(BASE_DIR, "data", "vlm_conversational_dataset.json")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")


# ══════════════════════════════════════════════════════════════════════════════
#  TOKENISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokeniser used as fallback when NLTK
    is not available.  Lowercases and strips non-alphanumeric tokens.
    """
    text = text.lower().strip()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def get_tokens(text: str) -> list[str]:
    if USE_NLTK:
        try:
            return word_tokenize(text.lower())
        except Exception:
            pass
    return tokenize(text)


# ══════════════════════════════════════════════════════════════════════════════
#  BLEU IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(reference_tokens: list[str],
                       hypothesis_tokens: list[str],
                       n: int) -> tuple[int, int]:
    """
    Modified n-gram precision (clipped to reference count).
    Returns (clipped_match_count, hypothesis_ngram_count).
    """
    ref_ngrams  = _ngrams(reference_tokens, n)
    hyp_ngrams  = _ngrams(hypothesis_tokens, n)

    clipped = sum(
        min(count, ref_ngrams[gram])
        for gram, count in hyp_ngrams.items()
    )
    return clipped, max(sum(hyp_ngrams.values()), 1)


def compute_bleu(reference: str, hypothesis: str) -> dict:
    """
    Compute sentence-level BLEU-1 through BLEU-4 with add-1 smoothing.

    Args:
        reference  : ground-truth caption string
        hypothesis : model-generated caption string

    Returns:
        {
          "bleu_1": float,   # unigram precision
          "bleu_2": float,   # bigram precision
          "bleu_3": float,   # trigram precision
          "bleu_4": float,   # 4-gram precision (standard MT metric)
          "bleu_avg": float  # geometric mean of BLEU-1..4
        }
    """
    if USE_NLTK:
        ref_tokens  = get_tokens(reference)
        hyp_tokens  = get_tokens(hypothesis)
        smoothie    = SmoothingFunction().method1
        scores = {}
        for n in range(1, 5):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
            scores[f"bleu_{n}"] = sentence_bleu(
                [ref_tokens], hyp_tokens,
                weights=weights,
                smoothing_function=smoothie
            )
        scores["bleu_avg"] = (
            scores["bleu_1"] * scores["bleu_2"] *
            scores["bleu_3"] * scores["bleu_4"]
        ) ** 0.25
        return {k: round(v, 6) for k, v in scores.items()}

    # ── Manual fallback ───────────────────────────────────────────────────────
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not hyp_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "bleu_avg": 0.0}

    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / max(len(hyp_tokens), 1)
    )

    log_precisions = []
    for n in range(1, 5):
        clipped, total = _clipped_precision(ref_tokens, hyp_tokens, n)
        # Add-1 smoothing
        log_precisions.append(math.log((clipped + 1) / (total + 1)))

    scores = {}
    for i, n in enumerate(range(1, 5)):
        scores[f"bleu_{n}"] = round(bp * math.exp(sum(log_precisions[:n]) / n), 6)

    scores["bleu_avg"] = round(
        (scores["bleu_1"] * scores["bleu_2"] *
         scores["bleu_3"] * scores["bleu_4"]) ** 0.25,
        6
    )
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  ROUGE IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _lcs_length(x: list, y: list) -> int:
    """Classic dynamic-programming LCS length."""
    m, n = len(x), len(y)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _f1(precision: float, recall: float, beta: float = 1.0) -> float:
    denom = (1 + beta ** 2) * precision + recall
    return ((1 + beta ** 2) * precision * recall / denom) if denom > 0 else 0.0


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        reference  : ground-truth caption string
        hypothesis : model-generated caption string

    Returns:
        {
          "rouge_1_p":  float, "rouge_1_r":  float, "rouge_1_f":  float,
          "rouge_2_p":  float, "rouge_2_r":  float, "rouge_2_f":  float,
          "rouge_l_p":  float, "rouge_l_r":  float, "rouge_l_f":  float,
        }
    """
    if USE_ROUGE_LIB:
        scorer  = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        result  = scorer.score(reference, hypothesis)
        return {
            "rouge_1_p": round(result["rouge1"].precision,  6),
            "rouge_1_r": round(result["rouge1"].recall,     6),
            "rouge_1_f": round(result["rouge1"].fmeasure,   6),
            "rouge_2_p": round(result["rouge2"].precision,  6),
            "rouge_2_r": round(result["rouge2"].recall,     6),
            "rouge_2_f": round(result["rouge2"].fmeasure,   6),
            "rouge_l_p": round(result["rougeL"].precision,  6),
            "rouge_l_r": round(result["rougeL"].recall,     6),
            "rouge_l_f": round(result["rougeL"].fmeasure,   6),
        }

    # ── Manual fallback ───────────────────────────────────────────────────────
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    scores = {}

    # ROUGE-1 and ROUGE-2
    for n, key in [(1, "rouge_1"), (2, "rouge_2")]:
        ref_ng = _ngrams(ref_tokens, n)
        hyp_ng = _ngrams(hyp_tokens, n)

        overlap   = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        ref_total = max(sum(ref_ng.values()), 1)
        hyp_total = max(sum(hyp_ng.values()), 1)

        p = overlap / hyp_total
        r = overlap / ref_total
        f = _f1(p, r)

        scores[f"{key}_p"] = round(p, 6)
        scores[f"{key}_r"] = round(r, 6)
        scores[f"{key}_f"] = round(f, 6)

    # ROUGE-L (LCS-based)
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    p   = lcs / max(len(hyp_tokens), 1)
    r   = lcs / max(len(ref_tokens),  1)
    f   = _f1(p, r)
    scores["rouge_l_p"] = round(p, 6)
    scores["rouge_l_r"] = round(r, 6)
    scores["rouge_l_f"] = round(f, 6)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  CORPUS-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_scores(all_bleu: list[dict], all_rouge: list[dict]) -> dict:
    """
    Average per-sentence scores across the full test corpus.
    Returns one dict with corpus-level BLEU + ROUGE.
    """
    if not all_bleu:
        return {}

    keys_bleu  = list(all_bleu[0].keys())
    keys_rouge = list(all_rouge[0].keys())
    n          = len(all_bleu)

    corpus = {}
    for k in keys_bleu:
        corpus[k] = round(sum(d[k] for d in all_bleu) / n, 6)
    for k in keys_rouge:
        corpus[k] = round(sum(d[k] for d in all_rouge) / n, 6)

    return corpus


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD GENERATED CAPTIONS FROM RESULTS FILE OR DB
# ══════════════════════════════════════════════════════════════════════════════

def load_results_txt(results_file: str) -> list[dict]:
    """
    Parse the preliminary_results.txt format:
      File: <filename> | Caption: <generated text>
    and match to ground-truth via dataset JSON.
    Returns list of {"image_path", "ground_truth", "generated"} dicts.
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Build ground-truth lookup  filename → caption
    gt_lookup = {}
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for rec in dataset:
            fname = os.path.basename(rec["image"])
            gt_lookup[fname] = rec["conversations"][1]["value"]

    pairs = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts     = line.split(" | Caption: ", 1)
            if len(parts) != 2:
                continue
            fname     = parts[0].replace("File: ", "").strip()
            generated = parts[1].strip()
            gt        = gt_lookup.get(fname, "")
            if gt:                       # only evaluate when ground truth exists
                pairs.append({
                    "image_path"  : fname,
                    "ground_truth": gt,
                    "generated"   : generated,
                })
    return pairs


def load_results_from_db(model_version: str) -> list[dict]:
    """Pull inference rows for a given model_version from the SQLite DB."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT image_path, ground_truth, generated_text "
        "FROM inferences WHERE model_version = ? AND ground_truth IS NOT NULL",
        (model_version,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"image_path": r["image_path"],
         "ground_truth": r["ground_truth"],
         "generated": r["generated_text"]}
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_version: str,
                   results_file: str | None = None,
                   log_to_db: bool = True,
                   verbose: bool = False) -> dict:
    """
    Run BLEU + ROUGE over every (ground_truth, generated) pair for a model.

    Args:
        model_version : label e.g. "zero_shot" or "fine_tuned"
        results_file  : path to a results .txt file (optional — falls back to DB)
        log_to_db     : write per-sample metrics to the SQLite metrics table
        verbose       : print per-sample scores

    Returns:
        Corpus-level aggregated score dict.
    """
    init_db()

    # ── Load pairs ─────────────────────────────────────────────────────────
    if results_file and os.path.exists(results_file):
        pairs = load_results_txt(results_file)
        print(f"  Loaded {len(pairs)} pairs from {results_file}")
    else:
        pairs = load_results_from_db(model_version)
        print(f"  Loaded {len(pairs)} pairs from DB (model_version='{model_version}')")

    if not pairs:
        print(f"  [WARN] No valid pairs found for '{model_version}'. "
              "Ensure ground_truth exists in the results file or DB.")
        return {}

    all_bleu, all_rouge = [], []

    for i, pair in enumerate(pairs, start=1):
        ref  = pair["ground_truth"]
        hyp  = pair["generated"]

        bleu  = compute_bleu(ref, hyp)
        rouge = compute_rouge(ref, hyp)

        all_bleu.append(bleu)
        all_rouge.append(rouge)

        if verbose:
            print(f"\n  [{i}/{len(pairs)}] {pair['image_path']}")
            print(f"    REF : {ref[:80]}...")
            print(f"    HYP : {hyp[:80]}...")
            print(f"    BLEU-4: {bleu['bleu_4']:.4f}  ROUGE-L: {rouge['rouge_l_f']:.4f}")

        # ── Log per-sample metrics to DB ───────────────────────────────────
        if log_to_db:
            from database import log_inference
            inference_id = log_inference(
                model_version = model_version,
                image_path    = pair["image_path"],
                generated_text= hyp,
                ground_truth  = ref,
            )
            for metric_name, val in {**bleu, **rouge}.items():
                log_metric(inference_id, metric_name, val)

    corpus = aggregate_scores(all_bleu, all_rouge)
    corpus["model_version"] = model_version
    corpus["n_samples"]     = len(pairs)
    corpus["evaluated_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Save corpus scores to JSON ─────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"metrics_{model_version}.json")
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"\n  Corpus scores saved → {out_path}")

    return corpus


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINT / ABLATION TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(scores_list: list[dict]):
    """
    Print a side-by-side comparison table for one or more model score dicts.
    This is the ablation table used in the report.
    """
    if not scores_list:
        return

    metrics = [
        ("BLEU-1",   "bleu_1"),
        ("BLEU-2",   "bleu_2"),
        ("BLEU-3",   "bleu_3"),
        ("BLEU-4",   "bleu_4"),
        ("BLEU-avg", "bleu_avg"),
        ("ROUGE-1 F","rouge_1_f"),
        ("ROUGE-2 F","rouge_2_f"),
        ("ROUGE-L F","rouge_l_f"),
    ]

    models = [s.get("model_version", f"model_{i}") for i, s in enumerate(scores_list)]
    col_w  = max(12, max(len(m) for m in models) + 2)
    lbl_w  = 14

    header = f"{'Metric':<{lbl_w}}" + "".join(f"{m:>{col_w}}" for m in models)
    print("\n" + "=" * len(header))
    print("  EVALUATION RESULTS — ABLATION TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, key in metrics:
        row = f"{label:<{lbl_w}}"
        for s in scores_list:
            val = s.get(key, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("-" * len(header))
    for s in scores_list:
        n = s.get("n_samples", "?")
        v = s.get("model_version", "?")
        print(f"  {v}: {n} samples evaluated")
    print("=" * len(header) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VLM captions with BLEU + ROUGE")
    p.add_argument(
        "--model", action="append", dest="models", default=[],
        metavar="MODEL_VERSION",
        help="Model version label(s) to evaluate. Use multiple times for ablation. "
             "E.g. --model zero_shot --model fine_tuned"
    )
    p.add_argument(
        "--results_file", type=str, default=None,
        help="Path to a results .txt file (overrides DB lookup). "
             "E.g. results/preliminary_results.txt"
    )
    p.add_argument(
        "--no_db", action="store_true",
        help="Skip writing per-sample scores to the database."
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample scores during evaluation."
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.models:
        # Default: evaluate zero-shot baseline from the results file
        args.models = ["zero_shot"]

    all_scores = []
    for model_version in args.models:
        print(f"\n{'='*55}")
        print(f"  Evaluating: {model_version}")
        print(f"{'='*55}")

        results_file = args.results_file
        # If only one model and a results_file given, use it; otherwise use DB
        if len(args.models) > 1 and args.results_file:
            # For multi-model ablation, results_file applies only to first model
            if model_version != args.models[0]:
                results_file = None

        scores = evaluate_model(
            model_version = model_version,
            results_file  = results_file,
            log_to_db     = not args.no_db,
            verbose       = args.verbose,
        )
        if scores:
            all_scores.append(scores)

    print_summary(all_scores)


if __name__ == "__main__":
    main()