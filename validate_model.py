#!/usr/bin/env python3
"""
validate_medgemma.py
=====================
Validates the fine-tuned MedGemma on a SEPARATE held-out parquet dataset
(100-150 samples created by create_reasoning_vqa_dataset.py but NOT used
during fine-tuning).

What it does
------------
  1. Loads a separate validation parquet (different file from training)
  2. Runs the fine-tuned model on each image + question
  3. Compares predictions against ground-truth answer and reasoning
  4. Reports metrics: ROUGE-1/L, BLEU-1/4, Exact Match, Answer F1
  5. Saves validation_results.json and validation_metrics.csv

Usage
-----
  # Step 1: Create a fresh validation dataset (100-150 samples)
  python create_reasoning_vqa_dataset.py \\
      --samples 150 \\
      --hf-token hf_... \\
      --output-dir ./validation_data

  # Step 2: Run validation
  python validate_medgemma.py \\
      --dataset  ./validation_data/reasoning_vqa_150samples_*.parquet \\
      --ft-model ./medgemma-vqa-finetuned \\
      --hf-token hf_... \\
      --gpu 1
"""

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
try:
    import torch
    import pandas as pd
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from tqdm import tqdm
except ImportError as e:
    sys.exit(
        f"[ERROR] Missing dependency: {e}\n"
        "  pip install torch transformers Pillow pandas tqdm"
    )

try:
    from rouge_score import rouge_scorer
except ImportError:
    sys.exit("[ERROR] pip install rouge-score")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
except ImportError:
    sys.exit("[ERROR] pip install nltk")


# ── system prompt (must match fine-tuning exactly) ────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert radiologist and medical imaging assistant. "
    "Given a medical image and a question, provide concise clinical reasoning "
    "followed by the answer.\n\n"
    "Always respond in this exact format:\n"
    "<reasoning>\n[2-3 sentence clinical reasoning based on visual findings]\n</reasoning>\n"
    "<answer>\n[concise final answer]\n</answer>"
)


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate fine-tuned MedGemma on a separate held-out parquet"
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="Path to the VALIDATION parquet file (separate from training data). "
             "Auto-detects the second-newest reasoning_vqa_*.parquet if omitted."
    )
    p.add_argument(
        "--ft-model", type=str, default="./medgemma-vqa-finetuned",
        help="Path to the fine-tuned model directory (default: ./medgemma-vqa-finetuned)"
    )
    p.add_argument("--hf-token",       type=str,   default=None)
    p.add_argument(
        "--num-samples", type=int, default=None,
        help="Max samples to validate. Default: all rows in the validation parquet."
    )
    p.add_argument("--gpu",            type=int,   default=None,
                   help="GPU index to pin to. Auto-selects freest GPU if omitted.")
    p.add_argument("--max-new-tokens", type=int,   default=200)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to write validation_results.json and validation_metrics.csv"
    )
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def resolve_token(cli: Optional[str]) -> Optional[str]:
    return cli or os.environ.get("HF_TOKEN", None)


def pin_gpu(requested: Optional[int]) -> str:
    """Pin to one GPU to avoid DataParallel OOM across multi-GPU nodes."""
    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    if n == 0:
        return "cpu"
    if requested is not None:
        idx = requested % n
    else:
        free = []
        for i in range(n):
            try:
                f, _ = torch.cuda.mem_get_info(i)
                free.append((f, i))
            except Exception:
                free.append((0, i))
        idx = max(free)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
    total_gb = torch.cuda.mem_get_info(0)[1] / 1024 ** 3
    print(f"  GPU {idx}: {torch.cuda.get_device_name(0)}  "
          f"({free_gb:.1f} / {total_gb:.1f} GiB free)")
    return "cuda"


def autodetect_validation_parquet() -> str:
    """
    Auto-detect the validation parquet.
    Picks the second-newest reasoning_vqa_*.parquet (the newest is assumed
    to be the training file). Falls back to any parquet if only one exists.
    """
    cwd = Path(".")
    candidates = sorted(
        cwd.glob("reasoning_vqa_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(candidates) >= 2:
        chosen = candidates[1]   # second newest = validation file
        print(f"  Auto-detected validation file: {chosen}")
        print(f"  (Skipping newest '{candidates[0].name}' — assumed to be training data)")
        return str(chosen)
    if len(candidates) == 1:
        print(f"  WARNING: Only one parquet found — using {candidates[0].name}")
        print("  Ideally create a SEPARATE validation dataset with:")
        print("    python create_reasoning_vqa_dataset.py --samples 150 \\")
        print("           --output-dir ./validation_data")
        return str(candidates[0])
    sys.exit(
        "[ERROR] No reasoning_vqa_*.parquet found.\n"
        "  Create a validation dataset first:\n"
        "    python create_reasoning_vqa_dataset.py --samples 150 "
        "--output-dir ./validation_data\n"
        "  Then pass: --dataset ./validation_data/reasoning_vqa_150samples_*.parquet"
    )


def bytes_to_pil(b: Any) -> Image.Image:
    if isinstance(b, Image.Image):
        return b.convert("RGB")
    if isinstance(b, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(b))).convert("RGB")
    if hasattr(b, "tobytes"):
        return Image.open(io.BytesIO(b.tobytes())).convert("RGB")
    raise ValueError(f"Cannot convert {type(b)} to PIL Image")


def parse_output(text: str) -> Tuple[str, str]:
    """Extract <reasoning> and <answer> blocks from model output."""
    r = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>",
                  text, re.DOTALL | re.IGNORECASE)
    a = re.search(r"<answer>\s*(.*?)\s*</answer>",
                  text, re.DOTALL | re.IGNORECASE)
    reasoning = r.group(1).strip() if r else text.strip()
    answer    = a.group(1).strip() if a else text.strip()
    return reasoning, answer


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(
    processor,
    model,
    image: Image.Image,
    question: str,
    device: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image", "image": image},
                     {"type": "text",  "text": question}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda" if device == "cuda" else "cpu")

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return processor.decode(
        out_ids[0][input_len:], skip_special_tokens=True
    ).strip()


# ── metrics ───────────────────────────────────────────────────────────────────
def rouge(pred: str, ref: str) -> Dict[str, float]:
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    s  = sc.score(ref, pred)
    return {
        "rouge1_f": round(s["rouge1"].fmeasure, 4),
        "rougeL_f": round(s["rougeL"].fmeasure, 4),
    }


def bleu(pred: str, ref: str) -> Dict[str, float]:
    smooth = SmoothingFunction().method1
    try:
        r = nltk.word_tokenize(ref.lower())
        p = nltk.word_tokenize(pred.lower())
    except Exception:
        r, p = ref.lower().split(), pred.lower().split()
    if not r or not p:
        return {"bleu1": 0.0, "bleu4": 0.0}
    return {
        "bleu1": round(sentence_bleu([r], p, (1, 0, 0, 0),           smooth), 4),
        "bleu4": round(sentence_bleu([r], p, (.25, .25, .25, .25),   smooth), 4),
    }


def exact_match(pred: str, ref: str) -> int:
    return int(pred.strip().lower() == ref.strip().lower())


def answer_f1(pred: str, ref: str) -> float:
    p_tok = set(pred.lower().split())
    r_tok = set(ref.lower().split())
    if not p_tok or not r_tok:
        return 0.0
    tp = len(p_tok & r_tok)
    if tp == 0:
        return 0.0
    pr = tp / len(p_tok)
    rc = tp / len(r_tok)
    return round(2 * pr * rc / (pr + rc), 4)


def all_metrics(
    pred_reasoning: str, pred_answer: str,
    gt_reasoning:   str, gt_answer:   str,
) -> Dict[str, float]:
    rr = rouge(pred_reasoning, gt_reasoning)
    ra = rouge(pred_answer,    gt_answer)
    br = bleu(pred_reasoning,  gt_reasoning)
    ba = bleu(pred_answer,     gt_answer)
    return {
        "reasoning_rouge1":   rr["rouge1_f"],
        "reasoning_rougeL":   rr["rougeL_f"],
        "reasoning_bleu1":    br["bleu1"],
        "reasoning_bleu4":    br["bleu4"],
        "answer_rouge1":      ra["rouge1_f"],
        "answer_rougeL":      ra["rougeL_f"],
        "answer_bleu1":       ba["bleu1"],
        "answer_bleu4":       ba["bleu4"],
        "answer_exact_match": exact_match(pred_answer, gt_answer),
        "answer_f1":          answer_f1(pred_answer, gt_answer),
    }


# ── display ───────────────────────────────────────────────────────────────────
def print_metrics_table(metrics_list: List[Dict], n_samples: int) -> None:
    keys = [
        ("reasoning_rouge1",   "Reasoning ROUGE-1"),
        ("reasoning_rougeL",   "Reasoning ROUGE-L"),
        ("reasoning_bleu1",    "Reasoning BLEU-1"),
        ("reasoning_bleu4",    "Reasoning BLEU-4"),
        ("answer_rouge1",      "Answer    ROUGE-1"),
        ("answer_rougeL",      "Answer    ROUGE-L"),
        ("answer_bleu1",       "Answer    BLEU-1"),
        ("answer_bleu4",       "Answer    BLEU-4"),
        ("answer_exact_match", "Answer    Exact Match"),
        ("answer_f1",          "Answer    F1"),
    ]
    print("\n" + "=" * 58)
    print(f"  VALIDATION RESULTS  ({n_samples} held-out samples)")
    print("=" * 58)
    print(f"  {'Metric':<30} {'Score':>8}   Bar")
    print("-" * 58)
    for key, label in keys:
        vals = [m[key] for m in metrics_list if key in m]
        avg  = sum(vals) / len(vals) if vals else 0.0
        bar  = "█" * int(avg * 25)
        print(f"  {label:<30} {avg:>8.4f}   {bar}")
    print("=" * 58)


def print_examples(results: List[Dict], n: int = 3) -> None:
    print(f"\n{'='*62}")
    print(f"  SAMPLE PREDICTIONS (first {n})")
    print("=" * 62)
    for i, r in enumerate(results[:n]):
        print(f"\n  ── Sample {i+1} ──────────────────────────────────────")
        print(f"  Question       : {r['question']}")
        print(f"  Ground Truth   : {r['gt_answer']}")
        print(f"  Predicted Ans  : {r['pred_answer']}")
        print(f"  Exact Match    : {'✅' if r['metrics']['answer_exact_match'] else '❌'}")
        print(f"  Answer F1      : {r['metrics']['answer_f1']:.4f}")
        print(f"  Answer ROUGE-L : {r['metrics']['answer_rougeL']:.4f}")
        print(f"\n  GT  reasoning  : {r['gt_reasoning'][:200]}...")
        print(f"  Pred reasoning : {r['pred_reasoning'][:200]}...")
        print(f"  Reasoning R-L  : {r['metrics']['reasoning_rougeL']:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args     = parse_args()
    hf_token = resolve_token(args.hf_token)
    np.random.seed(args.seed)

    # 0. Pin GPU
    print("\n[0/4] Selecting GPU...")
    device = pin_gpu(args.gpu)

    # 1. Load validation parquet
    print("\n[1/4] Loading validation dataset...")
    if args.dataset is None:
        args.dataset = autodetect_validation_parquet()
    elif not Path(args.dataset).exists():
        sys.exit(f"[ERROR] File not found: {args.dataset!r}\n"
                 "  Create a validation dataset first:\n"
                 "    python create_reasoning_vqa_dataset.py "
                 "--samples 150 --output-dir ./validation_data")

    df = pd.read_parquet(args.dataset)
    required = {"image", "question", "answer", "reasoning"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Parquet missing columns: {missing}")

    if args.num_samples is not None:
        df = df.sample(min(args.num_samples, len(df)),
                       random_state=args.seed).reset_index(drop=True)

    print(f"  Validation file : {args.dataset}")
    print(f"  Rows to validate: {len(df)}")

    # 2. Load fine-tuned model
    print("\n[2/4] Loading fine-tuned model...")
    if not Path(args.ft_model).exists():
        sys.exit(
            f"[ERROR] Fine-tuned model not found: {args.ft_model!r}\n"
            "  Run finetune_medgemma.py first."
        )
    load_kw: Dict[str, Any] = {"dtype": torch.bfloat16}
    if hf_token:
        load_kw["token"] = hf_token

    try:
        processor = AutoProcessor.from_pretrained(args.ft_model, **load_kw)
        model     = AutoModelForImageTextToText.from_pretrained(
            args.ft_model,
            device_map={"": 0} if device == "cuda" else None,
            **load_kw,
        )
        model.eval()
        print(f"  Loaded: {args.ft_model}")
    except Exception as e:
        sys.exit(f"[ERROR] Could not load fine-tuned model.\n  {e}")

    # 3. Run inference on validation set
    print(f"\n[3/4] Running inference on {len(df)} samples...")
    results:      List[Dict] = []
    metrics_list: List[Dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        image = bytes_to_pil(row["image"])
        raw   = run_inference(
            processor, model, image,
            row["question"], device, args.max_new_tokens
        )

        pred_reasoning, pred_answer = parse_output(raw)
        gt_reasoning = str(row["reasoning"])
        gt_answer    = str(row["answer"])

        m = all_metrics(pred_reasoning, pred_answer, gt_reasoning, gt_answer)
        metrics_list.append(m)

        results.append({
            "question"      : str(row["question"]),
            "gt_answer"     : gt_answer,
            "gt_reasoning"  : gt_reasoning,
            "pred_answer"   : pred_answer,
            "pred_reasoning": pred_reasoning,
            "raw_output"    : raw,
            "metrics"       : m,
        })

    # 4. Report and save
    print("\n[4/4] Saving results...")
    print_metrics_table(metrics_list, len(results))
    print_examples(results, n=3)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full per-sample JSON
    json_path = out_dir / "validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Aggregate metrics CSV  (mean / min / max / std per metric)
    metric_keys = list(metrics_list[0].keys())
    csv_rows = []
    for k in metric_keys:
        vals = [m[k] for m in metrics_list]
        csv_rows.append({
            "metric": k,
            "mean":   round(float(np.mean(vals)),  4),
            "median": round(float(np.median(vals)), 4),
            "min":    round(float(np.min(vals)),   4),
            "max":    round(float(np.max(vals)),   4),
            "std":    round(float(np.std(vals)),   4),
            "n":      len(vals),
        })
    csv_path = out_dir / "validation_metrics.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\n✅  Validation complete")
    print(f"   Samples validated : {len(results)}")
    print(f"   Full results      → {json_path}")
    print(f"   Metrics CSV       → {csv_path}")


if __name__ == "__main__":
    main()
