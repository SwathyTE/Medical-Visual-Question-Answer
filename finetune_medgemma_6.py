#!/usr/bin/env python3
"""
finetune_medgemma.py
=====================
Fine-tunes google/medgemma-1.5-4b-it on the reasoning-augmented VQA parquet
dataset produced by create_reasoning_vqa_dataset.py using LoRA (PEFT).

Training I/O
------------
  INPUT  : image + question
  OUTPUT : <reasoning>...</reasoning><answer>...</answer>

Prerequisites
-------------
  pip install torch transformers peft datasets Pillow pandas pyarrow accelerate
  python patch_peft.py          # fixes torchao/bitsandbytes in PEFT installation
  export HF_TOKEN=hf_...

Usage
-----
  python finetune_medgemma.py                          # auto-detects parquet
  python finetune_medgemma.py --dataset file.parquet   # explicit path
  python finetune_medgemma.py --gpu 1                  # pin to specific GPU
"""

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ── stdlib dtype fix: patch json encoder BEFORE any imports that use torch ───
import json as _json

class _SafeEncoder(_json.JSONEncoder):
    def default(self, obj):
        # torch.dtype, numpy dtype, etc.
        if hasattr(obj, '__module__') and 'torch' in str(getattr(obj, '__module__', '')):
            return str(obj)
        if type(obj).__name__ == 'dtype':
            return str(obj)
        return super().default(obj)

_orig_dumps = _json.dumps
def _safe_dumps(obj, **kwargs):
    kwargs.setdefault('cls', _SafeEncoder)
    return _orig_dumps(obj, **kwargs)
_json.dumps = _safe_dumps

# ── third-party ───────────────────────────────────────────────────────────────
try:
    import torch
    import pandas as pd
    import numpy as np
    from PIL import Image
    from datasets import Dataset
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — works on headless servers
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError as e:
    sys.exit(
        f"[ERROR] Missing dependency: {e}\n"
        "  pip install torch transformers peft datasets Pillow pandas pyarrow accelerate\n"
        "  Then run: python patch_peft.py"
    )

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "google/medgemma-1.5-4b-it"
DEFAULT_OUTPUT = "./medgemma-vqa-finetuned"

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
        description="Fine-tune MedGemma on reasoning-augmented VQA dataset"
    )
    p.add_argument("--dataset",    type=str,   default=None,
                   help="Path to .parquet file. Auto-detects reasoning_vqa_*.parquet if omitted.")
    p.add_argument("--model",      type=str,   default=DEFAULT_MODEL)
    p.add_argument("--output",     type=str,   default=DEFAULT_OUTPUT)
    p.add_argument("--hf-token",   type=str,   default=None)
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=2)
    p.add_argument("--grad-accum", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--max-len",    type=int,   default=512)
    p.add_argument("--lora-r",     type=int,   default=16)
    p.add_argument("--lora-alpha", type=int,   default=32)
    p.add_argument("--val-split",  type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--gpu",        type=int,   default=None,
                   help="GPU index to pin to (e.g. --gpu 1). "
                        "Prevents multi-GPU DataParallel OOM. "
                        "Auto-selects GPU with most free VRAM if omitted.")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo",    type=str,  default=None)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    return cli_token or os.environ.get("HF_TOKEN", None)


def pin_gpu(requested: Optional[int]) -> str:
    """
    Set CUDA_VISIBLE_DEVICES to a single GPU to prevent DataParallel
    from spreading across multiple partially-occupied devices.
    """
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
    free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
    total_gb = torch.cuda.mem_get_info(0)[1] / 1024**3
    print(f"  Pinned to GPU {idx}: {torch.cuda.get_device_name(0)}  "
          f"({free_gb:.1f} / {total_gb:.1f} GiB free)")
    return "cuda"


def autodetect_parquet() -> str:
    cwd = Path(".")
    for pattern in ("reasoning_vqa_*.parquet", "*.parquet"):
        candidates = sorted(cwd.glob(pattern),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return str(candidates[0])
    sys.exit(
        "[ERROR] No parquet file found.\n"
        "  Run create_reasoning_vqa_dataset.py first, or pass --dataset PATH"
    )


def bytes_to_pil(b: Any) -> Image.Image:
    if isinstance(b, Image.Image):
        return b.convert("RGB")
    if isinstance(b, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(b))).convert("RGB")
    if hasattr(b, "tobytes"):
        return Image.open(io.BytesIO(b.tobytes())).convert("RGB")
    raise ValueError(f"Cannot convert {type(b)} to PIL Image")


def build_target(reasoning: str, answer: str) -> str:
    return (
        f"<reasoning>\n{reasoning.strip()}\n</reasoning>\n"
        f"<answer>\n{answer.strip()}\n</answer>"
    )


# ── dataset ───────────────────────────────────────────────────────────────────
def load_splits(path: str, val_split: float, seed: int):
    print(f"  File : {path}")
    df = pd.read_parquet(path)
    missing = {"image", "question", "answer", "reasoning"} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Parquet missing columns: {missing}\n"
                 f"  Found: {list(df.columns)}")
    print(f"  Rows : {len(df)}")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val    = max(1, int(len(df) * val_split))
    df_val   = df.iloc[:n_val].reset_index(drop=True)
    df_train = df.iloc[n_val:].reset_index(drop=True)
    print(f"  Train: {len(df_train)}  |  Val: {len(df_val)}")
    return Dataset.from_pandas(df_train), Dataset.from_pandas(df_val)


# ── collator ──────────────────────────────────────────────────────────────────
class VQACollator:
    """
    INPUT  → system prompt + image + question   (masked, no loss)
    OUTPUT → <reasoning>...</reasoning><answer>...</answer>  (loss computed here)
    """
    def __init__(self, processor, max_len: int):
        self.processor = processor
        self.max_len   = max_len

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_ids, all_attn, all_labels, all_pixels = [], [], [], []

        for ex in examples:
            image  = bytes_to_pil(ex["image"])
            target = build_target(ex["reasoning"], ex["answer"])

            messages = [
                {"role": "system",
                 "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [{"type": "image", "image": image},
                              {"type": "text",  "text": ex["question"]}]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": target}]},
            ]

            full = self.processor.apply_chat_template(
                messages, add_generation_prompt=False,
                tokenize=True, return_dict=True, return_tensors="pt"
            )
            prompt_only = self.processor.apply_chat_template(
                messages[:-1], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt"
            )

            ids  = full["input_ids"][0][:self.max_len]
            attn = full["attention_mask"][0][:self.max_len]
            lbl  = ids.clone()
            lbl[:prompt_only["input_ids"].shape[-1]] = -100  # mask prompt

            all_ids.append(ids)
            all_attn.append(attn)
            all_labels.append(lbl)
            if "pixel_values" in full:
                all_pixels.append(full["pixel_values"][0])

        pad_id  = self.processor.tokenizer.pad_token_id or 0
        max_seq = max(t.shape[0] for t in all_ids)

        def lpad(t, val):
            n = max_seq - t.shape[0]
            return torch.cat([torch.full((n,), val, dtype=t.dtype), t])

        batch = {
            "input_ids":      torch.stack([lpad(t, pad_id) for t in all_ids]),
            "attention_mask": torch.stack([lpad(t, 0)      for t in all_attn]),
            "labels":         torch.stack([lpad(t, -100)   for t in all_labels]),
        }
        if all_pixels:
            batch["pixel_values"] = torch.stack(all_pixels)
        return batch


# ── LoRA ──────────────────────────────────────────────────────────────────────
def apply_lora(model, r: int, alpha: int):
    attention_keys = {"q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"}
    found = {name.split(".")[-1]
             for name, mod in model.named_modules()
             if isinstance(mod, torch.nn.Linear)
             and name.split(".")[-1] in attention_keys}
    targets = sorted(found) if found else ["q_proj", "v_proj"]
    print(f"  LoRA targets: {targets}")

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
        use_dora=False,
        loftq_config=None,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ── save processor safely ─────────────────────────────────────────────────────
def save_processor_safe(processor, save_dir: str) -> None:
    """
    Save processor, stripping any non-JSON-serializable values (e.g. torch.dtype)
    from the tokenizer config before writing.
    """
    try:
        processor.save_pretrained(save_dir)
    except TypeError:
        # Fallback: sanitize tokenizer_config.json manually
        tok = processor.tokenizer
        cfg = tok.init_kwargs.copy() if hasattr(tok, "init_kwargs") else {}
        safe_cfg: Dict[str, Any] = {}
        for k, v in cfg.items():
            try:
                _json.dumps(v, cls=_SafeEncoder)
                safe_cfg[k] = v
            except (TypeError, ValueError):
                safe_cfg[k] = str(v)
        cfg_path = Path(save_dir) / "tokenizer_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            _json.dump(safe_cfg, f, indent=2, cls=_SafeEncoder)
        # Save everything else (processor_config, special_tokens_map, etc.)
        try:
            tok.save_pretrained(save_dir)
        except TypeError:
            pass
        print(f"  Processor saved (with dtype sanitization) → {save_dir}")


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_training_graphs(log_history: list, output_dir: str, args) -> None:
    """
    Generate and save all training graphs from the Trainer log history.

    Graphs produced:
      1. Training Loss curve
      2. Validation (Eval) Loss curve
      3. Train vs Eval Loss overlay
      4. Learning Rate schedule
      5. Gradient Norm curve
      6. Perplexity curves (train + eval)
      7. Loss improvement per epoch (bar chart)
      8. Training summary dashboard (all-in-one)
    """
    import math
    from typing import List as _List  # local alias to avoid shadowing

    out = Path(output_dir) / "training_graphs"
    out.mkdir(parents=True, exist_ok=True)

    # ── extract data from log history ────────────────────────────────────────
    train_steps, train_losses, train_lrs, train_gnorms = [], [], [], []
    eval_epochs, eval_losses = [], []
    # Accuracy: derived as 1 - normalised_loss proxy, or read directly if logged
    train_accuracies: list = []
    eval_accuracies:  list = []

    for entry in log_history:
        # Training step entries
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_losses.append(entry["loss"])
            if "learning_rate" in entry:
                train_lrs.append(entry["learning_rate"])
            if "grad_norm" in entry:
                train_gnorms.append(entry["grad_norm"])
            # Use explicit accuracy if logged; otherwise approximate via e^-loss
            if "accuracy" in entry:
                train_accuracies.append(entry["accuracy"])
        # Eval entries (once per epoch)
        if "eval_loss" in entry:
            eval_epochs.append(entry.get("epoch", len(eval_epochs) + 1))
            eval_losses.append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                eval_accuracies.append(entry["eval_accuracy"])

    # Fallback: approximate token-level accuracy as exp(-loss) when not logged.
    # This is a proxy: a perfect model has loss→0, accuracy→1; random has high loss, accuracy→0.
    if not train_accuracies and train_losses:
        train_accuracies = [math.exp(-min(l, 20)) for l in train_losses]
    if not eval_accuracies and eval_losses:
        eval_accuracies = [math.exp(-min(l, 20)) for l in eval_losses]

    if not train_losses:
        print("  ⚠  No training logs found — skipping graphs.")
        return

    # Derived: perplexity = e^loss
    train_ppl  = [math.exp(min(l, 20)) for l in train_losses]
    eval_ppl   = [math.exp(min(l, 20)) for l in eval_losses]

    # Epoch boundaries (approximate from step count)
    steps_per_epoch = max(1, len(train_steps) // max(1, len(eval_epochs)))

    # ── style ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor" : "#0f1117",
        "axes.facecolor"   : "#1a1d27",
        "axes.edgecolor"   : "#2e3250",
        "axes.labelcolor"  : "#c8ccd8",
        "axes.titlecolor"  : "#e8eaf0",
        "axes.grid"        : True,
        "grid.color"       : "#2e3250",
        "grid.linestyle"   : "--",
        "grid.alpha"       : 0.6,
        "xtick.color"      : "#8890a8",
        "ytick.color"      : "#8890a8",
        "text.color"       : "#c8ccd8",
        "lines.linewidth"  : 2.2,
        "font.family"      : "monospace",
        "legend.facecolor" : "#1a1d27",
        "legend.edgecolor" : "#2e3250",
        "legend.fontsize"  : 9,
    })

    TRAIN_COLOR = "#4fc3f7"   # light blue
    EVAL_COLOR  = "#f06292"   # pink
    LR_COLOR    = "#81c784"   # green
    GNORM_COLOR = "#ffb74d"   # orange
    PPL_COLOR   = "#ce93d8"   # purple
    ACC_COLOR   = "#80cbc4"   # teal-green (accuracy)
    BAR_GOOD    = "#4db6ac"   # teal (improvement)
    BAR_BAD     = "#ef5350"   # red  (regression)

    def save(fig, name):
        path = out / name
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 1. Training Loss ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_losses, color=TRAIN_COLOR, label="Train Loss", alpha=0.9)
    ax.fill_between(train_steps, train_losses, alpha=0.15, color=TRAIN_COLOR)
    # Mark epoch boundaries
    for i, ep_step in enumerate(range(steps_per_epoch, len(train_steps), steps_per_epoch)):
        ax.axvline(ep_step, color="#ffffff", alpha=0.15, linestyle=":")
        ax.text(ep_step, max(train_losses) * 0.95, f"E{i+1}", color="#ffffff",
                fontsize=7, ha="center", alpha=0.5)
    ax.set_title("Training Loss", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    save(fig, "01_train_loss.png")

    # ── 2. Eval Loss ──────────────────────────────────────────────────────────
    if eval_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eval_epochs, eval_losses, color=EVAL_COLOR, marker="o",
                markersize=8, label="Eval Loss")
        ax.fill_between(eval_epochs, eval_losses, alpha=0.15, color=EVAL_COLOR)
        best_idx = eval_losses.index(min(eval_losses))
        ax.scatter([eval_epochs[best_idx]], [eval_losses[best_idx]],
                   color="#ffd54f", s=120, zorder=5, label=f"Best: {eval_losses[best_idx]:.4f}")
        ax.set_title("Validation Loss per Epoch", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(eval_epochs)
        ax.legend()
        fig.tight_layout()
        save(fig, "02_eval_loss.png")

    # ── 3. Train vs Eval Loss Overlay ─────────────────────────────────────────
    if eval_losses:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(train_steps, train_losses, color=TRAIN_COLOR, label="Train Loss", alpha=0.8)
        # Map eval epochs to approximate steps for overlay
        eval_steps = [int(e * steps_per_epoch) for e in eval_epochs]
        ax.plot(eval_steps, eval_losses, color=EVAL_COLOR, marker="o",
                markersize=9, linewidth=2.5, label="Eval Loss")
        ax.set_title("Train vs Eval Loss", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        save(fig, "03_train_vs_eval_loss.png")

    # ── 4. Learning Rate Schedule ─────────────────────────────────────────────
    if train_lrs:
        fig, ax = plt.subplots(figsize=(10, 4))
        lr_steps = train_steps[:len(train_lrs)]
        ax.plot(lr_steps, train_lrs, color=LR_COLOR, label="Learning Rate")
        ax.fill_between(lr_steps, train_lrs, alpha=0.15, color=LR_COLOR)
        ax.set_title("Learning Rate Schedule (Cosine Decay)", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend()
        fig.tight_layout()
        save(fig, "04_learning_rate.png")

    # ── 5. Gradient Norm ──────────────────────────────────────────────────────
    if train_gnorms:
        fig, ax = plt.subplots(figsize=(10, 4))
        gnorm_steps = train_steps[:len(train_gnorms)]
        ax.plot(gnorm_steps, train_gnorms, color=GNORM_COLOR,
                alpha=0.85, label="Grad Norm")
        # Rolling average
        window = max(1, len(train_gnorms) // 10)
        rolling = pd.Series(train_gnorms).rolling(window, min_periods=1).mean().tolist()
        ax.plot(gnorm_steps, rolling, color="#fff176", linewidth=2,
                linestyle="--", label=f"Rolling avg (w={window})")
        ax.set_title("Gradient Norm During Training", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Grad Norm")
        ax.legend()
        fig.tight_layout()
        save(fig, "05_grad_norm.png")

    # ── 6. Perplexity ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_ppl, color=PPL_COLOR, label="Train Perplexity", alpha=0.85)
    if eval_ppl:
        eval_steps = [int(e * steps_per_epoch) for e in eval_epochs]
        ax.plot(eval_steps, eval_ppl, color=EVAL_COLOR, marker="o",
                markersize=8, linewidth=2.5, label="Eval Perplexity")
    ax.set_title("Perplexity (e^loss) — Lower is Better", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.legend()
    fig.tight_layout()
    save(fig, "06_perplexity.png")

    # ── 7. Loss Improvement per Epoch (bar chart) ─────────────────────────────
    if len(eval_losses) > 1:
        improvements = [eval_losses[i-1] - eval_losses[i]
                        for i in range(1, len(eval_losses))]
        epoch_labels = [f"E{i}→E{i+1}" for i in range(1, len(eval_losses))]
        colors = [BAR_GOOD if v > 0 else BAR_BAD for v in improvements]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(epoch_labels, improvements, color=colors, width=0.5,
                      edgecolor="#0f1117", linewidth=1.2)
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.001 if val >= 0 else -0.005),
                    f"{val:+.4f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=10, color="#ffffff", fontweight="bold")
        ax.axhline(0, color="#ffffff", alpha=0.3, linewidth=1)
        ax.set_title("Eval Loss Improvement per Epoch (positive = improvement)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Loss Reduction")
        fig.tight_layout()
        save(fig, "07_loss_improvement.png")

    # ── 8a. Train Accuracy (vs Step) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_accuracies, color=ACC_COLOR,
            label="Train Accuracy (approx)", alpha=0.9)
    ax.fill_between(train_steps, train_accuracies, alpha=0.13, color=ACC_COLOR)
    # Mark epoch boundaries
    for i, ep_step in enumerate(range(steps_per_epoch, len(train_steps), steps_per_epoch)):
        ax.axvline(ep_step, color="#ffffff", alpha=0.18, linestyle=":")
        ax.text(ep_step, max(train_accuracies) * 0.97, f"E{i+1}",
                color="#ffffff", fontsize=7, ha="center", alpha=0.55)
    # Annotate final value
    ax.annotate(
        f"Final: {train_accuracies[-1]:.1%}",
        xy=(train_steps[-1], train_accuracies[-1]),
        xytext=(-60, -20), textcoords="offset points",
        color=ACC_COLOR, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=ACC_COLOR, lw=1.2),
    )
    ax.set_title(
        "Train Accuracy  (approx = exp(\u2212loss))  \u2014 Higher is Better",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    fig.tight_layout()
    save(fig, "08a_train_accuracy.png")

    # ── 8b. Eval Accuracy (vs Epoch) ─────────────────────────────────────────
    if eval_accuracies:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eval_epochs, eval_accuracies, color=EVAL_COLOR, marker="o",
                markersize=9, linewidth=2.5, label="Eval Accuracy (approx)")
        ax.fill_between(eval_epochs, eval_accuracies, alpha=0.13, color=EVAL_COLOR)
        # Highlight best epoch
        best_acc_idx = eval_accuracies.index(max(eval_accuracies))
        ax.scatter([eval_epochs[best_acc_idx]], [eval_accuracies[best_acc_idx]],
                   color="#ffd54f", s=140, zorder=5,
                   label=f"Best: {eval_accuracies[best_acc_idx]:.1%} @ epoch {eval_epochs[best_acc_idx]}")
        # Label each point
        for ep, acc in zip(eval_epochs, eval_accuracies):
            ax.text(ep, acc + 0.018, f"{acc:.1%}",
                    ha="center", va="bottom", fontsize=9,
                    color="#e8eaf0", fontweight="bold")
        ax.set_title(
            "Eval Accuracy per Epoch  (approx = exp(\u2212loss))  \u2014 Higher is Better",
            fontsize=14, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(eval_epochs)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend()
        fig.tight_layout()
        save(fig, "08b_eval_accuracy.png")

    # ── 9. Accuracy Improvement per Epoch (bar chart) ─────────────────────────
    if len(eval_accuracies) > 1:
        acc_improvements = [eval_accuracies[i] - eval_accuracies[i-1]
                            for i in range(1, len(eval_accuracies))]
        acc_epoch_labels = [f"E{i}→E{i+1}" for i in range(1, len(eval_accuracies))]
        acc_colors = [BAR_GOOD if v > 0 else BAR_BAD for v in acc_improvements]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(acc_epoch_labels, acc_improvements, color=acc_colors,
                      width=0.5, edgecolor="#0f1117", linewidth=1.2)
        for bar, val in zip(bars, acc_improvements):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.0005 if val >= 0 else -0.002),
                f"{val:+.4f}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10, color="#ffffff", fontweight="bold",
            )
        ax.axhline(0, color="#ffffff", alpha=0.3, linewidth=1)
        ax.set_title("Eval Accuracy Improvement per Epoch (positive = improvement)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Accuracy Gain")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.2%}"))
        fig.tight_layout()
        save(fig, "09_accuracy_improvement.png")

    # ── 10. Summary Dashboard (all graphs in one figure) ──────────────────────
    n_plots = (
        2                                    # train loss + eval loss
        + (1 if train_lrs       else 0)      # LR
        + (1 if train_gnorms    else 0)      # grad norm
        + 1                                  # perplexity
        + 1                                  # train accuracy
        + (1 if eval_accuracies else 0)      # eval accuracy (separate)
    )
    fig = plt.figure(figsize=(16, 4 * max(2, (n_plots + 1) // 2)), facecolor="#0f1117")
    dashboard_title = (
        "MedGemma LoRA Fine-tuning Dashboard  |  "
        f"Model: {args.model}  |  LoRA r={args.lora_r} alpha={args.lora_alpha}  |  "
        f"LR={args.lr}  |  Epochs={args.epochs}  |  "
        f"Effective batch={args.batch_size * args.grad_accum}"
    )
    fig.suptitle(dashboard_title, fontsize=10, color="#e8eaf0", y=1.01)

    gs = gridspec.GridSpec((n_plots + 1) // 2, 2, figure=fig,
                           hspace=0.45, wspace=0.3)
    plot_idx = 0

    def next_ax():
        nonlocal plot_idx
        row, col = divmod(plot_idx, 2)
        ax = fig.add_subplot(gs[row, col])
        plot_idx += 1
        return ax

    # Train loss
    ax = next_ax()
    ax.plot(train_steps, train_losses, color=TRAIN_COLOR)
    ax.set_title("Train Loss", fontsize=11)
    ax.set_xlabel("Step", fontsize=9)

    # Eval loss
    if eval_losses:
        ax = next_ax()
        ax.plot(eval_epochs, eval_losses, color=EVAL_COLOR, marker="o", markersize=6)
        best_idx = eval_losses.index(min(eval_losses))
        ax.scatter([eval_epochs[best_idx]], [eval_losses[best_idx]],
                   color="#ffd54f", s=80, zorder=5)
        ax.set_title("Eval Loss", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_xticks(eval_epochs)

    # Learning rate
    if train_lrs:
        ax = next_ax()
        ax.plot(train_steps[:len(train_lrs)], train_lrs, color=LR_COLOR)
        ax.set_title("Learning Rate", fontsize=11)
        ax.set_xlabel("Step", fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Grad norm
    if train_gnorms:
        ax = next_ax()
        ax.plot(train_steps[:len(train_gnorms)], train_gnorms,
                color=GNORM_COLOR, alpha=0.8)
        ax.set_title("Gradient Norm", fontsize=11)
        ax.set_xlabel("Step", fontsize=9)

    # Perplexity
    ax = next_ax()
    ax.plot(train_steps, train_ppl, color=PPL_COLOR, alpha=0.85, label="Train")
    if eval_ppl:
        ax.plot([int(e * steps_per_epoch) for e in eval_epochs],
                eval_ppl, color=EVAL_COLOR, marker="o", markersize=6, label="Eval")
        ax.legend(fontsize=8)
    ax.set_title("Perplexity", fontsize=11)
    ax.set_xlabel("Step", fontsize=9)

    # Train Accuracy (vs Step)
    ax = next_ax()
    ax.plot(train_steps, train_accuracies, color=ACC_COLOR, alpha=0.88)
    ax.fill_between(train_steps, train_accuracies, alpha=0.12, color=ACC_COLOR)
    ax.set_title("Train Accuracy (approx)", fontsize=11)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Eval Accuracy (vs Epoch) — x-axis = epoch number, NOT steps
    if eval_accuracies:
        ax = next_ax()
        ax.plot(eval_epochs, eval_accuracies, color=EVAL_COLOR, marker="o",
                markersize=7, linewidth=2.2)
        ax.fill_between(eval_epochs, eval_accuracies, alpha=0.12, color=EVAL_COLOR)
        best_acc_idx = eval_accuracies.index(max(eval_accuracies))
        ax.scatter([eval_epochs[best_acc_idx]], [eval_accuracies[best_acc_idx]],
                   color="#ffd54f", s=90, zorder=5,
                   label=f"Best: {eval_accuracies[best_acc_idx]:.1%}")
        ax.set_title("Eval Accuracy (approx)", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_xticks(eval_epochs)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=8)

    save(fig, "10_dashboard.png")

    # ── 11. Save raw log data as JSON for later analysis ──────────────────────
    log_path = out / "training_logs.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"  Saved: {log_path}")

    print(f"  All graphs saved to: {out.resolve()}/")
    print(f"  Files:")
    for f in sorted(out.glob("*.png")):
        print(f"    {f.name}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    hf_token = resolve_token(args.hf_token)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 0. Pin GPU — must happen before any CUDA allocations
    print("\n[0/4] Selecting GPU...")
    device = pin_gpu(args.gpu)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Output   : {args.output}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size} × {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  LR       : {args.lr}  |  LoRA r={args.lora_r} α={args.lora_alpha}")
    print(f"  Max len  : {args.max_len}")
    print(f"{'='*60}\n")

    # 1. Dataset
    print("[1/4] Loading dataset...")
    if args.dataset is None:
        args.dataset = autodetect_parquet()
        print(f"  Auto-detected: {args.dataset}")
    elif not Path(args.dataset).exists():
        sys.exit("[ERROR] File not found: " + repr(args.dataset))
    train_ds, val_ds = load_splits(args.dataset, args.val_split, args.seed)

    # 2. Model — load in bfloat16, single device
    print("\n[2/4] Loading model and processor...")
    load_kw: Dict[str, Any] = {"dtype": torch.bfloat16}
    if hf_token:
        load_kw["token"] = hf_token

    try:
        processor = AutoProcessor.from_pretrained(args.model, **load_kw)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            device_map={"": 0} if device == "cuda" else None,  # force single GPU
            **load_kw,
        )
    except Exception as e:
        sys.exit(
            f"[ERROR] Could not load model.\n  {e}\n\n"
            "  1. Accept model terms: huggingface.co/google/medgemma-1.5-4b-it\n"
            "  2. export HF_TOKEN=hf_...\n"
            "  3. Free up GPU memory — need ~8 GB for bfloat16\n"
            "     Check with: nvidia-smi"
        )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    print("  → Model loaded in bfloat16")

    # 3. LoRA
    print("\n[3/4] Applying LoRA adapters...")
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.enable_input_require_grads()
    model = apply_lora(model, args.lora_r, args.lora_alpha)

    # 4. Train
    print("\n[4/4] Starting training...")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                    = str(out),
        num_train_epochs              = args.epochs,
        per_device_train_batch_size   = args.batch_size,
        per_device_eval_batch_size    = args.batch_size,
        gradient_accumulation_steps   = args.grad_accum,
        learning_rate                 = args.lr,
        lr_scheduler_type             = "cosine",
        warmup_steps                  = 5,
        bf16                          = (device == "cuda"),
        fp16                          = False,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        logging_steps                 = 10,
        eval_strategy                 = "steps",
        eval_steps                    = 10
        save_strategy                 = "steps",
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        save_total_limit              = 2,
        report_to                     = "none",
        dataloader_num_workers        = 0,
        remove_unused_columns         = False,
        seed                          = args.seed,
        push_to_hub                   = args.push_to_hub,
        hub_model_id                  = args.hub_repo,
        hub_token                     = hf_token,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = VQACollator(processor, args.max_len),
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # 5. Save
    print(f"\n✅  Saving model → {out}")
    trainer.save_model(str(out))
    save_processor_safe(processor, str(out))

    if args.push_to_hub and args.hub_repo:
        print(f"  Pushing to Hub: {args.hub_repo}")
        trainer.push_to_hub()

    logs = trainer.state.log_history
    train_losses = [l["loss"]      for l in logs if "loss"      in l]
    eval_losses  = [l["eval_loss"] for l in logs if "eval_loss" in l]
    print("\nTraining summary:")
    if train_losses: print(f"  Final train loss : {train_losses[-1]:.4f}")
    if eval_losses:  print(f"  Best eval loss   : {min(eval_losses):.4f}")
    print(f"  Saved to         : {out.resolve()}")

    # ── Plot all training graphs ──────────────────────────────────────────────
    print("\n📊 Generating training graphs...")
    try:
        plot_training_graphs(logs, str(out), args)
    except Exception as e:
        print(f"  ⚠  Graph generation failed: {e}")
        print("     Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
