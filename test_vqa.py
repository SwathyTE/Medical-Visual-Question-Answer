#!/usr/bin/env python3
"""
test_medgemma_vqa.py
=====================
Zero-shot inference with google/medgemma-4b-it on a local image.
Runs all 7 RadImageNet-VQA questions and prints:
  Image + Question + Reasoning + Answer  for every question.

Usage
-----
  python test_medgemma_vqa.py
  python test_medgemma_vqa.py --image image.jpg --gpu 7

Requirements
------------
  pip install torch transformers accelerate pillow
  export HF_TOKEN=hf_...
"""

import argparse
import os
import sys
import time
import textwrap

try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError as e:
    sys.exit(
        f"[ERROR] Missing dependency: {e}\n"
        "  pip install torch transformers accelerate pillow"
    )

# ── config ─────────────────────────────────────────────────────────
MODEL_ID   = "google/medgemma-4b-it"
IMAGE_PATH = "image.jpg"

# ── strong system prompt that forces reasoning for EVERY question ───
SYSTEM_PROMPT = """You are an expert radiologist AI. For every question you MUST respond in this exact two-part format with no exceptions:

<reasoning>
Write 2-4 sentences of detailed clinical reasoning based only on what is visible in the image. For yes/no questions explain what visual features led to your answer. For multiple-choice questions explain why you selected that option and why the others are less likely.
</reasoning>
<answer>
Write the concise final answer only (one word, yes/no, or the option text).
</answer>

Never skip the reasoning block. Never answer without explaining your visual observations first."""

# ── all 9 Q/A pairs matching the RadImageNet-VQA instruct format ───
QUESTIONS = [
    "Which body part is visible in this image?",
    "Is the abdomen visible in the image?",
    "Can you identify the shoulder in this image?",
    (
        "What is the correct anatomical region for this scan?\n"
        "A. knee\nB. abdomen\nC. brain\nD. hip"
    ),
    "Is there an abnormality present?",
    "What condition is affecting the abdomen?",
    "Is there evidence of abnormal entire organ in this image?",
    "Are there signs of dilated urinary tract?",
    (
        "What is the most likely pathology in this image?\n"
        "A. abnormal entire organ\nB. renal lesion\n"
        "C. no pathology seen\nD. liver lesion"
    ),
]


# ── args ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    type=str, default=IMAGE_PATH)
    p.add_argument("--model",    type=str, default=MODEL_ID)
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--gpu",      type=int, default=None)
    return p.parse_args()


# ── GPU ────────────────────────────────────────────────────────────
def pin_gpu(requested):
    if not torch.cuda.is_available():
        print("  No GPU — using CPU")
        return "cpu"
    n   = torch.cuda.device_count()
    idx = (requested % n) if requested is not None else \
          max((torch.cuda.mem_get_info(i)[0], i) for i in range(n))[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    torch.cuda.empty_cache()
    free_gb  = torch.cuda.mem_get_info(0)[0] / 1024**3
    total_gb = torch.cuda.mem_get_info(0)[1] / 1024**3
    print(f"  GPU {idx}: {torch.cuda.get_device_name(0)} "
          f"({free_gb:.1f}/{total_gb:.1f} GiB free)")
    return "cuda"


# ── image ──────────────────────────────────────────────────────────
def load_image(path):
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] Image not found: '{path}'\n"
            f"  Full path checked: {os.path.abspath(path)}\n"
            "  Pass the correct path with --image /path/to/image.jpg"
        )
    img = Image.open(path).convert("RGB")
    print(f"  Loaded : {path}  ({img.width}x{img.height} px)")
    return img


# ── parse model output — robust to tags missing ────────────────────
def parse_output(raw: str) -> tuple[str, str]:
    """
    Try to extract <reasoning>…</reasoning> and <answer>…</answer>.
    Falls back gracefully if the model omits the tags.
    """
    raw = raw.strip()

    has_r = "<reasoning>" in raw and "</reasoning>" in raw
    has_a = "<answer>"    in raw and "</answer>"    in raw

    if has_r and has_a:
        reasoning = raw.split("<reasoning>")[1].split("</reasoning>")[0].strip()
        answer    = raw.split("<answer>")[1].split("</answer>")[0].strip()
        return reasoning, answer

    if has_a:
        answer    = raw.split("<answer>")[1].split("</answer>")[0].strip()
        reasoning = raw.split("<answer>")[0].replace("<reasoning>","").strip()
        return reasoning, answer

    # No tags at all — split at the last sentence as a best guess
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    if len(lines) >= 2:
        return "\n".join(lines[:-1]), lines[-1]
    return raw, raw   # return full text for both if very short


# ── inference for one question ─────────────────────────────────────
def ask(model, processor, image: Image.Image, question: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    t0 = time.time()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=400,   # enough for full reasoning
            do_sample=False,
            repetition_penalty=1.1,
        )
    elapsed = round(time.time() - t0, 2)

    raw = processor.decode(
        out_ids[0][input_len:], skip_special_tokens=True
    ).strip()

    reasoning, answer = parse_output(raw)
    return {"reasoning": reasoning, "answer": answer, "time_s": elapsed}


# ── display ────────────────────────────────────────────────────────
def print_result(idx: int, question: str, result: dict):
    W = 70
    print(f"\n{'='*W}")
    print(f"  Q{idx}  {question}")
    print(f"{'─'*W}")

    # wrap reasoning nicely
    r_lines = result["reasoning"].splitlines()
    print("  REASONING:")
    for line in r_lines:
        if line.strip():
            for wrapped in textwrap.wrap(line.strip(), width=W - 4):
                print(f"    {wrapped}")
        else:
            print()

    print(f"{'─'*W}")
    print(f"  ANSWER:  {result['answer']}")
    print(f"  Time  :  {result['time_s']}s")


# ── main ───────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    print("\n" + "=" * 70)
    print("  MedGemma-4B-IT  |  Medical VQA  |  Zero-Shot (No Fine-Tuning)")
    print("  Model  : google/medgemma-4b-it")
    print("  Dataset: raidium/RadImageNet-VQA")
    print("=" * 70)

    print("\n[0/3] Selecting GPU...")
    device = pin_gpu(args.gpu)

    print("\n[1/3] Loading image...")
    image = load_image(args.image)

    print(f"\n[2/3] Loading {args.model} ...")
    load_kw = {"dtype": torch.bfloat16}
    if hf_token:
        load_kw["token"] = hf_token

    try:
        processor = AutoProcessor.from_pretrained(args.model, **load_kw)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            device_map={"": 0} if device == "cuda" else None,
            **load_kw,
        )
    except Exception as e:
        sys.exit(
            f"\n[ERROR] Could not load model:\n  {e}\n\n"
            "  Steps to fix:\n"
            "  1. Accept terms at https://huggingface.co/google/medgemma-4b-it\n"
            "  2. export HF_TOKEN=hf_your_token_here\n"
            "  3. Need ~8 GB VRAM — check: nvidia-smi"
        )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.eval()
    print("  Model ready.\n")

    print(f"[3/3] Running {len(QUESTIONS)} questions on '{args.image}'...")

    total_time = 0
    for i, question in enumerate(QUESTIONS, 1):
        print(f"  → Q{i}/{len(QUESTIONS)} ...", end=" ", flush=True)
        result = ask(model, processor, image, question)
        total_time += result["time_s"]
        print(f"done ({result['time_s']}s)")
        print_result(i, question, result)

    print(f"\n{'='*70}")
    print(f"  Completed {len(QUESTIONS)} questions  |  Total time: {total_time:.1f}s")
    print(f"  Image: {args.image}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
