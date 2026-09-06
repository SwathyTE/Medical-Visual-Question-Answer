#!/usr/bin/env python3
"""
test_medgemma_inference.py
==========================
Feeds ONE real image + question to google/medgemma-1.5-4b-it and prints
the model's structured reasoning + answer.

This script mirrors the EXACT prompt format used in finetune_medgemma.py:
  - Same SYSTEM_PROMPT
  - Same message structure: system → user (image + question)
  - Same processor.apply_chat_template() call
  - Same <reasoning>...</reasoning><answer>...</answer> output format

✅  MedGemma-1.5-4B-IT is MULTIMODAL — it accepts real image files
   (JPEG, PNG, BMP, TIFF, etc.) via its SigLIP vision encoder.

Model access (REQUIRED — gated repo)
--------------------------------------
  1. Visit  https://huggingface.co/google/medgemma-1.5-4b-it
  2. Log in → click "Agree and access repository"
  3. Generate a token at https://huggingface.co/settings/tokens
  4. export HF_TOKEN=hf_xxxxxxxxxxxx

Usage
-----
  # Auto-downloads a sample chest X-ray, uses built-in question
  python test_medgemma_inference.py

  # Your own image
  python test_medgemma_inference.py --image /path/to/scan.jpg

  # Custom image + question
  python test_medgemma_inference.py \
      --image retina.png \
      --question "Is there signs of diabetic retinopathy?"

  # Point to a local fine-tuned checkpoint
  python test_medgemma_inference.py --model ./medgemma-vqa-finetuned

  # Greedy decoding (fully deterministic)
  python test_medgemma_inference.py --greedy

Prerequisites
-------------
  pip install torch transformers accelerate pillow sentencepiece
"""

import argparse
import io
import os
import sys
import textwrap
import urllib.request
from pathlib import Path

# ── dependency guard ──────────────────────────────────────────────────────────
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError as e:
    sys.exit(
        f"[ERROR] Missing dependency: {e}\n"
        "  pip install torch transformers accelerate pillow sentencepiece"
    )

# ═════════════════════════════════════════════════════════════════════════════
#  ✏️  EDIT THESE to test your own image and question
# ═════════════════════════════════════════════════════════════════════════════

# Path to your image file (JPEG, PNG, …).
# Set to None to auto-download a public sample chest X-ray.
IMAGE_PATH = "image.jpg"   # local image from raidium/RadImageNet-VQA

# The clinical question to ask about the image
QUESTION = (
    "What are the key findings in this medical image? "
    "Provide your step-by-step clinical reasoning and give the most likely diagnosis "
    "along with recommended management."
)

# HuggingFace model name or path to a local fine-tuned checkpoint
MODEL_NAME = "google/medgemma-1.5-4b-it"

# ═════════════════════════════════════════════════════════════════════════════

# ── Exact SYSTEM_PROMPT from finetune_medgemma.py ────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert radiologist and medical imaging assistant. "
    "Given a medical image and a question, provide concise clinical reasoning "
    "followed by the answer.\n\n"
    "Always respond in this exact format:\n"
    "<reasoning>\n[2-3 sentence clinical reasoning based on visual findings]\n</reasoning>\n"
    "<answer>\n[concise final answer]\n</answer>"
)

# Fallback: download the same image directly from the HuggingFace dataset repo
# This mirrors the local path: LLM/SFT_MCQ/GemmaLlama/VQA/image.jpg
SAMPLE_XRAY_URL = (
    "image.jpg"
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers — match finetune_medgemma.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def bytes_to_pil(b) -> Image.Image:
    """Convert bytes / bytearray / PIL Image → RGB PIL Image."""
    if isinstance(b, Image.Image):
        return b.convert("RGB")
    if isinstance(b, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(b))).convert("RGB")
    if hasattr(b, "tobytes"):
        return Image.open(io.BytesIO(b.tobytes())).convert("RGB")
    raise ValueError(f"Cannot convert {type(b)} to PIL Image")


def load_image(image_path) -> tuple:
    """
    Load image from a local file or download the sample chest X-ray.
    Returns (PIL.Image.Image, source_label_str).
    """
    # Fall back to the hardcoded IMAGE_PATH constant if no --image arg given
    if image_path is None:
        image_path = IMAGE_PATH

    if image_path:
        p = Path(image_path)
        if not p.exists():
            sys.exit(f"[ERROR] Image file not found: {image_path}")
        img = Image.open(p).convert("RGB")
        print(f"  ✓  Loaded: {p.resolve()}  ({img.size[0]}×{img.size[1]} px)")
        return img, str(p.resolve())

    print("  No --image provided — downloading sample chest X-ray ...")
    print(f"  URL: {SAMPLE_XRAY_URL}")
    try:
        with urllib.request.urlopen(SAMPLE_XRAY_URL, timeout=30) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        print(f"  ✓  Downloaded ({len(raw)//1024} KB, {img.size[0]}×{img.size[1]} px)")
        return img, "Sample chest X-ray PA view (Wikimedia Commons — public domain)"
    except Exception as e:
        sys.exit(
            f"[ERROR] Download failed: {e}\n"
            "  Provide a local image with:  --image path/to/file.jpg"
        )


# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_name: str, hf_token):
    """
    Load AutoProcessor + AutoModelForImageTextToText.
    Mirrors the load pattern in finetune_medgemma.py exactly:
      - dtype=torch.bfloat16
      - device_map={"": 0}  for single GPU
    """
    load_kw = {}
    if hf_token:
        load_kw["token"] = hf_token

    # ── device & dtype ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device     = "cuda"
        dtype      = torch.bfloat16          # same as fine-tune script
        device_map = {"": 0}                 # single GPU — same as fine-tune
        vram_free, vram_total = torch.cuda.mem_get_info(0)
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"        {vram_free/1024**3:.1f} / {vram_total/1024**3:.1f} GiB free")
        print(f"  dtype: bfloat16")
    elif torch.backends.mps.is_available():
        device     = "mps"
        dtype      = torch.float16
        device_map = None
        print("  Apple MPS  |  dtype: float16")
    else:
        device     = "cpu"
        dtype      = torch.float32
        device_map = None
        print("  ⚠  CPU only — inference will be slow (~5–15 min for 4B model)")

    # ── processor ─────────────────────────────────────────────────────────────
    print(f"\n  Loading processor ...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_name, dtype=dtype, **load_kw
        )
    except Exception as e:
        err = str(e)
        if "401" in err or "gated" in err.lower() or "access" in err.lower():
            sys.exit(
                "[ERROR] Access denied (401 / gated model).\n\n"
                "  Steps to fix:\n"
                "    1. Visit https://huggingface.co/google/medgemma-1.5-4b-it\n"
                "    2. Log in → click 'Agree and access repository'\n"
                "    3. Get token: https://huggingface.co/settings/tokens\n"
                "    4. Run:  export HF_TOKEN=hf_xxxx\n"
                "       or:  python test_medgemma_inference.py --hf_token hf_xxxx\n"
            )
        sys.exit(f"[ERROR] Processor load failed:\n  {e}")

    # Pad token — same fix as finetune_medgemma.py
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print("  ✓  Processor loaded.")

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"  Loading model weights (first run ~8 GB download) ...")
    model_kw = {"dtype": dtype, **load_kw}
    if device_map:
        model_kw["device_map"] = device_map

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kw)
    except Exception as e:
        sys.exit(
            f"[ERROR] Model load failed:\n  {e}\n\n"
            "  Checklist:\n"
            "    • Need ~8 GB VRAM in bfloat16 — check: nvidia-smi\n"
            "    • HF_TOKEN set and access approved on HF model page\n"
            "    • pip install transformers accelerate pillow sentencepiece"
        )

    if device in ("cpu", "mps"):
        model = model.to(device)

    model.eval()
    print("  ✓  Model loaded.\n")
    return processor, model, device


# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def run_inference(
    processor,
    model,
    device: str,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 512,
    temperature: float  = 0.1,
    do_sample: bool     = True,
    top_p: float        = 0.9,
) -> str:
    """
    Build the multimodal prompt using the EXACT same message structure as
    VQACollator in finetune_medgemma.py, then generate the response.

    Training message format (from VQACollator.__call__):
      [
        {"role": "system",    "content": [{"type": "text",  "text": SYSTEM_PROMPT}]},
        {"role": "user",      "content": [{"type": "image", "image": <PIL>},
                                          {"type": "text",  "text": question}]},
        {"role": "assistant", "content": [{"type": "text",  "text": target}]},  ← omitted at inference
      ]

    Inference uses add_generation_prompt=True instead of the assistant turn.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},   # real PIL.Image object
                {"type": "text",  "text": question},
            ],
        },
    ]

    # Tokenize — mirrors training collator's apply_chat_template call
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,   # appends the assistant opening token
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move to device
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    elif device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[-1]

    # ── generation ────────────────────────────────────────────────────────────
    gen_kwargs = dict(
        **inputs,
        max_new_tokens     = max_new_tokens,
        do_sample          = do_sample and (temperature > 0),
        repetition_penalty = 1.1,
        pad_token_id       = processor.tokenizer.pad_token_id,
        eos_token_id       = processor.tokenizer.eos_token_id,
    )
    if do_sample and temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"]       = top_p

    output_ids = model.generate(**gen_kwargs)

    # Decode ONLY newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][prompt_len:]
    response   = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


# ─────────────────────────────────────────────────────────────────────────────
def pretty_print(image_source: str, question: str, response: str) -> None:
    W   = 74
    SEP = "─" * W

    print(f"\n{'═'*W}")
    print("  google/medgemma-1.5-4b-it  ·  Inference Result")
    print(f"{'═'*W}")

    print(f"\n🖼   IMAGE")
    print(SEP)
    print(f"  {image_source}")

    print(f"\n❓  QUESTION")
    print(SEP)
    for line in textwrap.wrap(question, W - 4):
        print(f"  {line}")

    print(f"\n🤖  RAW MODEL OUTPUT")
    print(SEP)
    print(response)

    # Parse structured blocks
    has_reasoning = "<reasoning>" in response and "</reasoning>" in response
    has_answer    = "<answer>"    in response and "</answer>"    in response

    if has_reasoning or has_answer:
        print(f"\n{'─'*W}")
        print("  📋  PARSED BLOCKS")
        print(f"{'─'*W}")

        if has_reasoning:
            reasoning = (
                response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
            )
            print("\n  🧠  REASONING:")
            for line in reasoning.splitlines():
                stripped = line.strip()
                if stripped:
                    print(f"    {stripped}")

        if has_answer:
            answer = response.split("<answer>")[1].split("</answer>")[0].strip()
            print("\n  ✅  ANSWER:")
            for line in textwrap.wrap(answer, W - 6):
                print(f"    {line}")
    else:
        print(
            "\n  ⚠  Output did not use <reasoning>/<answer> tags.\n"
            "     This is expected for the base model before fine-tuning.\n"
            "     Fine-tune with finetune_medgemma.py to enforce this format.\n"
            "     Try --greedy or lowering --temperature for more focused output."
        )

    print(f"\n{'═'*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Test google/medgemma-1.5-4b-it with one image + question."
    )
    p.add_argument(
        "--image", default=None,
        help="Local image file path (JPEG, PNG …). "
             "Auto-downloads a chest X-ray sample if omitted.",
    )
    p.add_argument(
        "--question", default=QUESTION,
        help="Clinical question to ask about the image.",
    )
    p.add_argument(
        "--model", default=MODEL_NAME,
        help=f"HF model name or local checkpoint path (default: {MODEL_NAME})",
    )
    p.add_argument(
        "--hf_token", default=None,
        help="HuggingFace access token. Reads HF_TOKEN env var if not set.",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens to generate (default: 512)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature — lower = more deterministic (default: 0.1)",
    )
    p.add_argument(
        "--greedy", action="store_true",
        help="Greedy decoding — fully deterministic, overrides --temperature",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print("\n" + "="*74)
    print("  MedGemma-1.5-4B-IT  |  Single Image + Question Inference")
    print("="*74)
    print(f"\n  Model  : {args.model}")
    print(f"  Format : <reasoning>...</reasoning><answer>...</answer>")
    print(f"  Prompt : matches finetune_medgemma.py VQACollator exactly")
    if not hf_token:
        print(
            "\n  ⚠  HF_TOKEN not detected. If you get a 401 error:\n"
            "       export HF_TOKEN=hf_xxxx\n"
            "     or pass:  --hf_token hf_xxxx"
        )
    print()

    # 1. Image
    print("[1/3] Loading image ...")
    image, image_source = load_image(args.image)

    # 2. Model + processor
    print("[2/3] Loading model and processor ...")
    processor, model, device = load_model(args.model, hf_token)

    # 3. Inference
    print("[3/3] Running inference ...")
    response = run_inference(
        processor,
        model,
        device,
        image          = image,
        question       = args.question,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        do_sample      = not args.greedy,
    )

    # 4. Display result
    pretty_print(image_source, args.question, response)


if __name__ == "__main__":
    main()
