"""
Evaluation Script: Fine-tuned MedGemma on RadImageNet-VQA (1000 samples)
Model:   medgemma-radimagenet-vqa-full  (local fine-tuned checkpoint)
Dataset: raidium/RadImageNet-VQA  —  instruct / val split
Metrics: Accuracy, Precision, Recall, F1 (classification)
         BLEU, ROUGE-1/2/L          (generation quality)
"""

# ──────────────────────────────────────────────────────────────
# 0. Imports & dependency check
# ──────────────────────────────────────────────────────────────
import os, re, json, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── install missing packages quietly ──────────────────────────
import subprocess, sys

def _install(pkg):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

#for pkg in ["datasets", "transformers", "torch", "Pillow",
 #           "nltk", "rouge_score", "scikit-learn", "accelerate",
 #           "sentencepiece", "bitsandbytes", "peft"]:
    try:
        __import__(pkg.replace("-", "_").split("[")[0])
    except ImportError:
        print(f"Installing {pkg} …")
        _install(pkg)

# ── now safe to import ─────────────────────────────────────────
import torch
import nltk
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, PeftConfig
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from rouge_score import rouge_scorer as rouge_lib
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ──────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────
FINETUNED_MODEL_ID = "medgemma-radimagenet-vqa-full"   # local or HF repo
BASE_MODEL_ID      = "google/medgemma-4b-it"           # fallback / processor source
DATASET_ID         = "raidium/RadImageNet-VQA"
DATASET_SUBSET     = "instruct"
DATASET_SPLIT      = "val"
NUM_SAMPLES        = 1000
MAX_NEW_TOKENS     = 64
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE              = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_JSON       = "medgemma_eval_results.json"

print(f"Device : {DEVICE}")
print(f"Dtype  : {DTYPE}")
print(f"Samples: {NUM_SAMPLES}\n")

# ──────────────────────────────────────────────────────────────
# 2. Load dataset
# ──────────────────────────────────────────────────────────────
print("Loading dataset …")
ds = load_dataset(DATASET_ID, DATASET_SUBSET, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))
print(f"  Loaded {len(ds)} samples from {DATASET_SPLIT} split.\n")

# ──────────────────────────────────────────────────────────────
# 3. Load fine-tuned model & processor
#    Handles two cases automatically:
#      A) Full / merged checkpoint  -> load directly
#      B) PEFT / LoRA adapter       -> load base model, then wrap
#         (fixes KeyError: 'llava' in transformers' broken PEFT integration)
# ──────────────────────────────────────────────────────────────
print(f"Loading model: {FINETUNED_MODEL_ID} ...")

model_path = Path(FINETUNED_MODEL_ID)
load_id    = str(model_path) if model_path.exists() else FINETUNED_MODEL_ID

# ── Detect whether this is a PEFT adapter directory ──────────
def _is_peft_adapter(path_or_id):
    """Return True if the directory/repo looks like a PEFT adapter."""
    p = Path(path_or_id)
    if p.exists():
        return (p / "adapter_config.json").exists()
    try:
        from peft import PeftConfig as _PC
        _PC.from_pretrained(path_or_id)
        return True
    except Exception:
        return False

is_peft = _is_peft_adapter(load_id)
print(f"  Checkpoint type: {'PEFT/LoRA adapter' if is_peft else 'full / merged weights'}")

# ── Processor ────────────────────────────────────────────────
try:
    processor = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
    print("  Processor loaded from fine-tuned checkpoint.")
except Exception:
    print("  Processor not found in checkpoint; loading from base model.")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

# ── Model ─────────────────────────────────────────────────────
device_map = "auto" if DEVICE == "cuda" else None

if is_peft:
    from peft import PeftModel, PeftConfig
    # Step 1: read base model ID from adapter_config.json
    peft_cfg = PeftConfig.from_pretrained(load_id)
    base_id  = peft_cfg.base_model_name_or_path
    print(f"  Base model (from adapter_config): {base_id}")

    # Step 2: load base model WITHOUT going through transformers' PEFT path
    print("  Loading base model weights ...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_id,
        dtype=DTYPE,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Step 3: attach adapter via peft directly — bypasses the broken integration
    print("  Attaching LoRA adapter ...")
    model = PeftModel.from_pretrained(base_model, load_id, is_trainable=False)

    # Merge weights for faster inference (optional but recommended)
    try:
        model = model.merge_and_unload()
        print("  LoRA weights merged into base model.")
    except Exception as e:
        print(f"  merge_and_unload skipped ({e}); running with adapter attached.")
else:
    # Full / merged checkpoint — load directly
    model = AutoModelForImageTextToText.from_pretrained(
        load_id,
        dtype=DTYPE,
        device_map=device_map,
        trust_remote_code=True,
    )

if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()
print("  Model ready.\n")

# ──────────────────────────────────────────────────────────────
# 4. Helper: extract last human question & template answer
# ──────────────────────────────────────────────────────────────

def parse_conversation(conv_list):
    """
    Return (question, ground_truth_answer) from a conversation list.
    Conversations alternate: human → template → human → template …
    We pick the LAST human turn as the question and the following
    template turn as the ground-truth answer.
    """
    question, gt_answer = "", ""
    for i, turn in enumerate(conv_list):
        if turn.get("from") == "human":
            # Strip the image placeholder from the first turn
            val = turn["value"].replace("<image>\n", "").strip()
            if val:
                question  = val
                # next turn should be template answer
                if i + 1 < len(conv_list):
                    gt_answer = conv_list[i + 1].get("value", "").strip()
    return question, gt_answer


# ──────────────────────────────────────────────────────────────
# 5. Helper: generate model answer
# ──────────────────────────────────────────────────────────────

def generate_answer(image: Image.Image, question: str) -> str:
    """Run a single forward pass and return decoded text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ],
        }
    ]
    try:
        # MedGemma / Gemma-3 style processor (apply_chat_template)
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
        ).to(DEVICE)
    except Exception:
        # Fallback: plain text + image
        inputs = processor(
            text=question,
            images=image,
            return_tensors="pt",
        ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode only newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────
# 6. Canonical answer normalisation (for classification metrics)
# ──────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lower-case, strip punctuation/whitespace for comparison."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.I)

def extract_yn(text: str):
    m = YES_NO_PATTERN.search(text)
    return m.group(0).lower() if m else None


# ──────────────────────────────────────────────────────────────
# 7. Evaluation loop
# ──────────────────────────────────────────────────────────────
predictions_raw, references_raw = [], []   # full strings (for BLEU / ROUGE)
pred_labels,     ref_labels     = [], []   # normalised (for accuracy etc.)

print("Running inference …\n")
t0 = time.time()

for idx, sample in enumerate(ds):
    image = sample["image"]                    # PIL image
    conv  = sample["conversations"]            # list of dicts

    question, gt_answer = parse_conversation(conv)
    if not question:
        continue

    pred_answer = generate_answer(image, question)

    predictions_raw.append(pred_answer)
    references_raw.append(gt_answer)
    pred_labels.append(normalise(pred_answer))
    ref_labels.append(normalise(gt_answer))

    if (idx + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  [{idx+1:4d}/{NUM_SAMPLES}]  {elapsed:.0f}s elapsed  "
              f"| Q: {question[:60]!r}  "
              f"| GT: {gt_answer!r}  "
              f"| Pred: {pred_answer!r}")

total_time = time.time() - t0
print(f"\nInference done in {total_time:.1f}s  "
      f"({total_time/len(predictions_raw):.2f}s/sample)\n")

# ──────────────────────────────────────────────────────────────
# 8. Classification metrics  (exact-match at token level)
# ──────────────────────────────────────────────────────────────

# Build a shared label set from all unique ground-truth answers
unique_labels = sorted(set(ref_labels))
label2id      = {l: i for i, l in enumerate(unique_labels)}

# Map predictions to the closest known label (or "unknown")
def map_label(pred, label2id):
    if pred in label2id:
        return label2id[pred]
    # partial match – pick first containing label
    for lbl in label2id:
        if lbl and lbl in pred:
            return label2id[lbl]
    return -1   # unknown / out-of-vocab

y_pred = [map_label(p, label2id) for p in pred_labels]
y_true = [label2id[r]            for r in ref_labels]

# Filter out unknowns for cleaner metrics
valid = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp != -1]
if valid:
    y_true_v, y_pred_v = zip(*valid)
else:
    y_true_v, y_pred_v = y_true, y_pred

avg = "macro"
accuracy  = accuracy_score(y_true_v, y_pred_v)
precision = precision_score(y_true_v, y_pred_v, average=avg, zero_division=0)
recall    = recall_score   (y_true_v, y_pred_v, average=avg, zero_division=0)
f1        = f1_score       (y_true_v, y_pred_v, average=avg, zero_division=0)

# ──────────────────────────────────────────────────────────────
# 9. BLEU score  (corpus-level)
# ──────────────────────────────────────────────────────────────
tokenised_preds = [nltk.word_tokenize(p.lower()) for p in predictions_raw]
tokenised_refs  = [[nltk.word_tokenize(r.lower())] for r in references_raw]

smoother = SmoothingFunction().method4
bleu_score = corpus_bleu(tokenised_refs, tokenised_preds,
                         smoothing_function=smoother)

# ──────────────────────────────────────────────────────────────
# 10. ROUGE scores
# ──────────────────────────────────────────────────────────────
scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

r1_list, r2_list, rL_list = [], [], []
for pred, ref in zip(predictions_raw, references_raw):
    s = scorer.score(ref, pred)
    r1_list.append(s["rouge1"].fmeasure)
    r2_list.append(s["rouge2"].fmeasure)
    rL_list.append(s["rougeL"].fmeasure)

rouge1 = np.mean(r1_list)
rouge2 = np.mean(r2_list)
rougeL = np.mean(rL_list)

# ──────────────────────────────────────────────────────────────
# 11. Yes/No sub-metrics  (binary classification on yn-questions)
# ──────────────────────────────────────────────────────────────
yn_pred, yn_true = [], []
for p, r in zip(predictions_raw, references_raw):
    rn, pn = extract_yn(r), extract_yn(p)
    if rn is not None:
        yn_true.append(1 if rn == "yes" else 0)
        yn_pred.append(1 if pn == "yes" else 0)

if yn_true:
    yn_acc  = accuracy_score (yn_true, yn_pred)
    yn_prec = precision_score(yn_true, yn_pred, zero_division=0)
    yn_rec  = recall_score   (yn_true, yn_pred, zero_division=0)
    yn_f1   = f1_score       (yn_true, yn_pred, zero_division=0)
else:
    yn_acc = yn_prec = yn_rec = yn_f1 = 0.0

# ──────────────────────────────────────────────────────────────
# 12. Print results
# ──────────────────────────────────────────────────────────────
SEP = "─" * 55

print(SEP)
print(" EVALUATION RESULTS")
print(SEP)
print(f"  Samples evaluated : {len(predictions_raw)}")
print(f"  Total time        : {total_time:.1f}s")
print()
print("  ── Classification Metrics (exact-match) ──")
print(f"  Accuracy          : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision (macro) : {precision:.4f}")
print(f"  Recall    (macro) : {recall:.4f}")
print(f"  F1-Score  (macro) : {f1:.4f}")
print()
print("  ── Yes/No Binary Metrics ──")
print(f"  Accuracy          : {yn_acc:.4f}")
print(f"  Precision         : {yn_prec:.4f}")
print(f"  Recall            : {yn_rec:.4f}")
print(f"  F1-Score          : {yn_f1:.4f}")
print(f"  (on {len(yn_true)} yes/no samples)")
print()
print("  ── Generation Quality Metrics ──")
print(f"  BLEU              : {bleu_score:.4f}")
print(f"  ROUGE-1           : {rouge1:.4f}")
print(f"  ROUGE-2           : {rouge2:.4f}")
print(f"  ROUGE-L           : {rougeL:.4f}")
print(SEP)

# ──────────────────────────────────────────────────────────────
# 13. Save results to JSON
# ──────────────────────────────────────────────────────────────
results = {
    "model"            : FINETUNED_MODEL_ID,
    "dataset"          : DATASET_ID,
    "split"            : DATASET_SPLIT,
    "num_samples"      : len(predictions_raw),
    "total_time_sec"   : round(total_time, 2),
    "classification": {
        "accuracy"     : round(accuracy,  4),
        "precision"    : round(precision, 4),
        "recall"       : round(recall,    4),
        "f1"           : round(f1,        4),
    },
    "yes_no_binary": {
        "accuracy"     : round(yn_acc,  4),
        "precision"    : round(yn_prec, 4),
        "recall"       : round(yn_rec,  4),
        "f1"           : round(yn_f1,   4),
        "num_samples"  : len(yn_true),
    },
    "generation": {
        "bleu"         : round(bleu_score, 4),
        "rouge1"       : round(rouge1, 4),
        "rouge2"       : round(rouge2, 4),
        "rougeL"       : round(rougeL, 4),
    },
    "sample_predictions": [
        {"question": q, "ground_truth": r, "prediction": p}
        for q, r, p in zip(
            [parse_conversation(s["conversations"])[0] for s in ds],
            references_raw[:10],
            predictions_raw[:10],
        )
    ],
}

with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {RESULTS_JSON}")
