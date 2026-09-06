"""
MedGemma Reasoning Dataset — Output Viewer
============================================
Reads the dataset created by create_medgemma_dataset.py and prints
the Question, Chain-of-Thought Reasoning, and Answer for all 5 samples.

Two modes:
  1. DATASET MODE  — reads directly from medgemma_reasoning_chat.jsonl (no GPU needed)
  2. INFERENCE MODE — loads fine-tuned / base MedGemma model and runs live inference

Usage:
    # Mode 1: Read dataset file (default, no model needed)
    python run_medgemma_inference.py

    # Mode 2: Run live model inference
    python run_medgemma_inference.py --model google/medgemma-1.5-4b-it

Requirements (mode 2 only):
    pip install transformers torch accelerate
"""

import json
import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI color codes for terminal pretty-printing
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    WHITE   = "\033[97m"
    DIM     = "\033[2m"
    RED     = "\033[91m"

CATEGORY_COLORS = {
    "Cardiology":    C.RED,
    "Pharmacology":  C.GREEN,
    "Pulmonology":   C.CYAN,
    "Neurology":     C.MAGENTA,
    "Obstetrics":    C.YELLOW,
}

DIFFICULTY_COLORS = {
    "Hard":   C.RED,
    "Medium": C.YELLOW,
    "Easy":   C.GREEN,
}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_divider(char: str = "═", width: int = 72) -> None:
    print(f"{C.DIM}{char * width}{C.RESET}")


def print_sample(index: int, sample_id: str, category: str, difficulty: str,
                 question: str, reasoning: str, answer: str, explanation: str) -> None:
    """Prints a single dataset sample in a readable, structured format."""

    cat_color  = CATEGORY_COLORS.get(category, C.WHITE)
    diff_color = DIFFICULTY_COLORS.get(difficulty, C.WHITE)

    print_divider("═")
    # Header
    print(
        f"{C.BOLD}{C.WHITE} Sample {index}  "
        f"{C.RESET}{C.DIM}[{sample_id}]{C.RESET}  "
        f"{cat_color}{C.BOLD}{category}{C.RESET}  "
        f"{diff_color}▸ {difficulty}{C.RESET}"
    )
    print_divider("─")

    # Question
    print(f"\n{C.BOLD}{C.CYAN}QUESTION{C.RESET}")
    for line in question.strip().splitlines():
        print(f"  {line}")

    # Reasoning
    print(f"\n{C.BOLD}{C.YELLOW}REASONING{C.RESET}")
    for line in reasoning.strip().splitlines():
        # Highlight step headers
        if re.match(r"^Step \d+", line):
            print(f"  {C.BOLD}{C.BLUE}{line}{C.RESET}")
        elif line.startswith("Conclusion"):
            print(f"  {C.BOLD}{C.GREEN}{line}{C.RESET}")
        else:
            print(f"  {line}")

    # Answer
    print(f"\n{C.BOLD}{C.GREEN}ANSWER{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}Correct Option: {C.GREEN}{answer}{C.RESET}")
    print(f"\n{C.BOLD}Explanation:{C.RESET}")
    for line in explanation.strip().splitlines():
        print(f"  {line}")

    print()


# ---------------------------------------------------------------------------
# Mode 1: Read directly from JSONL dataset file
# ---------------------------------------------------------------------------

def extract_from_assistant_message(content: str) -> tuple[str, str, str]:
    """
    Parses the assistant message content into:
        reasoning    — text inside <reasoning>...</reasoning>
        answer       — letter after **Answer: ...**
        explanation  — text after the answer line
    """
    # Extract reasoning block
    reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", content, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Not found"

    # Extract answer letter
    answer_match = re.search(r"\*\*Answer:\s*([A-D])\*\*", content)
    answer = answer_match.group(1) if answer_match else "?"

    # Extract explanation (everything after the **Answer:** line)
    explanation_match = re.search(r"\*\*Answer:\s*[A-D]\*\*\s*\n+(.*)", content, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return reasoning, answer, explanation


def run_dataset_mode(jsonl_path: str) -> None:
    """Reads and displays all samples from the JSONL dataset file."""

    path = Path(jsonl_path)
    if not path.exists():
        print(f"{C.RED}Error: Dataset file not found: {jsonl_path}{C.RESET}")
        print("Run create_medgemma_dataset.py first to generate the dataset.")
        return

    print(f"\n{C.BOLD}{C.WHITE}MedGemma Reasoning Dataset — Output Viewer{C.RESET}")
    print(f"{C.DIM}Source: {path.resolve()}{C.RESET}\n")

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"{C.DIM}Loaded {len(records)} samples.{C.RESET}\n")

    for i, record in enumerate(records, 1):
        sample_id  = record.get("id", f"sample_{i}")
        category   = record.get("category", "Unknown")
        difficulty = record.get("difficulty", "Unknown")
        messages   = record.get("messages", [])

        # Extract user question and assistant response
        question    = next((m["content"] for m in messages if m["role"] == "user"), "")
        asst_content = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        reasoning, answer, explanation = extract_from_assistant_message(asst_content)

        print_sample(
            index=i,
            sample_id=sample_id,
            category=category,
            difficulty=difficulty,
            question=question,
            reasoning=reasoning,
            answer=answer,
            explanation=explanation,
        )

    print_divider("═")
    print(f"\n{C.BOLD}{C.GREEN}✓ All {len(records)} samples displayed.{C.RESET}\n")


# ---------------------------------------------------------------------------
# Mode 2: Live model inference
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are MedGemma, an expert medical AI assistant. "
    "For every clinical question, you must first reason through the problem step by step "
    "inside <reasoning>...</reasoning> tags, then state the correct answer letter in bold, "
    "followed by a concise clinical explanation.\n\n"
    "Format your response as:\n"
    "<reasoning>\n"
    "[detailed step-by-step clinical reasoning]\n"
    "</reasoning>\n\n"
    "**Answer: [letter]**\n\n"
    "[brief explanation]"
)


def run_inference_mode(model_id: str, jsonl_path: str) -> None:
    """Loads the model and runs live inference on each sample's question."""

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError:
        print(f"{C.RED}Error: transformers and torch are required for inference mode.{C.RESET}")
        print("Install them: pip install transformers torch accelerate")
        return

    print(f"\n{C.BOLD}{C.WHITE}MedGemma Inference Mode{C.RESET}")
    print(f"{C.DIM}Model: {model_id}{C.RESET}")
    print(f"{C.DIM}Loading model... (this may take a few minutes){C.RESET}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,
        do_sample=False,
        temperature=1.0,
    )

    path = Path(jsonl_path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    for i, record in enumerate(records, 1):
        sample_id  = record.get("id", f"sample_{i}")
        category   = record.get("category", "Unknown")
        difficulty = record.get("difficulty", "Unknown")
        messages   = record.get("messages", [])

        question = next((m["content"] for m in messages if m["role"] == "user"), "")

        print(f"{C.DIM}Running inference on sample {i}/{len(records)}...{C.RESET}")

        # Build prompt (system + user in Gemma format)
        prompt_messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = gen_pipeline(prompt)[0]["generated_text"]

        # Strip the prompt from the output
        generated = output[len(prompt):].strip()

        reasoning, answer, explanation = extract_from_assistant_message(generated)

        print_sample(
            index=i,
            sample_id=sample_id,
            category=category,
            difficulty=difficulty,
            question=question,
            reasoning=reasoning,
            answer=answer,
            explanation=explanation,
        )

    print_divider("═")
    print(f"\n{C.BOLD}{C.GREEN}✓ Inference complete for all {len(records)} samples.{C.RESET}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display Question → Reasoning → Answer for MedGemma dataset samples"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID or local path for live inference. "
            "If not provided, reads directly from the JSONL dataset file."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="medgemma_reasoning_chat.jsonl",
        help="Path to the JSONL dataset file (default: medgemma_reasoning_chat.jsonl)",
    )
    args = parser.parse_args()

    if args.model:
        run_inference_mode(model_id=args.model, jsonl_path=args.dataset)
    else:
        run_dataset_mode(jsonl_path=args.dataset)


if __name__ == "__main__":
    main()
