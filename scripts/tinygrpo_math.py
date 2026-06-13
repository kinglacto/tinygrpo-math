#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import subprocess
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig
    from math_verify import parse as math_parse
    from math_verify import verify as math_verify

    HAS_MATH_VERIFY = True
    MATH_VERIFY_IMPORT_ERROR = None
except Exception as exc:
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    math_parse = None
    math_verify = None
    HAS_MATH_VERIFY = False
    MATH_VERIFY_IMPORT_ERROR = str(exc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TinyGRPO-Math: rule-based GRPO on GSM8K."
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--init_model_path",
        default=None,
        help=(
            "Optional policy initialization checkpoint for train/debug. "
            "--model_name remains the KL reference model during training."
        ),
    )
    parser.add_argument("--train_offset", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=300)
    parser.add_argument("--eval_size", type=int, default=50)
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--context_batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--max_train_updates", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--reward_max_chars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/qwen25_15_grpo_gsm8k")
    parser.add_argument(
        "--eval_model_path",
        default=None,
        help="Model or checkpoint to load for --mode eval. Defaults to --model_name.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=0,
        help="Save policy checkpoint every N optimizer steps during training; 0 disables.",
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "train", "eval"],
        default="debug",
        help="debug samples rewards, train runs GRPO, eval scores held-out GSM8K.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save final policy checkpoint under output_dir/final_model.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_number(text):
    if text is None:
        return None
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("\\$", "$")
    text = text.replace("$", "")
    boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed.group(1).strip()
    text = re.sub(r"^[=:]+", "", text).strip()
    text = re.sub(r"[.;,%]+$", "", text).strip()
    return text


def number_to_fraction(text):
    text = normalize_number(text)
    if text is None or text == "":
        return None

    text = re.sub(r"\s+", "", text)
    latex_frac = re.fullmatch(
        r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}",
        text,
    )
    if latex_frac:
        text = f"{latex_frac.group(1)}/{latex_frac.group(2)}"
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(text))
    except (InvalidOperation, TypeError, ValueError, ZeroDivisionError):
        return None


def math_verify_equal(pred, gold):
    if not HAS_MATH_VERIFY:
        return None
    if pred is None or gold is None:
        return None

    pred_text = str(pred).strip()
    gold_text = str(gold).strip()
    if "\\" in pred_text and "$" not in pred_text:
        pred_text = f"${pred_text}$"
    if "\\" in gold_text and "$" not in gold_text:
        gold_text = f"${gold_text}$"

    try:
        gold_parsed = math_parse(
            gold_text,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
            extraction_mode="first_match",
        )
        pred_parsed = math_parse(
            pred_text,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            extraction_mode="first_match",
        )
        if len(gold_parsed) == 0 or len(pred_parsed) == 0:
            return None
        return bool(math_verify(gold_parsed, pred_parsed))
    except Exception:
        return None


def numbers_equal(left, right):
    left_value = number_to_fraction(left)
    right_value = number_to_fraction(right)
    if left_value is None or right_value is None:
        verified = math_verify_equal(left, right)
        if verified is not None:
            return verified
        return normalize_number(left) == normalize_number(right)
    if left_value == right_value:
        return True

    diff = abs(left_value - right_value)
    scale = max(abs(left_value), abs(right_value), Fraction(1, 1))
    tolerance = max(Fraction(1, 1_000_000), scale * Fraction(1, 1_000_000))
    if diff <= tolerance:
        return True

    verified = math_verify_equal(left, right)
    if verified is not None:
        return verified
    return False


def extract_gsm8k_answer(answer_text):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_text)
    if not match:
        return None
    return normalize_number(match.group(1))


NUMBER_PATTERN = (
    r"-?(?:\\?\$)?\d[\d,]*(?:\.\d+)?\s*/\s*-?(?:\\?\$)?\d[\d,]*(?:\.\d+)?"
    r"|-?(?:\\?\$)?\d[\d,]*(?:\.\d+)?"
)


def extract_last_boxed_content(text):
    marker = r"\boxed{"
    start = text.rfind(marker)
    while start != -1:
        content_start = start + len(marker)
        depth = 1
        i = content_start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return start, text[content_start:i].strip(), i + 1
            i += 1
        start = text.rfind(marker, 0, start)
    return None


def final_answer_matches(text):
    stripped = text.strip()
    matches = []

    answer_patterns = [
        ("answer_line", rf"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*({NUMBER_PATTERN})(?:\s*(?:dollars?|hours?|minutes?|meters?|cups?|bolts?|eggs?|feet|miles|geckos|students|cleaners|pastries|donuts|cupcakes|cheesecakes))?\s*[\].)]*\s*$"),
        ("final_answer_sentence", rf"(?:^|\n).*?(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*({NUMBER_PATTERN})(?:\s*(?:dollars?|hours?|minutes?|meters?|cups?|bolts?|eggs?|feet|miles|geckos|students|cleaners|pastries|donuts|cupcakes|cheesecakes))?\s*[\].)]*\s*$"),
    ]
    for kind, pattern in answer_patterns:
        for match in re.finditer(pattern, stripped, flags=re.IGNORECASE):
            matches.append((match.start(), kind, normalize_number(match.group(1))))

    # Treat a boxed expression in the final part of the completion as a final
    # answer, including nested LaTeX like \boxed{\frac{5}{6}}.
    tail_start = max(0, len(stripped) - 500)
    tail = stripped[tail_start:]
    boxed = extract_last_boxed_content(tail)
    if boxed is not None:
        start, content, end = boxed
        trailing = tail[end:].strip()
        if re.fullmatch(r"[\].,;:)]*", trailing):
            matches.append((tail_start + start, "boxed", normalize_number(content)))

    return sorted(matches, key=lambda item: item[0])


def extract_model_answer(text):
    final_matches = final_answer_matches(text)
    if final_matches:
        return final_matches[-1][2]

    answer_matches = re.findall(
        rf"(?:Final\s+)?Answer\s*:\s*({NUMBER_PATTERN})",
        text,
        flags=re.IGNORECASE,
    )
    if answer_matches:
        return normalize_number(answer_matches[-1])

    boxed = extract_last_boxed_content(text)
    if boxed is not None:
        return normalize_number(boxed[1])

    number_matches = re.findall(NUMBER_PATTERN, text)
    if number_matches:
        return normalize_number(number_matches[-1])

    return None


def final_answer_format(text):
    final_matches = final_answer_matches(text)
    if final_matches:
        return final_matches[-1][1]
    return None


def score_completion_components(
    completion,
    gold_answer,
    max_chars=1200,
    truncated_without_eos=False,
):
    pred = extract_model_answer(completion)
    gold = normalize_number(gold_answer)

    answer_format = final_answer_format(completion)
    has_answer_tag = bool(
        re.search(r"(?:Final\s+)?Answer\s*:", completion, flags=re.IGNORECASE)
        or re.search(r"\\boxed\{", completion)
        or re.search(
            r"\b(?:the\s+)?(?:final\s+)?answer\s+is\b",
            completion,
            flags=re.IGNORECASE,
        )
    )
    ends_with_answer = answer_format is not None
    is_parseable = pred is not None
    is_correct = is_parseable and numbers_equal(pred, gold)

    reward = 0.0
    correct_reward = 0.0
    format_reward = 0.0
    parseable_reward = 0.0
    wrong_answer_penalty = 0.0
    invalid_penalty = 0.0
    length_penalty = 0.0
    truncation_penalty = 0.0

    if is_correct:
        correct_reward = 1.0
        reward += correct_reward
    if ends_with_answer:
        format_reward = 0.2
        reward += format_reward
    if is_parseable:
        parseable_reward = 0.1
        reward += parseable_reward
    else:
        invalid_penalty = -0.5
        reward += invalid_penalty
    if is_parseable and not is_correct:
        wrong_answer_penalty = -0.4
        reward += wrong_answer_penalty
    if len(completion) > max_chars:
        length_penalty = -0.2
        reward += length_penalty
    if truncated_without_eos:
        truncation_penalty = -0.3
        reward += truncation_penalty

    return {
        "reward": float(reward),
        "pred": pred,
        "gold": gold,
        "pred_numeric": str(number_to_fraction(pred)) if pred is not None and number_to_fraction(pred) is not None else None,
        "gold_numeric": str(number_to_fraction(gold)) if gold is not None and number_to_fraction(gold) is not None else None,
        "has_answer_tag": has_answer_tag,
        "ends_with_answer": ends_with_answer,
        "final_answer_format": answer_format,
        "is_parseable": is_parseable,
        "is_correct": is_correct,
        "completion_chars": len(completion),
        "max_chars": max_chars,
        "length_penalty_applied": length_penalty != 0.0,
        "math_verify_available": HAS_MATH_VERIFY,
        "reward_correct": correct_reward,
        "reward_format": format_reward,
        "reward_parseable": parseable_reward,
        "reward_wrong_answer": wrong_answer_penalty,
        "reward_invalid": invalid_penalty,
        "reward_length": length_penalty,
        "reward_truncation": truncation_penalty,
    }


def score_completion(completion, gold_answer, max_chars=1200):
    return score_completion_components(completion, gold_answer, max_chars)["reward"]


def make_prompt(question):
    return (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer on its own final line as Answer: <number>.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Solve this grade-school math problem. Keep the reasoning concise. "
        "The final line must contain only: Answer: <number>\n\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_gsm8k_data(tokenizer, train_offset, train_size, eval_size, max_prompt_len):
    dataset = load_dataset("openai/gsm8k", "main")
    if train_offset < 0:
        raise ValueError("train_offset must be non-negative")
    train_end = train_offset + train_size
    if train_end > len(dataset["train"]):
        raise ValueError(
            f"Requested train slice [{train_offset}:{train_end}], "
            f"but GSM8K train split only has {len(dataset['train'])} rows"
        )

    train_data = dataset["train"].select(range(train_offset, train_end))
    eval_data = dataset["test"].select(range(eval_size))

    def format_gsm8k(example):
        return {
            "prompt": make_prompt(example["question"]),
            "gold_answer": extract_gsm8k_answer(example["answer"]),
        }

    def add_prompt_len(example):
        return {"prompt_len": len(tokenizer(example["prompt"])["input_ids"])}

    train_data = train_data.map(format_gsm8k).map(add_prompt_len)
    eval_data = eval_data.map(format_gsm8k).map(add_prompt_len)
    train_data = train_data.filter(lambda x: x["prompt_len"] <= max_prompt_len)
    eval_data = eval_data.filter(lambda x: x["prompt_len"] <= max_prompt_len)
    train_data = train_data.remove_columns(["prompt_len"])
    eval_data = eval_data.remove_columns(["prompt_len"])
    return train_data, eval_data


def build_dataloader(dataset, tokenizer, device, batch_size, shuffle):
    def collate_batch(batch):
        prompts = [item["prompt"] for item in batch]
        gold_answers = [item["gold_answer"] for item in batch]
        tensors = tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)
        tensors["gold_answer"] = gold_answers
        tensors["prompt_text"] = prompts
        return tensors

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )


def generate_token_by_policy(
    context,
    model,
    tokenizer,
    max_seq_len,
    num_generations=1,
    temperature=0.5,
    top_p=0.9,
    do_sample=True,
):
    if not do_sample and num_generations != 1:
        raise ValueError("num_generations must be 1 when do_sample=False")

    input_ids = context["input_ids"]
    attention_mask = context["attention_mask"]

    prompt_len = input_ids.shape[1]
    max_new_tokens = max_seq_len - prompt_len
    if max_new_tokens <= 0:
        raise ValueError(
            f"Prompt length {prompt_len} already reaches max_seq_len={max_seq_len}"
        )

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_generations,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.inference_mode():
        completion_ids = model.generate(**generate_kwargs)

    repeated_prompt_mask = attention_mask.repeat_interleave(num_generations, dim=0)
    completion_mask = torch.zeros_like(completion_ids)
    completion_mask[:, :prompt_len] = repeated_prompt_mask

    generated_ids = completion_ids[:, prompt_len:]
    generated_mask = torch.ones_like(generated_ids)

    if tokenizer.eos_token_id is not None and generated_ids.numel() > 0:
        eos_hits = generated_ids.eq(tokenizer.eos_token_id)
        has_eos = eos_hits.any(dim=1)
        if has_eos.any():
            first_eos = eos_hits.int().argmax(dim=1)
            positions = torch.arange(
                generated_ids.shape[1], device=generated_ids.device
            )
            generated_mask[has_eos] = (
                positions.unsqueeze(0) <= first_eos[has_eos].unsqueeze(1)
            ).long()

    completion_mask[:, prompt_len:] = generated_mask
    return completion_ids, completion_mask


def compute_token_logprobs(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids[:, :-1],
        attention_mask=attention_mask[:, :-1],
    )
    logits = outputs.logits.float()
    target_ids = input_ids[:, 1:]
    return -F.cross_entropy(
        logits.transpose(1, 2),
        target_ids,
        reduction="none",
    )


def generation_metadata(
    gen_iids,
    gen_mask,
    tokenizer,
    input_seq_len,
    max_seq_len,
    group_size,
):
    metadata = []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    for row_idx in range(gen_iids.shape[0]):
        row_ids = gen_iids[row_idx]
        row_mask = gen_mask[row_idx].bool()
        total_token_count = int(row_mask.sum().item())
        prompt_token_count = min(input_seq_len, total_token_count)
        generated_token_count = max(total_token_count - input_seq_len, 0)

        generated_ids_all = row_ids[input_seq_len:]
        generated_mask_all = row_mask[input_seq_len:]
        generated_token_ids = (
            generated_ids_all[generated_mask_all].detach().cpu().tolist()
        )

        ended_with_eos = bool(
            generated_token_ids
            and eos_token_id is not None
            and generated_token_ids[-1] == eos_token_id
        )
        reached_max_seq_len = total_token_count >= max_seq_len
        hit_max_seq_len = reached_max_seq_len and not ended_with_eos

        metadata.append(
            {
                "sample_idx": row_idx,
                "prompt_idx": row_idx // group_size,
                "generation_idx": row_idx % group_size,
                "group_size": group_size,
                "prompt_token_count": prompt_token_count,
                "generated_token_count": generated_token_count,
                "total_token_count": total_token_count,
                "tensor_seq_len": int(gen_iids.shape[1]),
                "pad_token_count": int((~row_mask).sum().item()),
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
                "ended_with_eos": ended_with_eos,
                "reached_max_seq_len": reached_max_seq_len,
                "hit_max_seq_len": hit_max_seq_len,
                "truncated_without_eos": hit_max_seq_len,
                "generated_token_ids": generated_token_ids,
            }
        )

    return metadata


def reward_details_for_outputs(
    texts,
    gold_answers,
    prompts,
    generation_details=None,
    group_size=None,
    epoch=0,
    step=0,
    max_chars=2000,
):
    details = []
    rewards = []
    for sample_idx, (text, gold, prompt) in enumerate(zip(texts, gold_answers, prompts)):
        generation = generation_details[sample_idx] if generation_details is not None else {}
        components = score_completion_components(
            text,
            gold,
            max_chars=max_chars,
            truncated_without_eos=generation.get("truncated_without_eos", False),
        )
        reward = components["reward"]
        rewards.append(reward)

        item = {
            "epoch": epoch,
            "step": step,
            "sample_idx": sample_idx,
            "prompt_idx": sample_idx // group_size if group_size else None,
            "generation_idx": sample_idx % group_size if group_size else None,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "gold": components["gold"],
            "pred": components["pred"],
            "reward": reward,
            "reward_components": {
                "correct": components["reward_correct"],
                "format": components["reward_format"],
                "parseable": components["reward_parseable"],
                "wrong_answer": components["reward_wrong_answer"],
                "invalid": components["reward_invalid"],
                "length": components["reward_length"],
                "truncation": components["reward_truncation"],
            },
            "has_answer_tag": components["has_answer_tag"],
            "ends_with_answer": components["ends_with_answer"],
            "format_exact_final_answer": components["ends_with_answer"],
            "final_answer_format": components["final_answer_format"],
            "is_parseable": components["is_parseable"],
            "is_correct": components["is_correct"],
            "length_penalty_applied": components["length_penalty_applied"],
            "math_verify_available": components["math_verify_available"],
            "reward_max_chars": components["max_chars"],
            "completion_chars": components["completion_chars"],
            "completion": text,
        }

        if generation_details is not None:
            item["generation"] = generation

        details.append(item)
    return rewards, details


def summarize_details(details):
    num_samples = len(details)
    if num_samples == 0:
        return {
            "num_samples": 0,
            "correct_count": 0,
            "correct_rate": 0.0,
            "answer_tag_count": 0,
            "answer_tag_rate": 0.0,
            "ends_with_answer_count": 0,
            "ends_with_answer_rate": 0.0,
            "parseable_count": 0,
            "parseable_rate": 0.0,
            "length_penalty_count": 0,
            "eos_count": 0,
            "truncated_count": 0,
            "avg_reward": 0.0,
            "avg_completion_chars": 0.0,
            "avg_generated_tokens": 0.0,
        }

    correct_count = sum(item["is_correct"] for item in details)
    answer_tag_count = sum(item["has_answer_tag"] for item in details)
    ends_with_answer_count = sum(item["ends_with_answer"] for item in details)
    parseable_count = sum(item["is_parseable"] for item in details)
    length_penalty_count = sum(item["length_penalty_applied"] for item in details)
    eos_count = sum(
        item.get("generation", {}).get("ended_with_eos", False) for item in details
    )
    truncated_count = sum(
        item.get("generation", {}).get("truncated_without_eos", False)
        for item in details
    )
    avg_reward = sum(item["reward"] for item in details) / num_samples
    avg_completion_chars = sum(item["completion_chars"] for item in details) / num_samples
    avg_generated_tokens = (
        sum(
            item.get("generation", {}).get("generated_token_count", 0)
            for item in details
        )
        / num_samples
    )

    return {
        "num_samples": num_samples,
        "correct_count": correct_count,
        "correct_rate": correct_count / num_samples,
        "answer_tag_count": answer_tag_count,
        "answer_tag_rate": answer_tag_count / num_samples,
        "ends_with_answer_count": ends_with_answer_count,
        "ends_with_answer_rate": ends_with_answer_count / num_samples,
        "parseable_count": parseable_count,
        "parseable_rate": parseable_count / num_samples,
        "length_penalty_count": length_penalty_count,
        "eos_count": eos_count,
        "truncated_count": truncated_count,
        "avg_reward": avg_reward,
        "avg_completion_chars": avg_completion_chars,
        "avg_generated_tokens": avg_generated_tokens,
    }


def write_jsonl(path, items):
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_gpu_stats():
    stats = {}
    if not torch.cuda.is_available():
        return stats

    device_index = torch.cuda.current_device()
    stats.update(
        {
            "gpu_name": torch.cuda.get_device_name(device_index),
            "cuda_memory_allocated_mb": round(
                torch.cuda.memory_allocated(device_index) / 1024**2, 2
            ),
            "cuda_memory_reserved_mb": round(
                torch.cuda.memory_reserved(device_index) / 1024**2, 2
            ),
            "cuda_max_memory_allocated_mb": round(
                torch.cuda.max_memory_allocated(device_index) / 1024**2, 2
            ),
            "cuda_max_memory_reserved_mb": round(
                torch.cuda.max_memory_reserved(device_index) / 1024**2, 2
            ),
        }
    )

    query = (
        "temperature.gpu,utilization.gpu,memory.used,memory.total,"
        "power.draw,power.limit"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_gpu = result.stdout.strip().splitlines()[0]
        temp, util, mem_used, mem_total, power_draw, power_limit = [
            value.strip() for value in first_gpu.split(",")
        ]
        stats.update(
            {
                "nvidia_temp_c": float(temp),
                "nvidia_utilization_pct": float(util),
                "nvidia_memory_used_mb": float(mem_used),
                "nvidia_memory_total_mb": float(mem_total),
                "nvidia_power_draw_w": float(power_draw),
                "nvidia_power_limit_w": float(power_limit),
            }
        )
    except Exception as exc:
        stats["nvidia_smi_error"] = str(exc)

    return stats


def load_causal_lm(model_name, dtype, device):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(device)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)


def run_debug(args, model, tokenizer, dataloader, output_dir):
    started_at = time.time()
    batch = next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        gen_iids, gen_mask = generate_token_by_policy(
            batch,
            model,
            tokenizer,
            max_seq_len=args.max_seq_len,
            num_generations=args.group_size,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    input_seq_len = batch["input_ids"].shape[1]
    texts = tokenizer.batch_decode(gen_iids[:, input_seq_len:], skip_special_tokens=True)
    golds = [gold for gold in batch["gold_answer"] for _ in range(args.group_size)]
    prompts = [prompt for prompt in batch["prompt_text"] for _ in range(args.group_size)]
    gen_details = generation_metadata(
        gen_iids,
        gen_mask,
        tokenizer,
        input_seq_len,
        args.max_seq_len,
        args.group_size,
    )
    _, details = reward_details_for_outputs(
        texts,
        golds,
        prompts,
        generation_details=gen_details,
        group_size=args.group_size,
        max_chars=args.reward_max_chars,
    )

    debug_path = output_dir / "debug_samples.jsonl"
    write_jsonl(debug_path, details)
    write_jsonl(
        output_dir / "debug_metrics.jsonl",
        [
            {
                "mode": "debug",
                "elapsed_sec": round(time.time() - started_at, 3),
                "num_samples": len(details),
                "gpu": get_gpu_stats(),
            }
        ],
    )

    for item in details:
        print(f"sample: {item['sample_idx']}")
        print("prompt:")
        print(item["prompt"])
        print("gold:", item["gold"])
        print("pred:", item["pred"])
        print("reward:", item["reward"])
        print("reward components:", item["reward_components"])
        print("has Answer tag:", item["has_answer_tag"])
        print("is parseable:", item["is_parseable"])
        print("is correct:", item["is_correct"])
        print("generation:", item.get("generation", {}))
        print("completion chars:", item["completion_chars"])
        print("completion:")
        print(item["completion"][:1500])
        print("=" * 80)
    print(f"Wrote {debug_path}")


def evaluate(args, model, tokenizer, dataloader, output_dir, model_source):
    started_at = time.time()
    samples_path = output_dir / "eval_samples.jsonl"
    metrics_path = output_dir / "eval_metrics.json"
    samples_path.unlink(missing_ok=True)
    metrics_path.unlink(missing_ok=True)

    all_details = []
    model.eval()

    for step, batch in enumerate(dataloader):
        input_seq_len = batch["input_ids"].shape[1]
        with torch.no_grad():
            gen_iids, gen_mask = generate_token_by_policy(
                batch,
                model,
                tokenizer,
                max_seq_len=args.max_seq_len,
                num_generations=1,
                do_sample=False,
            )

        texts = tokenizer.batch_decode(gen_iids[:, input_seq_len:], skip_special_tokens=True)
        gen_details = generation_metadata(
            gen_iids,
            gen_mask,
            tokenizer,
            input_seq_len,
            args.max_seq_len,
            group_size=1,
        )
        _, details = reward_details_for_outputs(
            texts,
            batch["gold_answer"],
            batch["prompt_text"],
            generation_details=gen_details,
            group_size=1,
            step=step + 1,
            max_chars=args.reward_max_chars,
        )

        for local_idx, item in enumerate(details):
            item["eval_idx"] = len(all_details) + local_idx

        write_jsonl(samples_path, details)
        all_details.extend(details)

        if (step + 1) % 10 == 0:
            current = summarize_details(all_details)
            print(
                f"eval batch {step + 1}/{len(dataloader)} "
                f"correct {current['correct_count']}/{current['num_samples']} "
                f"parseable {current['parseable_count']}/{current['num_samples']} "
                f"trunc {current['truncated_count']}/{current['num_samples']}",
                flush=True,
            )

    metrics = summarize_details(all_details)
    metrics.update(
        {
            "mode": "eval",
            "model_source": str(model_source),
            "elapsed_sec": round(time.time() - started_at, 3),
            "max_seq_len": args.max_seq_len,
            "reward_max_chars": args.reward_max_chars,
            "temperature": None,
            "top_p": None,
            "gpu": get_gpu_stats(),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"eval done: correct {metrics['correct_count']}/{metrics['num_samples']} "
        f"parseable {metrics['parseable_count']}/{metrics['num_samples']} "
        f"answer_marker {metrics['ends_with_answer_count']}/{metrics['num_samples']} "
        f"trunc {metrics['truncated_count']}/{metrics['num_samples']} "
        f"avg_reward {metrics['avg_reward']:.4f}"
    )
    print(f"wrote {samples_path}")
    print(f"wrote {metrics_path}")


def train(args, pi_new, pi_ref, tokenizer, dataloader, output_dir, device):
    if (args.context_batch_size * args.group_size) % args.train_batch_size != 0:
        raise ValueError("train_batch_size must divide context_batch_size * group_size")

    num_train_update = min(len(dataloader), args.max_train_updates)
    optimizer = torch.optim.AdamW(
        params=pi_new.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def cosine_schedule(current_step, num_training_steps, num_warmup_steps=0):
        if current_step < num_warmup_steps:
            return 1.0
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_schedule(
            step,
            num_training_steps=args.num_epochs * num_train_update,
            num_warmup_steps=math.ceil(args.num_epochs * num_train_update * 0.1),
        ),
    )

    pi_ref.eval()
    for p in pi_ref.parameters():
        p.requires_grad = False

    reward_log = output_dir / "reward.log"
    debug_log = output_dir / "reward_debug.jsonl"
    metrics_log = output_dir / "train_metrics.jsonl"
    reward_log.unlink(missing_ok=True)
    debug_log.unlink(missing_ok=True)
    metrics_log.unlink(missing_ok=True)

    record_reward = []

    for epoch in range(args.num_epochs):
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            if step >= args.max_train_updates:
                break

            step_started_at = time.time()
            input_seq_len = batch["input_ids"].shape[1]
            gold_answers = batch["gold_answer"]

            pi_new.eval()
            with torch.no_grad():
                gen_iids, gen_mask = generate_token_by_policy(
                    batch,
                    pi_new,
                    tokenizer,
                    max_seq_len=args.max_seq_len,
                    num_generations=args.group_size,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                old_logprb = compute_token_logprobs(
                    pi_new,
                    gen_iids,
                    gen_mask,
                ).detach()

                gen_texts = tokenizer.batch_decode(
                    gen_iids[:, input_seq_len:],
                    skip_special_tokens=True,
                )
                repeated_gold_answers = [
                    gold for gold in gold_answers for _ in range(args.group_size)
                ]
                repeated_prompts = [
                    prompt for prompt in batch["prompt_text"] for _ in range(args.group_size)
                ]
                gen_details = generation_metadata(
                    gen_iids,
                    gen_mask,
                    tokenizer,
                    input_seq_len,
                    args.max_seq_len,
                    args.group_size,
                )
                rewards, details = reward_details_for_outputs(
                    gen_texts,
                    repeated_gold_answers,
                    repeated_prompts,
                    generation_details=gen_details,
                    group_size=args.group_size,
                    epoch=epoch + 1,
                    step=step + 1,
                    max_chars=args.reward_max_chars,
                )

                seq_rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                reward_current = seq_rewards.mean().item()
                record_reward.append(reward_current)

                reward_grouped = seq_rewards.view(-1, args.group_size)
                reward_mean = reward_grouped.mean(dim=1).repeat_interleave(
                    args.group_size
                )
                reward_std = reward_grouped.std(dim=1).repeat_interleave(
                    args.group_size
                )
                adv = torch.where(
                    reward_std == 0.0,
                    torch.zeros_like(seq_rewards),
                    (seq_rewards - reward_mean) / reward_std,
                )

                seq_len = gen_iids.shape[1]
                target_positions = torch.arange(1, seq_len, device=device)
                inf_mask = gen_mask[:, 1:] * (
                    target_positions >= input_seq_len
                ).long()

                valid_rows = inf_mask.sum(dim=1) > 0
                if not valid_rows.any():
                    continue

                gen_iids = gen_iids[valid_rows]
                gen_mask = gen_mask[valid_rows]
                old_logprb = old_logprb[valid_rows]
                inf_mask = inf_mask[valid_rows]
                adv = adv[valid_rows]

            pi_new.train()
            batch_size_actual = gen_iids.shape[0]
            total_loss = None

            for cur_batch_idx in range(0, batch_size_actual, args.train_batch_size):
                train_gen_iids = gen_iids[
                    cur_batch_idx : cur_batch_idx + args.train_batch_size
                ]
                train_gen_mask = gen_mask[
                    cur_batch_idx : cur_batch_idx + args.train_batch_size
                ]
                train_old_logprb = old_logprb[
                    cur_batch_idx : cur_batch_idx + args.train_batch_size
                ]
                train_inf_mask = inf_mask[
                    cur_batch_idx : cur_batch_idx + args.train_batch_size
                ]
                train_adv = adv[cur_batch_idx : cur_batch_idx + args.train_batch_size]

                logprb_new = compute_token_logprobs(
                    pi_new,
                    train_gen_iids,
                    train_gen_mask,
                )
                with torch.no_grad():
                    logprb_ref = compute_token_logprobs(
                        pi_ref,
                        train_gen_iids,
                        train_gen_mask,
                    )

                prb_ratio = torch.exp(logprb_new - train_old_logprb)
                prb_ratio_clipped = torch.clamp(
                    prb_ratio,
                    1.0 - args.epsilon,
                    1.0 + args.epsilon,
                )

                pg_loss1 = train_adv.unsqueeze(-1) * prb_ratio
                pg_loss2 = train_adv.unsqueeze(-1) * prb_ratio_clipped
                pg_loss = -torch.min(pg_loss1, pg_loss2)
                kl_loss = (
                    torch.exp(logprb_ref - logprb_new)
                    - (logprb_ref - logprb_new)
                    - 1
                )

                total_loss = pg_loss + args.beta * kl_loss
                total_loss = torch.masked_select(
                    total_loss,
                    train_inf_mask.bool(),
                ).mean()
                total_loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            with reward_log.open("a", encoding="utf-8") as f:
                f.write(f"{reward_current}\n")
            write_jsonl(debug_log, details)

            summary = summarize_details(details)
            loss_value = float(total_loss.item()) if total_loss is not None else float("nan")
            current_lr = scheduler.get_last_lr()[0]
            global_step = epoch * num_train_update + step + 1
            metrics = {
                "epoch": epoch + 1,
                "step": step + 1,
                "global_step": global_step,
                "num_train_update": num_train_update,
                "loss": loss_value,
                "reward_mean": reward_current,
                "input_seq_len": input_seq_len,
                "generated_seq_len": int(gen_iids.shape[1]),
                "learning_rate": current_lr,
                "elapsed_sec": round(time.time() - step_started_at, 3),
                "gpu": get_gpu_stats(),
            }
            metrics.update(summary)
            write_jsonl(metrics_log, [metrics])
            print(
                f"epoch {epoch + 1} step {step + 1}/{num_train_update} "
                f"reward {reward_current:.4f} loss {loss_value:.4f} "
                f"correct {summary['correct_count']}/{summary['num_samples']} "
                f"answer_marker {summary['ends_with_answer_count']}/{summary['num_samples']} "
                f"trunc {summary['truncated_count']}/{summary['num_samples']} "
                f"mem {metrics['gpu'].get('nvidia_memory_used_mb', 'na')}MB "
                f"temp {metrics['gpu'].get('nvidia_temp_c', 'na')}C",
                flush=True,
            )

            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                checkpoint_dir = output_dir / f"checkpoint_step_{global_step:06d}"
                pi_new.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"saved checkpoint to {checkpoint_dir}", flush=True)

    print(f"avg reward: {sum(record_reward) / len(record_reward):.6f}")

    if args.save_model:
        final_dir = output_dir / "final_model"
        pi_new.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"saved model to {final_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    if HAS_MATH_VERIFY:
        print("math_verify: available")
    else:
        print(f"math_verify: unavailable ({MATH_VERIFY_IMPORT_ERROR})")

    if args.mode == "eval":
        model_source = args.eval_model_path if args.eval_model_path else args.model_name
        tokenizer_source = model_source
        print(f"eval model: {model_source}")
    else:
        policy_source = args.init_model_path if args.init_model_path else args.model_name
        reference_source = args.model_name
        tokenizer_source = policy_source
        print(f"policy init model: {policy_source}")
        if args.mode == "train":
            print(f"KL reference model: {reference_source}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data, eval_data = load_gsm8k_data(
        tokenizer,
        args.train_offset,
        args.train_size,
        args.eval_size,
        args.max_prompt_len,
    )
    print(f"train rows: {len(train_data)} eval rows: {len(eval_data)}")

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if args.mode == "eval":
        eval_loader = build_dataloader(
            eval_data,
            tokenizer,
            device,
            args.context_batch_size,
            shuffle=False,
        )
        model = load_causal_lm(model_source, model_dtype, device)
        model.config.pad_token_id = tokenizer.pad_token_id
        evaluate(args, model, tokenizer, eval_loader, output_dir, model_source)
        return

    train_loader = build_dataloader(
        train_data,
        tokenizer,
        device,
        args.context_batch_size,
        shuffle=args.mode == "train",
    )

    pi_new = load_causal_lm(policy_source, model_dtype, device)
    pi_new.config.pad_token_id = tokenizer.pad_token_id

    if args.mode == "debug":
        run_debug(args, pi_new, tokenizer, train_loader, output_dir)
        return

    pi_ref = load_causal_lm(reference_source, model_dtype, device)
    pi_ref.config.pad_token_id = tokenizer.pad_token_id

    train(args, pi_new, pi_ref, tokenizer, train_loader, output_dir, device)


if __name__ == "__main__":
    main()
