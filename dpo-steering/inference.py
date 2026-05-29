import argparse
import os
from typing import List, Optional, Tuple
import sys
import pandas as pd
import torch

from inference import _build_custom_prompt, _load_model_and_tokenizer


def _generate_batch(
    model,
    tokenizer,
    device: str,
    prompt_texts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    results: List[str] = []
    for i, in_len in enumerate(input_lengths):
        gen_tokens = out[i, int(in_len) :]
        results.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
    return results


def _collect_prompts(
    df: pd.DataFrame,
    row_indices: List[int],
    question_col: str,
    concept_col: str,
    tokenizer,
    conversation: bool = True,
) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    for row_idx in row_indices:
        question = "" if pd.isna(df.at[row_idx, question_col]) else str(df.at[row_idx, question_col])
        concept = "" if pd.isna(df.at[row_idx, concept_col]) else str(df.at[row_idx, concept_col])
        if conversation:
            msgs = _build_custom_prompt(question, concept)
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = f"question: {question}\nsteering concept: {concept}\nI think "
        items.append((row_idx, prompt_text))
    return items


def _infer_file_format(path: str, forced: Optional[str]) -> str:
    if forced is not None:
        return forced
    lower = path.lower()
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".parquet"):
        return "parquet"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".tsv"):
        return "tsv"
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return "excel"
    raise SystemExit(f"Could not infer file format from {path!r}; pass --data_format.")


def _read_dataframe(path: str, data_format: Optional[str]) -> pd.DataFrame:
    fmt = _infer_file_format(path, data_format)
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t")
    if fmt == "jsonl":
        return pd.read_json(path, lines=True)
    if fmt == "json":
        return pd.read_json(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "excel":
        return pd.read_excel(path)
    raise SystemExit(f"Unsupported data format: {fmt!r}")


def _write_dataframe(df: pd.DataFrame, path: str, data_format: Optional[str]) -> None:
    fmt = _infer_file_format(path, data_format)
    if fmt == "csv":
        df.to_csv(path, index=False)
        return
    if fmt == "tsv":
        df.to_csv(path, sep="\t", index=False)
        return
    if fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        return
    if fmt == "json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
        return
    if fmt == "parquet":
        df.to_parquet(path, index=False)
        return
    if fmt == "excel":
        df.to_excel(path, index=False)
        return
    raise SystemExit(f"Unsupported data format: {fmt!r}")


def _is_filled(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return str(value).strip() != ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch inference from a pandas file with separate concept and question columns"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Input file (csv/tsv/json/jsonl/parquet/xlsx); outputs are written back to this path",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default=None,
        choices=[None, "csv", "tsv", "json", "jsonl", "parquet", "excel"],
        help="Force file format instead of inferring from extension",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint directory (full model or LoRA adapter)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Base model id/path (used if checkpoint is adapter)")

    parser.add_argument("--concept_col", type=str, required=True, help="Column containing steering concepts")
    parser.add_argument("--question_col", type=str, required=True, help="Column containing questions")
    parser.add_argument(
        "--output_col",
        type=str,
        default="model_output",
        help="Column to write generated text into (created if missing)",
    )

    parser.add_argument("--start", type=int, default=0, help="Start row index (inclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rows where output_col already has a non-empty value",
    )
    parser.add_argument("--save_every", type=int, default=50, help="Write results back to disk every N rows (0 = only at end)")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of prompts to generate per forward pass")

    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--conversation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply chat template to prompts (default: true; use --no-conversation for raw text)",
    )

    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/srv/scratch/z5517269/HF_HOME")

    df = _read_dataframe(args.data_path, args.data_format)
    for col in (args.concept_col, args.question_col):
        if col not in df.columns:
            raise SystemExit(f"Column {col!r} not found. Available columns: {list(df.columns)}")

    if args.output_col not in df.columns:
        df[args.output_col] = None
    df[args.output_col] = df[args.output_col].astype(object)

    end = len(df) if args.limit is None else min(len(df), args.start + args.limit)
    if args.start < 0 or args.start >= len(df):
        raise SystemExit(f"--start must be in [0, {len(df) - 1}]")
    if end <= args.start:
        raise SystemExit(f"No rows to process: start={args.start}, end={end}")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")

    model, tokenizer, device = _load_model_and_tokenizer(args.checkpoint_path, args.model)
    model.eval()

    row_indices = [
        row_idx
        for row_idx in range(args.start, end)
        if not (args.skip_existing and _is_filled(df.at[row_idx, args.output_col]))
    ]
    if not row_indices:
        print("No rows to process (all skipped or empty range).")
        return

    processed = 0
    next_save_at = args.save_every if args.save_every > 0 else None
    for batch_start in range(0, len(row_indices), args.batch_size):
        batch_row_indices = row_indices[batch_start : batch_start + args.batch_size]
        batch_items = _collect_prompts(
            df, batch_row_indices, args.question_col, args.concept_col, tokenizer, args.conversation
        )
        prompt_texts = [prompt for _, prompt in batch_items]
        gen_texts = _generate_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_texts=prompt_texts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        for (row_idx, _), gen_text in zip(batch_items, gen_texts):
            df.at[row_idx, args.output_col] = f"I think {gen_text}"
            processed += 1
            preview = gen_text[:100] + ("..." if len(gen_text) > 100 else "")
            print(f"[{processed}/{len(row_indices)}] row={row_idx} generated {len(gen_text)} chars, {preview}")

        if next_save_at is not None and processed >= next_save_at:
            _write_dataframe(df, args.data_path, args.data_format)
            print(f"Checkpoint saved to {args.data_path}")
            while next_save_at is not None and next_save_at <= processed:
                next_save_at += args.save_every

    _write_dataframe(df, args.data_path, args.data_format)
    print(f"Done. Processed {processed} row(s). Wrote outputs to column {args.output_col!r} in {args.data_path}")


if __name__ == "__main__":
    main()
