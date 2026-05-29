import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, TextIO

import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer


Message = Dict[str, str]


def _config_for_wandb(ns: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(ns).items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


class _WandbTeeIO:
    """Writes to the original stream and mirrors line-oriented output to wandb."""

    def __init__(self, underlying: TextIO, stream_label: str, log_line: Callable[[str, str], None]) -> None:
        self._underlying = underlying
        self._stream_label = stream_label
        self._partial = ""
        self._log_line = log_line

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        self._underlying.write(s)
        self._partial += s
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._log_line(self._stream_label, line)
        return len(s)

    def flush(self) -> None:
        self._underlying.flush()

    def isatty(self) -> bool:
        return getattr(self._underlying, "isatty", lambda: False)()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


def _install_print_capture(
    saved_stdout: TextIO,
    saved_stderr: TextIO,
) -> tuple[_WandbTeeIO, _WandbTeeIO, Callable[[_WandbTeeIO, _WandbTeeIO], None]]:
    console_step_holder: List[int] = [0]

    def _log_line(stream_label: str, line: str) -> None:
        if wandb.run is None:
            return
        console_step_holder[0] += 1
        step = console_step_holder[0]
        text = f"[{stream_label}] {line}"
        if len(text) > 10000:
            text = text[:9997] + "..."
        try:
            wandb.log({"console_log_step": step, "console_text": text})
        except Exception:
            pass

    def _flush_partials(out_tee: _WandbTeeIO, err_tee: _WandbTeeIO) -> None:
        for tee in (out_tee, err_tee):
            rest = tee._partial
            if not rest:
                continue
            tee._partial = ""
            _log_line(tee._stream_label, rest)

    out_tee = _WandbTeeIO(saved_stdout, "stdout", _log_line)
    err_tee = _WandbTeeIO(saved_stderr, "stderr", _log_line)
    sys.stdout = out_tee
    sys.stderr = err_tee
    return out_tee, err_tee, _flush_partials


def _as_messages(x: Any) -> List[Message]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[Message] = []
        for m in x:
            if isinstance(m, dict):
                out.append({"role": str(m.get("role", "")), "content": str(m.get("content", ""))})
            elif isinstance(m, str):
                out.append({"role": "user", "content": m})
        return out
    if isinstance(x, dict):
        return [{"role": str(x.get("role", "user")), "content": str(x.get("content", ""))}]
    if isinstance(x, str):
        return [{"role": "user", "content": x}]
    return [{"role": "user", "content": str(x)}]


def _assistant_last_content(x: Any) -> str:
    msgs = _as_messages(x)
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return msgs[-1].get("content", "") if msgs else ""


def _infer_local_format(path: str, forced: Optional[str]) -> str:
    if forced is not None:
        return forced
    if path.endswith(".json") or path.endswith(".jsonl"):
        return "json"
    if path.endswith(".parquet"):
        return "parquet"
    if path.endswith(".csv"):
        return "csv"
    raise SystemExit(f"Could not infer --data_format from {path!r}; pass --data_format.")


def _load_local_file(path: str, data_format: Optional[str]) -> Dataset:
    fmt = _infer_local_format(path, data_format)
    return load_dataset(fmt, data_files=path, split="train")


def _prepare_dpo_columns(
    raw: Dataset,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> Dataset:
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    keep_cols = [c for c in [args.prompt_col, args.chosen_col, args.rejected_col] if c]
    remove_cols = [c for c in raw.column_names if c not in keep_cols]

    def _map_ex(ex: Dict[str, Any]) -> Dict[str, str]:
        prompt_raw = ex.get(args.prompt_col)
        chosen_raw = ex.get(args.chosen_col)
        rejected_raw = ex.get(args.rejected_col)

        # conversational = args.conversational or isinstance(prompt_raw, list) or isinstance(chosen_raw, list) or isinstance(rejected_raw, list)
        conversational = args.conversational
        if conversational:
            prompt_msgs = _as_messages(prompt_raw)
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            return {
                "prompt": prompt_text,
                "chosen": _assistant_last_content(chosen_raw),
                "rejected": _assistant_last_content(rejected_raw),
            }
        else:
            return {
                "prompt": prompt_raw[0]['content'],
                "chosen": chosen_raw[0]['content'],
                "rejected": rejected_raw[0]['content'],
            }
        # print(str(prompt_raw).strip())
        # sys.exit()
        # return {
        #     "prompt": "" if prompt_raw is None else str(prompt_raw).strip(),
        #     "chosen": "" if chosen_raw is None else str(chosen_raw).strip(),
        #     "rejected": "" if rejected_raw is None else str(rejected_raw).strip(),
        # }

    return raw.map(_map_ex, remove_columns=remove_cols)


def _run_training(args: argparse.Namespace) -> None:
    train_path = args.train_data_path or args.data_path
    if args.dataset is None and train_path is None:
        raise SystemExit("Provide --dataset (HF hub) or --train_data_path / --data_path (local train file).")

    if args.dataset is not None and train_path is not None:
        raise SystemExit("Use either --dataset or local --train_data_path/--data_path, not both.")

    eval_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None

    if args.dataset is not None:
        multi = (
            args.dataset_train_split is not None
            or args.dataset_valid_split is not None
            or args.dataset_test_split is not None
        )
        if multi:
            ds_dict = load_dataset(args.dataset)
            train_key = args.dataset_train_split or "train"
            if train_key not in ds_dict:
                raise SystemExit(f"Split {train_key!r} not in dataset keys: {list(ds_dict.keys())}")
            train_dataset = ds_dict[train_key]
            if args.dataset_valid_split:
                if args.dataset_valid_split not in ds_dict:
                    raise SystemExit(f"Split {args.dataset_valid_split!r} not in dataset keys: {list(ds_dict.keys())}")
                eval_dataset = ds_dict[args.dataset_valid_split]
            if args.dataset_test_split:
                if args.dataset_test_split not in ds_dict:
                    raise SystemExit(f"Split {args.dataset_test_split!r} not in dataset keys: {list(ds_dict.keys())}")
                test_dataset = ds_dict[args.dataset_test_split]
        else:
            train_dataset = load_dataset(args.dataset, split=args.dataset_split)
    else:
        assert train_path is not None
        train_dataset = _load_local_file(train_path, args.data_format)
        if args.valid_data_path:
            eval_dataset = _load_local_file(args.valid_data_path, args.data_format)
        if args.test_data_path:
            test_dataset = _load_local_file(args.test_data_path, args.data_format)

    # TRL DPO expects left padding (doc requirement).
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        # Llama 3+ ships a dedicated fine-tuning pad token; prefer it over reusing
        # eos_token so EOS isn't masked out as padding during DPO loss.
        vocab = tokenizer.get_vocab()
        if "<|finetune_right_pad_id|>" in vocab:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = _prepare_dpo_columns(train_dataset, tokenizer, args)
    if eval_dataset is not None:
        eval_dataset = _prepare_dpo_columns(eval_dataset, tokenizer, args)
    if test_dataset is not None:
        test_dataset = _prepare_dpo_columns(test_dataset, tokenizer, args)
    # print(len(train_dataset))
    # print(len(eval_dataset))
    # print(len(test_dataset))
    # sys.exit()
    # Use bf16 if available, otherwise fp16.
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = torch.cuda.is_available() and not bf16

    eval_steps = args.eval_steps if args.eval_steps is not None else args.save_steps
    eval_strategy = "steps" if eval_dataset is not None else "no"

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_dataset is not None else None,
        report_to=["wandb"] if not args.no_wandb else [],
        run_name=args.wandb_run_name,
        seed=args.seed,
        max_length=args.max_length,
        bf16=bf16,
        fp16=fp16,
        model_init_kwargs={"dtype": torch.bfloat16 if bf16 else torch.float16},
        remove_unused_columns=False, 
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = DPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    if test_dataset is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print(test_metrics)
    trainer.save_model(args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple DPO training with TRL + Qwen3-0.6B")

    # Data
    parser.add_argument("--dataset", type=str, default=None, help="HF dataset name (e.g. trl-lib/ultrafeedback_binarized)")
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="HF split when loading a single split (ignored if --dataset_*_split selects a hub DatasetDict)",
    )
    parser.add_argument(
        "--dataset_train_split",
        type=str,
        default=None,
        help="HF DatasetDict key for training (enables multi-split hub load with --dataset_valid_split / --dataset_test_split)",
    )
    parser.add_argument("--dataset_valid_split", type=str, default=None, help="HF DatasetDict key for validation / eval during training")
    parser.add_argument("--dataset_test_split", type=str, default=None, help="HF DatasetDict key for held-out test (evaluated after training)")
    parser.add_argument("--data_path", type=str, default=None, help="Local training file (json/jsonl/parquet/csv); alias for --train_data_path")
    parser.add_argument("--train_data_path", type=str, default=None, help="Local training file")
    parser.add_argument("--valid_data_path", type=str, default=None, help="Local validation file (same schema as train)")
    parser.add_argument("--test_data_path", type=str, default=None, help="Local test file (same schema as train); evaluated after training")
    parser.add_argument("--data_format", type=str, default=None, choices=[None, "json", "csv", "parquet"], help="Force local loader format")
    parser.add_argument("--conversational", action="store_true", help="Treat columns as chat messages (role/content dicts) or strings")
    parser.add_argument("--prompt_col", type=str, default="prompt", help="Prompt column name")
    parser.add_argument("--chosen_col", type=str, default="chosen", help="Chosen column name")
    parser.add_argument("--rejected_col", type=str, default="rejected", help="Rejected column name")

    # Model / training
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Base model id or path")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5517269/out", help="Output dir")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Good default for LoRA")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (preference strength)")
    parser.add_argument("--max_length", type=int, default=256, help="Max total sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Max prompt length (if explicit prompt)")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--eval_steps", type=int, help="Run validation eval every N steps (default: same as --save_steps)")
    parser.add_argument("--seed", type=int, default=42)

    # W&B
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "RL-Steering"))
    parser.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME", "dpo-qwen3-0.6b"))
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases (no metrics or console capture)")

    # LoRA (small model still benefits; keeps VRAM low)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    saved_out, saved_err = sys.stdout, sys.stderr
    flush_partials: Optional[Callable[[_WandbTeeIO, _WandbTeeIO], None]] = None
    out_tee: Optional[_WandbTeeIO] = None
    err_tee: Optional[_WandbTeeIO] = None

    os.environ.setdefault("HF_HOME", "/srv/scratch/z5517269/HF_HOME")
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=_config_for_wandb(args))
        wandb.define_metric("console_log_step")
        wandb.define_metric("console_text", step_metric="console_log_step")
        out_tee, err_tee, flush_partials = _install_print_capture(saved_out, saved_err)

    try:
        _run_training(args)
    finally:
        if flush_partials is not None and out_tee is not None and err_tee is not None:
            flush_partials(out_tee, err_tee)
        sys.stdout, sys.stderr = saved_out, saved_err
        if not args.no_wandb:
            run = wandb.run
            if run is not None:
                try:
                    wandb.finish()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
