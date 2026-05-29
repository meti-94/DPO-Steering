# RL Steering — DPO training & AxBench inference

Two CLI scripts for preference-tuning a small causal LM (default: `Qwen/Qwen3-0.6B`) and running batch inference on steering-style prompts (question + concept).

## Environment

Install dependencies from `requirements.txt` (see that file when available).

---

## `DPO.py` — Direct Preference Optimization (LoRA)

Runs [TRL](https://github.com/huggingface/trl) `DPOTrainer` with LoRA on attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Checkpoints are written to `--output_dir` (adapter weights usable by `inference.py`).

### Data

Each training example needs three fields (names configurable):

| Column (default) | Meaning |
|------------------|---------|
| `prompt` | User/context turn(s) |
| `chosen` | Preferred assistant reply |
| `rejected` | Dispreferred assistant reply |

**Source (pick one):**

- **Hugging Face Hub:** `--dataset <name>` with optional split keys (`--dataset_train_split`, `--dataset_valid_split`, `--dataset_test_split`) or a single `--dataset_split`.
- **Local files:** `--train_data_path` or `--data_path` (`.json`, `.jsonl`, `.parquet`, `.csv`). Optional `--valid_data_path`, `--test_data_path`.

**Format:**

- With `--conversational`: columns are chat message lists (`role` / `content`). The script applies a chat template to the prompt and extracts the last assistant message from `chosen` / `rejected`.
- Without `--conversational`: expects list-of-dicts with a `content` field on the first message in each column.

### Example

```bash
python DPO.py \
  --train_data_path data/train.jsonl \
  --conversational \
  --model Qwen/Qwen3-0.6B \
  --output_dir /srv/scratch/z5517269/out/my-dpo-run \
  --num_train_epochs 1 \
  --beta 0.1
```

Hub example:

```bash
python DPO.py --dataset trl-lib/ultrafeedback_binarized --dataset_split train
```

### Important arguments

| Argument | Default | Notes |
|----------|---------|--------|
| `--model` | `Qwen/Qwen3-0.6B` | Base model ID or local path |
| `--output_dir` | `/srv/scratch/z5517269/out` | Saved LoRA adapter + trainer state |
| `--beta` | `0.1` | DPO temperature; higher = stronger preference signal |
| `--learning_rate` | `1e-5` | Tuned for LoRA |
| `--max_length` | `256` | Max total sequence length |
| `--per_device_train_batch_size` | `4` | Per-GPU batch size |
| `--gradient_accumulation_steps` | `8` | Effective batch = batch × accum × GPUs |
| `--num_train_epochs` | `1` | |
| `--lora_r` / `--lora_alpha` | `32` / `16` | LoRA rank and scaling |
| `--prompt_col` / `--chosen_col` / `--rejected_col` | `prompt` / `chosen` / `rejected` | Column names in your file |
| `--conversational` | off | Enable for chat-style rows |
| `--save_steps` / `--eval_steps` | `400` / same as save | Eval runs only if a validation set is provided |
| `--wandb_project` / `--wandb_run_name` | `RL-Steering` / `dpo-qwen3-0.6b` | Metrics + mirrored stdout/stderr |
| `--no_wandb` | off | Disable Weights & Biases |

`HF_HOME` defaults to `/srv/scratch/z5517269/HF_HOME` if unset.

---

## `inference.py` — Batch inference on tabular data

Reads a table (CSV, TSV, JSON, JSONL, Parquet, Excel), builds prompts from a **question** column and a **steering concept** column, generates completions, and **writes results back to the same file** in `--output_col` (default: `model_output`). Each completion is stored as `I think {generated text}`.

Uses `inference._build_custom_prompt` and `inference._load_model_and_tokenizer` (full checkpoint or PEFT adapter from `DPO.py`).

### Input columns

- `--question_col` — question text (required)
- `--concept_col` — steering concept text (required)
- `--output_col` — column to fill (default: `model_output`)

### Example

```bash
python inference.py \
  --data_path data/axbench_eval.csv \
  --checkpoint_path /srv/scratch/z5517269/out/my-dpo-run \
  --model Qwen/Qwen3-0.6B \
  --question_col question \
  --concept_col concept \
  --batch_size 8 \
  --skip_existing
```

### Important arguments

| Argument | Default | Notes |
|----------|---------|--------|
| `--data_path` | *(required)* | Input/output file (updated in place) |
| `--checkpoint_path` | *(required)* | DPO output dir or full model dir |
| `--model` | `Qwen/Qwen3-0.6B` | Base model when checkpoint is a LoRA adapter |
| `--question_col` / `--concept_col` | *(required)* | Source columns |
| `--output_col` | `model_output` | Generated text column |
| `--batch_size` | `8` | Prompts per forward pass |
| `--max_new_tokens` | `300` | |
| `--temperature` / `--top_p` | `0.8` / `0.95` | Sampling |
| `--conversation` / `--no-conversation` | chat template on | `--no-conversation` uses plain `question: … steering concept: …` text |
| `--start` / `--limit` | `0` / all | Row slice |
| `--skip_existing` | off | Skip rows with non-empty `--output_col` |
| `--save_every` | `50` | Periodic flush to disk (`0` = only at end) |

### Typical workflow

1. Train: `python DPO.py … --output_dir runs/my_adapter`
2. Infer: `python inference.py --checkpoint_path runs/my_adapter --data_path eval.csv …`