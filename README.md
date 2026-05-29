
# DPO-Steering

This repository contains the implementation, experimental artifacts, and evaluation resources associated with the research project on **steering large language models using the Direct Preference Optimization (DPO) algorithm**.

---

# Repository Overview

The repository is organized into three main components:

1. **Task 1 (T1): AxBench replication and extensions**
2. **Task 2 (T2): SAE-TS experiments**
3. **DPO-based steering experiments**

---

# Task 1 (T1): AxBench Replication and Extensions

The experiments for Task 1 are based on the original implementation from the Stanford NLP **AxBench** repository:

- Original repository:  
  https://github.com/stanfordnlp/axbench/tree/main

Our work reproduces the original framework while introducing several modifications and extensions.

## Modifications

Compared to the original AxBench setup:

- The number of tested concepts was increased from **10 to 20**
- The number of trials per concept was reduced from **10 to 5**

The experimental configurations were adapted from the original AxBench sweep configurations:

- Configuration source:  
  https://github.com/stanfordnlp/axbench/tree/main/axbench/sweep/wuzhengx

## Model Support Extensions

While preserving the original folder structure and pipeline design, we extended the framework to support the **Qwen** family of models in addition to the original Gemma and Llama model families.

## Provided Artifacts

All generated artifacts and experimental outputs are available in:

```text
./main_results/T1-results
```

This directory includes:

- Configuration files
- Generated datasets
- Training artifacts
- Inference outputs
- Evaluation results using the LLM-as-a-Judge paradigm

---

# Environment and Dependencies

## Gemma and Llama Models

For the Gemma and Llama model families, we used the original AxBench environment configuration:

- Original environment file:  
  https://github.com/stanfordnlp/axbench/blob/main/pyproject.toml

The only additional dependency required was:

```text
stanza>=1.8.2
```

This modification was necessary to ensure proper execution of the evaluation pipeline.

---

## Qwen Models

For experiments involving the Qwen model family, the following modifications were required:

- Upgraded Transformers version:

```text
transformers>=4.51
```

- Disabled `pyvene` due to compatibility issues with newer Transformer releases.

---

# Task 2 (T2): SAE-TS Experiments

For Task 2, we used the official implementation associated with the SAE-TS paper without modification.

- Original repository:  
  https://github.com/slavachalnev/SAE-TS

---

# Inter-Rater Agreement Experiments

We additionally conducted a side experiment investigating **inter-rater agreement in the LLM-as-a-Judge evaluation paradigm**.

---

## Task 1 Inter-Rater Evaluation

For T1, the experiment required only replacing the original LLM client with alternative providers that support the OpenAI-compatible API interface. No additional modifications to the evaluation pipeline were necessary.

The resulting evaluation outputs for **Gemma2-2B Layer 10** are available in:

```text
./Inter-Rater-results/T1
```

---

## Task 2 Inter-Rater Evaluation

For T2, we ported the original OpenAI multi-criteria evaluation implementation to additional providers, including:

- Anthropic Claude
- Google models

Original implementation source:

https://github.com/slavachalnev/SAE-TS/blob/main/src/sae_ts/steering/evals_utils.py

The implementation used for these experiments is available in:

```text
./Inter-Rater-implementation/T2
```

---

# DPO Steering Experiments

This repository also includes the implementation used for the DPO-based steering experiments.

---

## Training Data

Training datasets used for DPO steering experiments are located in:

```text
./dpo-steering/data
```

---

## Training and Inference

The provided DPO implementation and inference scripts support:

- Training steering models using preference datasets
- Running inference using trained checkpoints

---

## Preparing Training Data

To prepare training data for mDPO steering experiments, we used the original AxBench datasets:

### LLaMA Dataset

https://huggingface.co/datasets/pyvene/axbench-concept16k_v2

### Gemma Dataset

https://huggingface.co/datasets/pyvene/axbench-concept16k

The preprocessing notebooks used to generate the final training datasets are provided in:

```text
./dpo-steering/data-preparation
```

