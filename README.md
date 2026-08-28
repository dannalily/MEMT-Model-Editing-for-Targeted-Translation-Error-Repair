# MEMT: Model Editing for Targeted Translation Error Repair

This repository contains the code and documentation for our EMNLP paper. MEMT studies whether model editing can repair a targeted idiomatic translation error while preserving unrelated translation and reasoning behavior.

The code uses the base models listed below and downloads them from Hugging Face when they are first used.

## Repository map

```text
MEMT-public-release/
├── bench_construction/        # Build MEMT from IdiomKB; generate and evaluate data
├── causal_tracing/             # Locate important layers and tokens
├── editing_code/               # Editing entry points, algorithms, and hyperparameters
│   ├── edit_main.py            # One edit at a time
│   ├── edit_batch_main.py      # Multiple edits in one update
│   ├── edit_sequential_main.py # Edits applied one after another
│   ├── editors/                # FT, ROME, MEMIT, AlphaEdit, UNKE, WISE, GRACE
│   └── hparams/                # Direction-specific YAML configurations
├── models/                     # Qwen, NLLB, M2M100, and Claude/Bedrock wrappers
├── util_editing/               # Shared hooks, generation, representations, statistics
├── evaluation_editing.py       # Reliability/generalization/locality/FLORES/MMLU evaluation
├── language_detection.py       # Check target-language correctness
├── print_*                     # Aggregate JSON results into Excel/plots
├── scripts/                    # Multi-GPU launchers
└── docs/data.md               # Data placement and external-data instructions
```

In the released data, `Editing_Samples/` contains incorrect translations used
as editing descriptors, while `Keep_Samples/` contains translations that the
model already gets correct. The latter are retained as control samples for
causal-tracing analysis; they are not edited.

Directories created only after experiments start:

| Directory | Created by | Contents |
|---|---|---|
| `editing_results/` | `edit_main.py` | Generated outputs after single edits |
| `editing_eval/` | `evaluation_editing.py` | Scores for single editing |
| `editing_batch_results/` | `edit_batch_main.py` | Generated outputs after batch edits |
| `editing_batch_eval/` | evaluation scripts | Scores for batch editing |
| `editing_sequential_results/` | `edit_sequential_main.py` | Outputs after each sequential step |
| `editing_sequential_eval/` | evaluation scripts | Scores across sequential steps |
| `Causal_Tracing_Results/` | `causal_tracing/` | Layer/token causal effects |
| `logs*/` | shell launchers | Standard output and error logs |
| `excel*/`, `plot*/` | `print_*` scripts | Tables and figures for analysis |

The `MEMT/` directory is intentionally not stored in GitHub. Download the companion Hugging Face dataset and place its `data/` contents under `MEMT/` before running experiments.

## 1. Installation

Run all commands from the repository root:

```bash
git clone https://github.com/dannalily/MEMT-Model-Editing-for-Targeted-Translation-Error-Repair.git
cd MEMT-public-release
python -m venv .venv
source .venv/bin/activate
```

Install the packages required by the part of the release you want to run:

```bash
# Model editing and causal tracing
pip install torch transformers datasets numpy tqdm pyyaml sentencepiece

# Evaluation and result organization
pip install pandas openpyxl matplotlib

# Benchmark construction and external model/API access
pip install boto3 nltk
```

The evaluation code loads some metrics lazily. Install these only when using
the corresponding metric:

```bash
pip install unbabel-comet bleurt-pytorch
```

MetricX is not bundled with this repository. When using MetricX evaluation,
install it from the upstream repository:

```bash
git clone https://github.com/google-research/metricx src/metricx
pip install -r src/metricx/requirements.txt
```

For GPU experiments, install a CUDA-compatible PyTorch version. The complete
experiments use multiple GPUs and a large base model; start with one method and
one language direction as a smoke test.

Optional Hugging Face cache configuration:

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

## 2. Download and install the data

Download the MEMT dataset from the Hugging Face dataset page, then copy the contents of its `data/` directory into `MEMT/`:

```bash
mkdir -p MEMT
cp -R /path/to/MEMT-dataset/data/. MEMT/
```

At minimum, the following files should exist:

```text
MEMT/ParaIdiomSent/en2x.json
MEMT/ParaIdiomSent/zh2x.json
MEMT/ParaIdiomSent_generality/en2x.json
MEMT/ParaIdiomSent_locality/en2x.json
MEMT/Editing_Samples/Qwen2-5-3B-Instruct/en2x.json
MEMT/idiom_span_en2x.json
MEMT/idiom_span_zh2x.json
```

The released project data include the primary idiom data, generalization data, locality data, editing descriptors, retained samples, and the fixed shuffled control subsets used in the paper. Full locality evaluation uses:

```text
MEMT/locality_flores_shuffle1000.json
MEMT/locality_MMMLU_shuffle1000.json
```

See [docs/data.md](docs/data.md) for the complete layout and provenance.

## 3. Base models

The default experiments use `Qwen/Qwen2.5-3B-Instruct`. Other model names referenced by the code are:

```text
Qwen/Qwen2.5-7B-Instruct
facebook/m2m100_1.2B
facebook/nllb-200-3.3B
```

`transformers` downloads the selected base model automatically, subject to the model's own license and Hugging Face access requirements.

## 4. Single editing

Start with one method, one direction, and one GPU:

```bash
python editing_code/edit_main.py \
  --editing_method FT \
  --hparam_file editing_code/hparams/FT/qwen2-5-3b-instruct_en2x.yaml \
  --slang en \
  --tlang zh
```

Available methods are `FT`, `ROME`, `MEMIT`, `AlphaEdit`, `UNKE`, `WISE`, and `GRACE`. The source language is `en` or `zh`; the target language can be `en`, `zh`, `ja`, `de`, `fr`, or `ar`. The YAML file must match the source language.

The causal-tracing-based layer settings are:

| Source | Single-layer methods | Multi-layer methods |
|---|---:|---:|
| English | layer 7 | layers 5–9 |
| Chinese | layer 5 | layers 2–6 |
| WISE/GRACE | layer 30 | — |

Generated outputs are written under `editing_results/Qwen2-5-3B-Instruct/<METHOD>/`.

## 5. Evaluate single editing

After the editing command finishes:

```bash
python evaluation_editing.py \
  --metric bleurt \
  --editing FT \
  --annotate layer7 \
  --slang en \
  --tlang zh \
  --translator Qwen2-5-3B-Instruct \
  --type single
```

Results are written to `editing_eval/Qwen2-5-3B-Instruct/FT/en_zh/`.

The evaluation subsets are:

- `r`: reliability on the edited descriptor
- `g`: generalization to related contexts
- `l`: locality on unrelated inputs
- `f`: FLORES translation preservation
- `mmlu`: general factual/reasoning preservation

The two control files are fixed shuffled subsets; use them as provided and do not reshuffle them when reproducing the reported results.

## 6. Batch and sequential editing

Batch editing applies multiple descriptors in one update:

```bash
python editing_code/edit_batch_main.py \
  --editing_method MEMIT \
  --hparam_file editing_code/hparams/MEMIT/qwen2-5-3b-instruct_en2x.yaml \
  --slang en --tlang zh --seed 42
```

Sequential editing applies descriptors one at a time:

```bash
python editing_code/edit_sequential_main.py \
  --editing_method GRACE \
  --hparam_file editing_code/hparams/GRACE/qwen2-5-3b-instruct_en2x.yaml \
  --slang en --tlang zh --seed 42
```

Batch outputs go to `editing_batch_results/`; sequential outputs go to `editing_sequential_results/`. Their evaluation outputs use the corresponding `*_eval/` directories.

For complete multi-GPU runs, edit the GPU IDs, language arrays, and environment commands in:

```text
scripts/run_single_editing.sh
scripts/run_batch_editing.sh
scripts/run_seq_editing.sh
```

The scripts use the Python environment currently active in your shell. They use eight GPU slots by default; edit the `gpus=(...)` arrays before running on a different machine. The batch launcher contains post-processing sections that assume batch result files already exist; use the direct command above for a fresh smoke test.

## 7. Causal tracing

Causal tracing identifies where idiomatic translation information is represented in the base model. It uses the correctly translated control samples under `MEMT/Keep_Samples/`. These samples are retained because the base model already produces the expected translation; they are used to measure which layers and tokens support successful idiom translation, not as editing targets:

```bash
python causal_tracing/causal_tracing.py \
  --translator Qwen2-5-3B-Instruct --slang en --tlang zh

python causal_tracing/causal_affect_visualization.py \
  --translator Qwen2-5-3B-Instruct --slang en --tlang zh
```

Results are written under `Causal_Tracing_Results/`.

## 8. Rebuild the benchmark from scratch

This is optional. It is not needed when using the released data. Obtain IdiomKB from its original repository, configure access to Claude/Bedrock, and run:

```bash
python bench_construction/clear_duplicate_idiom.py
python bench_construction/sentence_w_idiom_generation.py
python bench_construction/generality_sample_generate.py
python bench_construction/generality_sample_trans.py
python bench_construction/locality_sample_generate.py
python bench_construction/locality_sample_trans.py
python bench_construction/generality_locality_samples_organize.py
python bench_construction/quality_check_IdiomSent2PlainSent.py
python bench_construction/quality_check_trans.py
python bench_construction/idiom_span_detect.py
```

Then use `translation_generate.py`, `evaluation.py`, and `editing_sample_collect.py` to generate model outputs, evaluate them, and collect editing descriptors. These scripts need external model/API access and are not required for ordinary benchmark use.

## 9. Result organization

Run the `print_*` scripts only after the corresponding editing and evaluation directories exist. They aggregate raw JSON results into:

```text
excel_editing/
excel_seq_editing/
excel_batch_editing/
plot_seq_editing/
plot_batch_editing/
```

## Citation

```bibtex
@inproceedings{memt,
  title     = {Model Editing for Targeted Translation Error Repair},
  author    = {Zheng, Danna and Liu, Yang and Liu, Zhu and Bhat, Vimal and Agrawal, Manoj and Medioni, Gerard and Sadoughi, Najmeh},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025}
}
```

## License and attribution

This repository and the accompanying MEMT data are released for non-commercial
research and evaluation under [CC BY-NC 4.0](LICENSE). Please read
[DATA_CARD.md](DATA_CARD.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before using or redistributing the data.
