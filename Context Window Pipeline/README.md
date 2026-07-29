# Context Window Pipeline

Tests whether an intermediate "translator" LLM that rewrites casual user prompts into structured geometry specifications improves downstream CAD generation.

## The Pipeline

```
human prompt  --[Gemini: rewriter]--> structured prompt  --[local GPU: generator]--> CadQuery code --> STL
```

The rewriter applies a **context window** that structures each prompt into four sections:

| Section | Purpose |
|---|---|
| OVERVIEW | Base geometric form in one or two sentences |
| DETAILS | Every feature in CAD construction order |
| FUNCTION | The part's mechanical function and the features it implies *(experimental variable)* |
| ASSUMPTIONS | Everything the rewriter added that the user didn't state *(always on — transparency feature)* |

## Hypothesis

Stating a part's mechanical function helps the generation model produce correct geometry, for parts whose function implies conventional geometric features.

## Experiment

The same 28 function-implied prompts are run twice — once with the FUNCTION section (`--ablation full`) and once without (`--ablation no_function`). ASSUMPTIONS is present in both conditions. A manipulation check verifies the rewriter actually produced the expected sections before any downstream result is interpreted.

## Files

- `run_experiment.py` — main entry point; orchestrates the full pipeline.
- `run_experiment_fewshot.py` — variant that adds few-shot CadQuery examples to the generation prompt, to test whether showing the model worked code improves output.
- `bootstrap.py` — startup environment setup (config loading, GPU pinning, libstdc++ fix). Must be imported before anything that loads torch.
- `prompts.py` — the context window definition and the codegen prompt. This is the scientifically interesting file: it defines the experiment's independent variable.
- `prompts_fewshot.py` — variant of `prompts.py` with three few-shot CadQuery demonstrations added to the generation stage.
- `checks.py` — manipulation check (did the rewriter follow the context window?) and geometry validation (did the code produce a real solid?).
- `models.py` — Gemini API backend (rewriter) and local GPU backend (generator).
- `diagnose_key.py` — troubleshooting script for Gemini API key issues.
- `cad_prompts_function.csv` — the 28-prompt dataset (9 from the original benchmark set, 19 drafted). Each prompt names a functional part where geometry must be inferred from the function.
- `config.example.txt` — template for `config.txt` (API key, model ID, GPU device). Copy to `config.txt` and fill in your values.

## Usage

```bash
cp config.example.txt config.txt        # fill in your Gemini key, model ID, GPU
python run_experiment.py --check-gpu    # confirm GPU is visible
python run_experiment.py --dry-run --limit 2 --out res_test/   # scaffolding check
python run_experiment.py --inputs cad_prompts_function.csv --ablation full        --out results_full/
python run_experiment.py --inputs cad_prompts_function.csv --ablation no_function --out results_nofunc/
```

## Setup

Requires:
- A CUDA GPU
- A conda environment with `torch`, `transformers`, `cadquery`, and `matplotlib` installed
- A `config.txt` with a valid Gemini API key (see `config.example.txt`)

The script applies an `LD_PRELOAD` patch at startup to avoid a CadQuery/torch C++ ABI crash, and pins to the GPU specified in `config.txt`.
