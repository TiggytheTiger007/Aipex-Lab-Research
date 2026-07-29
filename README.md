# Aipex Lab Research

Benchmarking and improving text-to-CAD generation. Given a natural-language description of a 3D shape, a model generates [CadQuery](https://cadquery.readthedocs.io/) Python code, which is executed to produce an STL mesh.

The first half of this work benchmarks existing models on a fixed prompt set. The second half investigates whether an intermediate LLM that rewrites the user's prompt into a structured, function-aware specification improves generation quality.

## Structure

### `Zero-to-CAD Benchmark/`

Benchmarks [ADSKAILab/Zero-To-CAD-Qwen3-VL-2B](https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B) on a fixed list of shape-description prompts.

**Note:** The fine-tuned model from the Zero-to-CAD paper was trained for **Image-to-CAD**, not text-to-CAD — its learned mapping expects rendered views as input, not text. Since that modality does not transfer, this benchmark uses the paper's **zero-shot prompting approach** instead, which does not rely on the fine-tuned weights.

- `test_zero.py` — single-prompt run of the model end to end (generate CAD code, execute it, export STL).
- `run_all_benchmarks.py` — runs the full prompt list, generating CAD code with and without reasoning for each.
- `run_formatted_prompts.py` — same as above, against a reworded/paraphrased version of the prompt list.
- `split_cad.py` — takes generated code blocks, executes them with CadQuery, and exports each result to `.stl`. If execution fails or no `result` variable is produced, writes a `_error.txt` note alongside instead.

### `Qwen 3.5-2B/`

Benchmarks the general-purpose `Qwen/Qwen3.5-2B` model on the same style of task, for comparison against the CAD-specialized model above.

- `benchmark_qwen` — interactive/agentic benchmark script (Python, no extension) that loads the model and drives generation.

**Method:** An intermediate LLM was used to rework the original dataset prompts into sanitized, zero-shot-style prompts before feeding them to Qwen. The transformation prompt instructed the LLM to:

1. Strip conversational filler ("The 3D shape is…")
2. State the primary geometric solid first using formal terms ("solid rectangular prism", "cylindrical sleeve")
3. Convert vague spatial descriptions into precise CAD operations ("hole inside" → "concentric through-hole")
4. Use proportional constraints instead of explicit dimensions ("thin-walled", "flush", "full-length")
5. Return a single condensed plain-text sentence with no formatting

For example, *"The 3D shape is a cylinder and a hexagonal hole inside, which is smaller and makes the wall very thin"* becomes *"A thin-walled cylindrical sleeve featuring a central, full-length hexagonal through-hole."*

A **feedback loop** (4 iterations) was also used: if the generated CadQuery code failed to execute, the error was fed back to the model so it could attempt to fix its own code before the case was marked as failed.

### `Context Window Pipeline/`

The independent research contribution. Tests whether an intermediate "translator" LLM that rewrites casual user prompts into structured geometry specifications improves downstream CAD generation.

**The pipeline:**

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

**The hypothesis:** stating a part's mechanical function helps the generation model produce correct geometry, for parts whose function implies conventional geometric features.

**The experiment:** the same 28 function-implied prompts are run twice — once with the FUNCTION section (`--ablation full`) and once without (`--ablation no_function`). ASSUMPTIONS is present in both conditions. A manipulation check verifies the rewriter actually produced the expected sections before any downstream result is interpreted.

**Files:**

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

**Usage:**

```bash
cd "Context Window Pipeline"
cp config.example.txt config.txt        # fill in your Gemini key, model ID, GPU
python run_experiment.py --check-gpu    # confirm GPU is visible
python run_experiment.py --dry-run --limit 2 --out res_test/   # scaffolding check
python run_experiment.py --inputs cad_prompts_function.csv --ablation full        --out results_full/
python run_experiment.py --inputs cad_prompts_function.csv --ablation no_function --out results_nofunc/
```

## Setup notes

All benchmarks expect:

- A CUDA GPU (scripts pin to a specific device via `CUDA_VISIBLE_DEVICES`).
- A conda environment with `torch`, `transformers`, and `cadquery` installed.
- The Context Window Pipeline additionally requires `openai` (for the Gemini rewriter) and a `config.txt` with a valid Gemini API key.
- Several scripts apply an `LD_PRELOAD` patch at startup to point at a newer `libstdc++.so.6` (from the active conda env) to avoid a CadQuery/torch C++ ABI crash.

## What's not tracked

Generated artifacts are gitignored: `*.stl` outputs, `*.log` run logs, `*.zip` result archives, `res_*/` and `results_*/` output directories, `config.txt` (contains API keys), editor backups, and `__pycache__/`.
