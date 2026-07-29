# Aipex Lab Research

Benchmarking and improving text-to-CAD generation. Given a natural-language description of a 3D shape, a model generates [CadQuery](https://cadquery.readthedocs.io/) Python code, which is executed to produce an STL mesh.

The first half of this work benchmarks existing models on a fixed prompt set. The second half investigates whether an intermediate LLM that rewrites the user's prompt into a structured, function-aware specification improves generation quality.

## Projects

### [`Zero-to-CAD Benchmark/`](Zero-to-CAD%20Benchmark/)
Benchmarks the Zero-to-CAD model on a fixed prompt set using zero-shot prompting. The fine-tuned model was trained for Image-to-CAD, so the text-based zero-shot approach is used instead.

### [`Qwen 3.5-2B/`](Qwen%203.5-2B/)
Benchmarks the general-purpose Qwen 3.5-2B model, using an intermediate LLM to rework prompts into sanitized technical specifications and a 4-iteration feedback loop for error correction.

### [`Context Window Pipeline/`](Context%20Window%20Pipeline/)
The independent research contribution. Tests whether adding a FUNCTION section to the rewriter's context window improves CAD generation for parts whose function implies conventional geometric features.

## Setup

All benchmarks expect:
- A CUDA GPU (scripts pin to a specific device via `CUDA_VISIBLE_DEVICES`)
- A conda environment with `torch`, `transformers`, and `cadquery` installed
- Several scripts apply an `LD_PRELOAD` patch at startup to avoid a CadQuery/torch C++ ABI crash

See each project's README for additional requirements.

## What's not tracked

Generated artifacts are gitignored: `*.stl` outputs, `*.log` run logs, `*.zip` result archives, `res_*/` and `results_*/` output directories, `config.txt` (contains API keys), editor backups, and `__pycache__/`.
