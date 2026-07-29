# Zero-to-CAD Benchmark

Benchmarks [ADSKAILab/Zero-To-CAD-Qwen3-VL-2B](https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B) on a fixed list of shape-description prompts.

**Note:** The fine-tuned model from the Zero-to-CAD paper was trained for **Image-to-CAD**, not text-to-CAD — its learned mapping expects rendered views as input, not text. Since that modality does not transfer, this benchmark uses the paper's **zero-shot prompting approach** instead, which does not rely on the fine-tuned weights.

## Files

- `test_zero.py` — single-prompt run of the model end to end (generate CAD code, execute it, export STL).
- `run_all_benchmarks.py` — runs the full prompt list, generating CAD code with and without reasoning for each.
- `run_formatted_prompts.py` — same as above, against a reworded/paraphrased version of the prompt list.
- `split_cad.py` — takes generated code blocks, executes them with CadQuery, and exports each result to `.stl`. If execution fails or no `result` variable is produced, writes a `_error.txt` note alongside instead.

## Setup

Requires a CUDA GPU and a conda environment with `torch`, `transformers`, and `cadquery`. The script pins to `CUDA_VISIBLE_DEVICES=1` and applies an `LD_PRELOAD` patch at startup to avoid a CadQuery/torch C++ ABI crash.
