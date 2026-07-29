# Qwen 3.5-2B Benchmark

Benchmarks the general-purpose `Qwen/Qwen3.5-2B` model on text-to-CAD generation, for comparison against the CAD-specialized Zero-to-CAD model.

## Method

An intermediate LLM was used to rework the original dataset prompts into sanitized, zero-shot-style prompts before feeding them to Qwen. The transformation prompt instructed the LLM to:

1. Strip conversational filler ("The 3D shape is…")
2. State the primary geometric solid first using formal terms ("solid rectangular prism", "cylindrical sleeve")
3. Convert vague spatial descriptions into precise CAD operations ("hole inside" → "concentric through-hole")
4. Use proportional constraints instead of explicit dimensions ("thin-walled", "flush", "full-length")
5. Return a single condensed plain-text sentence with no formatting

For example, *"The 3D shape is a cylinder and a hexagonal hole inside, which is smaller and makes the wall very thin"* becomes *"A thin-walled cylindrical sleeve featuring a central, full-length hexagonal through-hole."*

A **feedback loop** (4 iterations) was also used: if the generated CadQuery code failed to execute, the error was fed back to the model so it could attempt to fix its own code before the case was marked as failed.

## Files

- `benchmark_qwen` — interactive/agentic benchmark script (Python, no extension) that loads the model and drives generation.

## Setup

Requires a CUDA GPU and a conda environment with `torch`, `transformers`, and `cadquery`. The script applies an `LD_PRELOAD` patch at startup to avoid a CadQuery/torch C++ ABI crash.
