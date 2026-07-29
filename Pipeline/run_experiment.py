#!/usr/bin/env python3
"""
run_experiment.py — entry point. Run this file.

    human prompt --[Gemini: rewriter]--> structured prompt
                 --[local GPU: generator, with repair loop]--> CadQuery code
                 --[subprocess]--> validated geometry (STL)

Usage:
    python run_experiment.py --check-gpu
    python run_experiment.py --dry-run --limit 2
    python run_experiment.py --inputs cad_prompts_function.csv --ablation full        --out results_full/
    python run_experiment.py --inputs cad_prompts_function.csv --ablation no_function --out results_nofunc/

Settings live in config.txt (see config.example.txt).

Project layout:
    bootstrap.py  env setup — must be imported first
    prompts.py    the context window (the experiment's independent variable)
    checks.py     manipulation check + geometry validation
    models.py     Gemini API and local GPU backends
    run_experiment.py   this file
"""

import bootstrap  # noqa: F401  — MUST be first: sets CUDA_VISIBLE_DEVICES before torch loads

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import models
from checks import validate_cadquery_code, validate_rewritten_prompt
from prompts import (ABLATION_CONDITIONS, CODEGEN_SYSTEM_PROMPT,
                     build_codegen_user_message, build_rewriter_system_prompt,
                     build_rewriter_user_message)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_TEXT_COLUMNS = ["prompt", "text", "input", "description", "instruction"]
# A tag column recorded in the results for your own analysis (e.g. part_type).
# It is NEVER sent to the model — that would leak the experimental design into
# the prompt and bias the thing being measured.
_TAG_COLUMNS = ["part_type", "group", "category", "tag", "label"]


def _guess_column(keys, candidates) -> Optional[str]:
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def load_dataset(path: Path, text_column: Optional[str] = None,
                 tag_column: Optional[str] = None) -> list:
    """Load prompts from a CSV. Auto-detects the prompt and tag columns."""
    if path.suffix.lower() != ".csv":
        raise SystemExit(f"Expected a .csv file, got {path.suffix!r}.")
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise SystemExit(f"{path} has no rows.")

    keys = rows[0].keys()
    text_col = text_column or _guess_column(keys, _TEXT_COLUMNS)
    if not text_col:
        raise SystemExit(f"Could not find the prompt column in {path}. "
                         f"Columns found: {list(keys)}. Pass --text-column <name>.")
    tag_col = tag_column or _guess_column(keys, _TAG_COLUMNS)

    return [{"raw_input": r[text_col], "tag": r.get(tag_col, "") if tag_col else ""}
            for r in rows if r.get(text_col)]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class Result:
    case_id: str
    raw_input: str
    tag: str = ""
    ablation: str = "full"
    rewritten_prompt: Optional[str] = None
    prompt_check: dict = field(default_factory=dict)
    generated_code: Optional[str] = None
    repair_attempts: int = 0
    validation: dict = field(default_factory=dict)
    stage_reached: str = "not_started"
    error: Optional[str] = None


def _slug(text: str, n: int = 40) -> str:
    return (re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n]) or "case"


_DRY_RUN_CODE = ("import cadquery as cq\n"
                 "result = (cq.Workplane('XY').box(30, 20, 10)\n"
                 "          .faces('>Z').workplane().hole(6))\n")


def _dry_run_prompt(include_function: bool) -> str:
    parts = ["[OVERVIEW]\nA rectangular block with a through-hole (dry-run stub).",
             "[DETAILS]\n- extrude - rectangular base block\n"
             "- hole - centered through-hole - top face - medium"]
    if include_function:
        parts.append("[FUNCTION]\nA test fixture: needs a flat base and a central bore.")
    parts.append("[ASSUMPTIONS]\nnone")
    return "\n\n".join(parts)


def run_case(idx: int, raw_input: str, tag: str, out_dir: Path, ablation: str,
             max_repair_attempts: int, dry_run: bool, temperature: float,
             max_new_tokens: int) -> Result:
    case_dir = out_dir / f"case_{idx:03d}_{_slug(raw_input)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    r = Result(case_id=case_dir.name, raw_input=raw_input, tag=tag, ablation=ablation)

    include_function = ABLATION_CONDITIONS[ablation]

    # --- Stage 1: rewrite -----------------------------------------------
    if dry_run:
        r.rewritten_prompt = _dry_run_prompt(include_function)
    else:
        try:
            r.rewritten_prompt = models.call_api([
                {"role": "system", "content": build_rewriter_system_prompt(include_function)},
                {"role": "user", "content": build_rewriter_user_message(raw_input)},
            ])
        except Exception as e:
            r.stage_reached, r.error = "rewrite_failed", str(e)
            (case_dir / "error.txt").write_text(f"Rewrite failed: {e}")
            return r
    (case_dir / "rewritten_prompt.txt").write_text(r.rewritten_prompt)

    # --- Manipulation check: did the context window actually work? -------
    r.prompt_check = validate_rewritten_prompt(r.rewritten_prompt, raw_input, include_function)
    (case_dir / "prompt_check.json").write_text(json.dumps(r.prompt_check, indent=2))
    if r.prompt_check["ablation_broken"]:
        print("    WARNING: [FUNCTION] present despite --ablation no_function; "
              "this case cannot be used.", file=sys.stderr)
    elif not r.prompt_check["prompt_ok"]:
        print(f"    WARNING: prompt failed structure check "
              f"(missing={r.prompt_check['missing_sections']!r}, "
              f"empty={r.prompt_check['empty_sections']!r})", file=sys.stderr)
    r.stage_reached = "rewritten"

    # --- Stage 2: generate code, with repair loop ------------------------
    if dry_run:
        r.generated_code = _DRY_RUN_CODE
        (case_dir / "generated_code.py").write_text(r.generated_code)
        r.validation = validate_cadquery_code(r.generated_code, case_dir)
        r.stage_reached = "validated"
        return r

    messages = [
        {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
        {"role": "user", "content": build_codegen_user_message(r.rewritten_prompt)},
    ]
    for attempt in range(max_repair_attempts + 1):
        try:
            raw_response = models.call_local(messages, temperature=temperature,
                                             max_new_tokens=max_new_tokens)
        except Exception as e:
            r.stage_reached, r.error = "codegen_failed", str(e)
            (case_dir / "error.txt").write_text(f"Codegen failed: {e}")
            return r

        r.generated_code = models.extract_code_block(raw_response)
        r.repair_attempts = attempt
        r.validation = validate_cadquery_code(r.generated_code, case_dir)

        if r.validation.get("executed") in (True, None):
            break
        if attempt < max_repair_attempts:
            messages += [
                {"role": "assistant", "content": raw_response},
                {"role": "user", "content":
                    f"That code failed with this error:\n\n{r.validation.get('error')}\n\n"
                    f"Fix the specific problem. Do not simplify or remove features to "
                    f"make the error go away. Output the full corrected code."},
            ]

    (case_dir / "generated_code.py").write_text(r.generated_code or "")
    r.stage_reached = "validated"
    return r


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_CSV_COLUMNS = ["case_id", "ablation", "tag", "status", "raw_input", "repair_attempts",
                "num_faces", "volume", "stl_path", "error",
                # manipulation check
                "prompt_ok", "ablation_broken", "sections_found", "missing_sections",
                "n_features", "over_feature_limit", "vocabulary_violations",
                "fabricated_numbers", "function_wordcount", "assumptions_none"]


def write_summary(results: list, out_dir: Path) -> Path:
    path = out_dir / "summary.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in results:
            executed = r.validation.get("executed")
            status = ("VALID" if executed is True else
                      "UNKNOWN (cadquery not installed)" if executed is None else "FAILED")
            stl_rel = f"{r.case_id}/model.stl"
            pc = r.prompt_check or {}
            w.writerow({
                "case_id": r.case_id, "ablation": r.ablation, "tag": r.tag,
                "status": status, "raw_input": r.raw_input,
                "repair_attempts": r.repair_attempts,
                "num_faces": r.validation.get("num_faces"),
                "volume": r.validation.get("volume"),
                "stl_path": stl_rel if (out_dir / stl_rel).exists() else "",
                "error": r.validation.get("error") or "",
                **{k: pc.get(k, "") for k in _CSV_COLUMNS[10:]},
            })
    return path


def print_summary(results: list, summary_path: Path, zip_path: Path) -> None:
    n = len(results)
    ok = sum(1 for r in results if (r.prompt_check or {}).get("prompt_ok"))
    broken = sum(1 for r in results if (r.prompt_check or {}).get("ablation_broken"))
    valid = sum(1 for r in results if r.validation.get("executed") is True)
    repaired = sum(1 for r in results if r.repair_attempts > 0)

    print("=" * 62)
    print("CONTEXT WINDOW CHECK (did the rewriter follow instructions?)")
    print(f"  Prompts with correct structure: {ok}/{n}")
    if broken:
        print(f"  *** ABLATION BROKEN on {broken}/{n} cases: [FUNCTION] appeared when "
              f"it should not have. Exclude these before analysing. ***")
    print("-" * 62)
    print("CAD GENERATION")
    print(f"  Valid, exportable geometry: {valid}/{n}")
    print(f"  Needed at least 1 repair:   {repaired}/{n}")
    print("-" * 62)
    print(f"Summary CSV:          {summary_path}")
    print(f"Everything zipped to: {zip_path}   <-- download this one file")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, help="CSV of prompts.")
    p.add_argument("--text-column", help="Column holding the prompt, if not auto-detected.")
    p.add_argument("--tag-column", help="Column holding your own analysis tag (e.g. part_type). "
                                        "Recorded in the results; never sent to the model.")
    p.add_argument("--ablation", default="full", choices=list(ABLATION_CONDITIONS),
                   help="'full' includes the FUNCTION section; 'no_function' omits it. "
                        "ASSUMPTIONS is always included.")
    p.add_argument("--out", type=Path, default=Path("results"), help="Output directory.")
    p.add_argument("--limit", type=int, help="Only run the first N cases.")
    p.add_argument("--max-repair-attempts", type=int, default=3,
                   help="Retries after a failed execution (default: 3; 0 = one-shot).")
    p.add_argument("--gen-temperature", type=float, default=0.6,
                   help="Sampling temperature for the local generator (default: 0.6).")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--gpu-device", help="Which GPU to use, e.g. '0'. Read at startup by bootstrap.")
    p.add_argument("--dry-run", action="store_true",
                   help="No API or GPU calls; exercises the scaffolding with a stub shape.")
    p.add_argument("--check-gpu", action="store_true", help="Print GPU visibility and exit.")
    args = p.parse_args()

    if args.check_gpu:
        models.check_gpu()
        return
    if not args.inputs:
        p.error("--inputs is required (or use --check-gpu).")

    cases = load_dataset(args.inputs, args.text_column, args.tag_column)
    if args.limit:
        cases = cases[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Running in --dry-run mode: no API or GPU calls will be made.")
    else:
        print(f"Rewriter (API):        {models.REWRITER.name} ({models.REWRITER.model})")
        print(f"Generator (local GPU): {models.GENERATION_MODEL}")
        models.load_local_model(models.GENERATION_MODEL)  # load once, up front
    print(f"Ablation: {args.ablation} "
          f"(FUNCTION={'included' if ABLATION_CONDITIONS[args.ablation] else 'omitted'}, "
          f"ASSUMPTIONS always included)")
    print(f"Cases: {len(cases)}\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['raw_input'][:70]}...")
        r = run_case(i, case["raw_input"], case["tag"], args.out, args.ablation,
                     args.max_repair_attempts, args.dry_run,
                     args.gen_temperature, args.max_new_tokens)
        results.append(r)
        executed = r.validation.get("executed")
        status = "VALID" if executed is True else ("UNKNOWN" if executed is None else "FAILED")
        print(f"    status={status}  repairs={r.repair_attempts}  "
              f"note={r.validation.get('error')}\n")

    (args.out / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    summary_path = write_summary(results, args.out)
    zip_path = Path(shutil.make_archive(str(args.out), "zip", root_dir=args.out))
    print_summary(results, summary_path, zip_path)


if __name__ == "__main__":
    main()
