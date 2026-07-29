"""
checks.py — the two validation steps.

1. validate_rewritten_prompt() — the manipulation check. Did the rewriter actually
   obey the context window? Without this, a null result is ambiguous: it could mean
   "FUNCTION doesn't help" or "the model ignored the instruction to include it."

2. validate_cadquery_code() — did the generated code produce real geometry?
   Executes it in a subprocess and exports an STL.
"""

import ast
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

from prompts import expected_sections, FORBIDDEN_SYNONYMS, MAX_FEATURES

# ---------------------------------------------------------------------------
# 1. Manipulation check on the rewritten prompt
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^\[([A-Z_]+)\]\s*$", re.MULTILINE)


def parse_prompt_sections(text: str) -> dict:
    """Split a rewritten prompt into {SECTION_NAME: body_text}."""
    if not text:
        return {}
    matches = list(_SECTION_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1)] = text[start:end].strip()
    return out


def validate_rewritten_prompt(text: str, raw_input: str, include_function: bool) -> dict:
    """Check the rewriter followed the context window.

    `prompt_ok` gates a case into the analysis. Structural problems (missing,
    empty, or leaked sections) set it False. Style problems (bad vocabulary,
    fabricated numbers) are reported but do NOT disqualify a case — they are for
    review, not exclusion.
    """
    expected = expected_sections(include_function)
    sections = parse_prompt_sections(text)

    missing = [s for s in expected if s not in sections]
    unexpected = [s for s in sections if s not in expected]
    empty = [s for s in expected if s in sections and not sections[s].strip()]

    # The most important check. If FUNCTION appears under --ablation no_function,
    # the manipulation never took effect and the case is unusable.
    ablation_broken = (not include_function) and ("FUNCTION" in sections)

    details = sections.get("DETAILS", "")
    n_features = len([ln for ln in details.splitlines() if ln.strip()])

    # Rule 3: forbidden operation synonyms.
    words = set(re.findall(r"[a-z]+", details.lower()))
    synonyms_used = sorted(words & FORBIDDEN_SYNONYMS)

    # Numbers in DETAILS that were not in the user's input — the context window
    # forbids inventing specific dimensions.
    nums_in = set(re.findall(r"\d+(?:\.\d+)?", raw_input))
    nums_out = set(re.findall(r"\d+(?:\.\d+)?", details))
    fabricated_numbers = sorted(nums_out - nums_in)

    assumptions = sections.get("ASSUMPTIONS", "").strip().lower().rstrip(".")

    return {
        "prompt_ok": (not missing) and (not empty) and (not ablation_broken),
        "ablation_broken": ablation_broken,
        "sections_found": ",".join(sections),
        "missing_sections": ",".join(missing),
        "unexpected_sections": ",".join(unexpected),
        "empty_sections": ",".join(empty),
        "n_features": n_features,
        "over_feature_limit": n_features > MAX_FEATURES,
        "vocabulary_violations": ",".join(synonyms_used),
        "fabricated_numbers": ",".join(fabricated_numbers),
        "function_wordcount": len(sections.get("FUNCTION", "").split()),
        "assumptions_none": assumptions in ("none", ""),
    }


# ---------------------------------------------------------------------------
# 2. Geometry validation
# ---------------------------------------------------------------------------

_VALIDATOR_TEMPLATE = textwrap.dedent("""\
    import sys, json

    report = {{"executed": False, "error": None, "num_faces": None,
               "volume": None, "exported_stl": False}}
    try:
        import cadquery as cq

    {indented_code}

        if "result" not in dir():
            raise NameError("Generated code did not define a variable named `result`")

        solid = result.val()
        report["executed"] = True
        report["num_faces"] = len(solid.Faces())
        report["volume"] = solid.Volume()

        cq.exporters.export(result, "{stl_path}")
        report["exported_stl"] = True

    except Exception as e:
        report["error"] = f"{{type(e).__name__}}: {{e}}"

    print(json.dumps(report))
    """)

_FAIL = {"executed": False, "num_faces": None, "volume": None, "exported_stl": False}


# Modules the generated code must never import. CadQuery wraps OpenCascade, but
# the low-level bindings are not a supported entry point here and importing them
# either crashes or produces geometry the exporter cannot handle. Small models
# reach for these constantly, so we catch it before spending a subprocess.
_BANNED_IMPORTS = ("OCC", "OCP", "pythonocc", "TopoDS", "BRep", "gp_Pnt", "gp_Vec",
                   "BRepBuilderAPI", "BRepPrimAPI", "TopExp", "TopAbs")


def lint_generated_code(code: str) -> Optional[str]:
    """Cheap static checks before execution.

    Returns an error string if the code is definitely broken, else None. The
    message is written to be fed straight back into the repair loop, so it says
    what to do rather than just what is wrong.
    """
    if not code.strip():
        return "The response contained no code."

    for line in code.splitlines():
        s = line.strip()
        if not (s.startswith("import ") or s.startswith("from ")):
            continue
        for banned in _BANNED_IMPORTS:
            if re.search(rf"\b{re.escape(banned)}\b", s):
                return (f"Forbidden import: {s!r}. The only permitted import is "
                        f"`import cadquery as cq`. Rewrite the whole script using "
                        f"only the high-level cq.Workplane fluent API — no OCC, "
                        f"OCP, or other low-level OpenCascade modules.")

    if not re.search(r"^\s*result\s*=", code, re.MULTILINE):
        return ("The script never assigns to a variable named `result`. The final "
                "solid must be stored in `result`.")

    if "exporters.export" in code or "cq.exporters" in code:
        return ("Remove the export statement. Exporting is handled separately; "
                "the script must only build `result`.")

    return None


def validate_cadquery_code(code: str, case_dir: Path, timeout: int = 60) -> dict:
    """Run generated CadQuery code in a subprocess; save model.stl on success.

    A subprocess is used so that a hang, segfault, or infinite loop in generated
    code kills only that child, not the whole experiment run.
    """
    lint_error = lint_generated_code(code)
    if lint_error:
        return {**_FAIL, "error": lint_error, "failed_lint": True}

    stl_path = case_dir / "model.stl"
    script_path = case_dir / "_validator.py"
    script_path.write_text(_VALIDATOR_TEMPLATE.format(
        indented_code=textwrap.indent(code, "    "), stl_path=stl_path))

    try:
        proc = subprocess.run([sys.executable, str(script_path)],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**_FAIL, "error": f"Timed out after {timeout}s"}
    finally:
        script_path.unlink(missing_ok=True)

    lines = proc.stdout.strip().splitlines()
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass

    err = (proc.stderr.strip()[-2000:] if proc.stderr else "Unknown failure")
    # If cadquery isn't installed, fall back to a syntax-only check so the rest
    # of the pipeline can still be exercised.
    if "No module named 'cadquery'" in err:
        try:
            ast.parse(code)
            return {**_FAIL, "executed": None, "error": "cadquery not installed; syntax OK"}
        except SyntaxError as e:
            return {**_FAIL, "error": f"SyntaxError: {e}"}
    return {**_FAIL, "error": err}