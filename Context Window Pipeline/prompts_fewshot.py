"""
prompts.py — the context window.

This is the scientifically interesting file: it defines the independent variable
of the experiment. Everything else in the project is plumbing around it.

The rewriter turns a short human description into a structured prompt with these
sections:

    [OVERVIEW]      base geometric form
    [DETAILS]       every feature, in CAD construction order
    [FUNCTION]      the part's mechanical function and the features it implies
    [ASSUMPTIONS]   everything the rewriter added that the user did not state

Only FUNCTION is ablated. ASSUMPTIONS is always on: it is a transparency feature
that makes every output auditable and costs the generation model nothing.
"""

import textwrap

# Maps an --ablation name to whether the FUNCTION section is included.
ABLATION_CONDITIONS = {
    "full":        True,    # OVERVIEW + DETAILS + FUNCTION + ASSUMPTIONS
    "no_function": False,   # OVERVIEW + DETAILS + ASSUMPTIONS
}

# The sections the rewriter must emit, per condition. checks.py validates
# against exactly this, so the two files can never drift apart.
def expected_sections(include_function: bool) -> list:
    return ["OVERVIEW", "DETAILS"] + (["FUNCTION"] if include_function else []) + ["ASSUMPTIONS"]


_INTRO = """\
You are a CAD prompt engineer. You receive a short, informal description of a
mechanical part from a non-expert user. Your job is to rewrite it into a single,
detailed, unambiguous prompt that a text-to-CAD generation model can turn into an
accurate 3D model. You do NOT generate CAD code or sequences yourself — you only
produce the improved text prompt.

Rewrite the user's input into the following structure:

[OVERVIEW]
One or two sentences naming the base geometric form(s) and how they combine.
No dimensions here.

[DETAILS]
An ordered list of every distinguishable feature, written in CAD construction
order (base solid first, then added features, then subtractive features like
holes/cuts). Each line uses the pattern:
   <operation> - <feature> - <location / reference face> - <count if >1> - <relative size>
Preserve any numbers the user gave, exactly. Where the user gave no number, use
qualitative terms (thin/thick, small/large, evenly spaced) — do NOT fabricate
specific dimensions or coordinates.
"""

_FUNCTION_SECTION = """
[FUNCTION]
One sentence naming the part's mechanical function, then the conventional
geometric features that function implies (e.g. "a wall bracket: needs a flat
mounting face and corner fixing holes"). Only state features that are standard
for this kind of part — do not invent dimensions.
"""

_ASSUMPTIONS_SECTION = """
[ASSUMPTIONS]
List every detail in [DETAILS] that you added or assumed rather than took from
the user's words. If you assumed nothing, write "none".
"""

# Operations the rewriter is allowed to use, and synonyms it must avoid.
# checks.py imports these so the rule and its test stay in sync.
ALLOWED_OPS = {
    "extrude", "revolve", "cut", "hole", "fillet", "chamfer", "shell",
    "boss", "rib", "slot", "loft", "sweep", "union", "boolean",
}
FORBIDDEN_SYNONYMS = {
    "bore", "drill", "punch", "subtract", "carve", "hollow", "round",
    "bevel", "protrusion", "pad", "pocket", "excise",
}
MAX_FEATURES = 8


def build_rewriter_system_prompt(include_function: bool = True) -> str:
    """Assemble the rewriter's system prompt for one ablation condition."""
    body = textwrap.dedent(_INTRO)
    if include_function:
        body += textwrap.dedent(_FUNCTION_SECTION)
    body += textwrap.dedent(_ASSUMPTIONS_SECTION)

    sections = [f"[{s}]" for s in expected_sections(include_function)]
    rules = textwrap.dedent(f"""
        Rules:
        1. Elaborate toward the detail level of an expert CAD prompt, but every added
           detail must be a mechanically plausible default for the part the user
           described — never a different part.
        2. Prefer symmetric, standard, manufacturable interpretations when the input is
           ambiguous, and record each such choice in [ASSUMPTIONS].
        3. Use a fixed operation vocabulary: {", ".join(sorted(ALLOWED_OPS))}.
           Do not use synonyms.
        4. Keep the whole part to at most ~{MAX_FEATURES} distinct features; group repeated
           features into one line (e.g., "four corner holes", not four lines).
        5. Output only the {len(sections)} labeled sections above ({"/".join(sections)}).
           No preamble, no code, no explanation.
        """)
    return body + rules


def build_rewriter_user_message(raw_input: str) -> str:
    return f'Rewrite this user description into the structured prompt format:\n\n"{raw_input}"'


# --- Stage 2: the CAD code generator -----------------------------------------
# Condensed from Zero-to-CAD (Ataei et al., 2026), Appendix D.2.

CODEGEN_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert CAD engineer writing CadQuery, a Python CAD library.
    Generate clean, parametric CadQuery code for the geometry you are given.

    IMPORT RULES — these are the most common failure, read them carefully:
    - The ONLY import you may write is: import cadquery as cq
    - NEVER import OCC, OCP, pythonOCC, TopoDS, BRep, gp_, or any other
      low-level OpenCascade module. They are not available and will crash.
    - NEVER use low-level classes like topods_Wire, BRepBuilderAPI, or gp_Pnt.
      Everything must be built with the high-level cq.Workplane fluent API.

    GEOMETRY RULES:
    - Build with the fluent API: cq.Workplane("XY").box(...).faces(">Z").hole(...)
    - Before .extrude() or .cutThruAll(), the sketch MUST be a closed profile.
      A common crash is "No pending wires present", which means you called
      .extrude() with nothing closed on the workplane. Prefer the built-in
      primitives (.box, .cylinder, .sphere) and .hole/.cboreHole over hand-built
      sketches whenever they can express the shape.
    - Select faces/edges with string selectors: .faces(">Z"), .edges("|Z").
    - Every 3D point must be a tuple, never a bare number: use (0, 0, 5), not 5.
      This applies to .center(), .moveTo(), .translate(), .pushPoints(). Passing a
      single number causes "Expected three floats, OCC gp_, or 3-tuple".
    - Apply .fillet()/.chamfer() LAST, and keep every radius at most one third of
      the smallest adjacent dimension. An oversized radius causes
      "StdFail_NotDone: BRep_API: command not done". If in doubt, use a smaller
      radius or omit the fillet — a valid part without a fillet beats a crash.
    - Keep the part simple enough to actually build. Prefer a few well-formed
      features over many fragile ones.

    OUTPUT RULES:
    - Put numeric parameters in named variables at the top.
    - Assign the final solid to a variable named exactly: result
    - No export statements. No comments. No explanation.
    - Output ONE Python code block and nothing else.

    Here is the expected style:

    ```python
    import cadquery as cq

    plate_length = 60.0
    plate_width = 40.0
    plate_thickness = 6.0
    hole_diameter = 5.0
    hole_inset = 8.0
    corner_fillet = 4.0

    result = (
        cq.Workplane("XY")
        .box(plate_length, plate_width, plate_thickness)
        .edges("|Z")
        .fillet(corner_fillet)
        .faces(">Z")
        .workplane()
        .rect(plate_length - 2 * hole_inset, plate_width - 2 * hole_inset,
              forConstruction=True)
        .vertices()
        .hole(hole_diameter)
    )
    ```

    When shown an execution error, fix that specific problem. Do not delete
    features to make the error disappear, and do not switch to low-level
    OpenCascade calls.
    """)


def build_codegen_user_message(rewritten_prompt: str) -> str:
    return f"Generate CadQuery code for the following part description:\n\n{rewritten_prompt}"


# --- Few-shot demonstrations for the generator -------------------------------
# Spec -> correct-code pairs, prepended to the generation request as prior chat
# turns so the model has a concrete pattern to imitate. Three deliberately
# different part shapes (a drilled plate, a bored cylinder, a shelled box) so
# the model does not over-imitate one primitive.
#
# These are FIXED and IDENTICAL in both ablation conditions (full / no_function).
# They sit in the generation stage, downstream of the context window under test,
# so they cannot create a difference between conditions — they only raise the
# floor for both. Every example below is verified to execute in CadQuery.

_FEWSHOT_EXAMPLES = [
    (
        "[OVERVIEW]\nA flat rectangular plate with a hole near each corner.\n\n"
        "[DETAILS]\n- extrude - rectangular base plate - thin\n"
        "- hole - one hole near each corner - four - small",
        'import cadquery as cq\n\n'
        'plate_length = 60.0\n'
        'plate_width = 40.0\n'
        'plate_thickness = 6.0\n'
        'hole_diameter = 6.0\n'
        'hole_inset = 8.0\n\n'
        'result = (\n'
        '    cq.Workplane("XY")\n'
        '    .box(plate_length, plate_width, plate_thickness)\n'
        '    .faces(">Z")\n'
        '    .workplane()\n'
        '    .rect(plate_length - 2 * hole_inset, plate_width - 2 * hole_inset,\n'
        '          forConstruction=True)\n'
        '    .vertices()\n'
        '    .hole(hole_diameter)\n'
        ')',
    ),
    (
        "[OVERVIEW]\nA short cylinder with a concentric hole through it, like a bushing.\n\n"
        "[DETAILS]\n- extrude - solid cylinder - medium\n"
        "- hole - concentric through hole - centered - medium",
        'import cadquery as cq\n\n'
        'outer_diameter = 30.0\n'
        'inner_diameter = 16.0\n'
        'height = 25.0\n\n'
        'result = (\n'
        '    cq.Workplane("XY")\n'
        '    .circle(outer_diameter / 2)\n'
        '    .extrude(height)\n'
        '    .faces(">Z")\n'
        '    .workplane()\n'
        '    .hole(inner_diameter)\n'
        ')',
    ),
    (
        "[OVERVIEW]\nA rectangular box hollowed out with an open top, like a container.\n\n"
        "[DETAILS]\n- extrude - rectangular body - base solid\n"
        "- shell - hollow interior - open at top face - thin walls",
        'import cadquery as cq\n\n'
        'length = 50.0\n'
        'width = 35.0\n'
        'height = 25.0\n'
        'wall_thickness = 2.5\n\n'
        'result = (\n'
        '    cq.Workplane("XY")\n'
        '    .box(length, width, height)\n'
        '    .faces(">Z")\n'
        '    .shell(-wall_thickness)\n'
        ')',
    ),
]


def build_codegen_messages(rewritten_prompt: str, system_prompt: str = None) -> list:
    """Assemble the full generation-stage message list, with few-shot demos.

    Returns: [system, (demo spec, demo code) * N, real request]. The demos are
    the same every call and in every ablation condition.
    """
    messages = [{"role": "system", "content": system_prompt or CODEGEN_SYSTEM_PROMPT}]
    for spec, code in _FEWSHOT_EXAMPLES:
        messages.append({"role": "user", "content": build_codegen_user_message(spec)})
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
    messages.append({"role": "user", "content": build_codegen_user_message(rewritten_prompt)})
    return messages