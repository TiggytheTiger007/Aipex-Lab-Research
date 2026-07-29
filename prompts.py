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
    - Apply .fillet()/.chamfer() last, and keep radii small relative to the part
      (a fillet larger than the adjacent wall will fail).

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