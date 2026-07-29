"""
models.py — the two model backends.

  Rewriter  : Gemini, over its free OpenAI-compatible API. Needs no GPU.
  Generator : a local checkpoint on the lab GPU, via transformers.

torch is imported lazily, inside the functions that need it, so that --dry-run
and --check-gpu stay fast and so bootstrap.py always gets to set
CUDA_VISIBLE_DEVICES first.
"""

import os
import sys
import time
from dataclasses import dataclass

import bootstrap  # noqa: F401  — must be imported before torch; sets env vars


@dataclass
class ApiProvider:
    name: str
    base_url: str
    api_key_env: str
    model: str


REWRITER = ApiProvider(
    name="gemini",
    # Native Gemini API, NOT the /openai/ compatibility path — see call_api().
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key_env="GEMINI_API_KEY",
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
)

# HuggingFace id, or a path to an already-downloaded checkpoint for offline use.
GENERATION_MODEL = os.environ.get("LOCAL_MODEL_ID", "deepseek-ai/deepseek-coder-6.7b-instruct")


def require_bootstrap():
    """Guard against import-order mistakes that would silently ignore GPU_DEVICE."""
    if not bootstrap.READY:
        raise RuntimeError("bootstrap.setup() did not run before models was imported.")


# ---------------------------------------------------------------------------
# Rewriter: Gemini API
# ---------------------------------------------------------------------------

def _gemini_request_body(messages: list, temperature: float) -> dict:
    """Convert OpenAI-style messages to Gemini's native request format.

    Gemini takes the system prompt in its own `system_instruction` field rather
    than as a message with role="system", and uses "model" where OpenAI uses
    "assistant".
    """
    body = {"contents": [], "generationConfig": {"temperature": temperature}}
    for m in messages:
        if m["role"] == "system":
            body["system_instruction"] = {"parts": [{"text": m["content"]}]}
        else:
            role = "model" if m["role"] == "assistant" else "user"
            body["contents"].append({"role": role, "parts": [{"text": m["content"]}]})
    return body


def call_api(messages: list, temperature: float = 0.4, max_retries: int = 3) -> str:
    """Call Gemini's native REST endpoint, retrying on transient errors.

    Uses the native API with an `x-goog-api-key` header rather than the
    OpenAI-compatibility layer. The compatibility layer sends the key as a
    `Bearer` token, which the newer auth keys (the ones beginning "AQ") are
    rejected under with a 400 "Please pass a valid API key". The native
    endpoint accepts them. Uses urllib so there is no extra dependency.
    """
    import json as _json
    import urllib.error
    import urllib.request

    api_key = os.environ.get(REWRITER.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing {REWRITER.api_key_env}. Put it in config.txt "
                         f"(see config.example.txt).")

    url = f"{REWRITER.base_url}/models/{REWRITER.model}:generateContent"
    payload = _json.dumps(_gemini_request_body(messages, temperature)).encode()

    last_err = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read().decode())
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"No candidates returned. Response: {str(data)[:300]}")
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            last_err = f"HTTP {e.code}: {detail}"
            # Auth/permission failures will not fix themselves on retry.
            if e.code in (400, 401, 403):
                raise SystemExit(
                    f"\nGemini rejected the request ({last_err}).\n"
                    f"Run `python diagnose_key.py` to check the key.\n")
        except Exception as e:  # noqa: BLE001 — retry on anything transient
            last_err = e
        wait = 2 ** attempt
        print(f"  [{REWRITER.name}] attempt {attempt} failed ({last_err}); "
              f"retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"{REWRITER.name} failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Generator: local GPU
# ---------------------------------------------------------------------------

_model_cache = {}


def load_local_model(model_id: str):
    """Load the checkpoint into VRAM once and reuse it.

    Cached because loading takes minutes; without this it would reload on every
    case and every repair-loop turn.
    """
    require_bootstrap()
    if model_id not in _model_cache:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise SystemExit("Missing dependency. Run: pip install torch transformers accelerate")
        print(f"Loading local model '{model_id}' onto GPU (one-time cost for this run)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16)
        print("Local model loaded.")
        _model_cache[model_id] = (tokenizer, model)
    return _model_cache[model_id]


def call_local(messages: list, temperature: float = 0.6, max_new_tokens: int = 1024,
               model_id: str = None) -> str:
    """Generate a chat completion from the local model."""
    import torch
    tokenizer, model = load_local_model(model_id or GENERATION_MODEL)

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=True, temperature=temperature, top_p=0.9)
    new_tokens = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


def extract_code_block(text: str) -> str:
    """Pull the python block out of a markdown-fenced response."""
    if "```" not in text:
        return text.strip()
    block = text.split("```")[1]
    if block.startswith("python"):
        block = block[len("python"):]
    return block.strip()


def check_gpu() -> None:
    """Report what torch can see. Run before the first real run."""
    try:
        import torch
    except ImportError:
        print("torch is not installed. Run: pip install torch transformers accelerate")
        return
    print(f"torch version:        {torch.__version__}")
    print(f"CUDA available:       {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set — all GPUs visible)')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        print("No GPU visible to torch — generation would run on CPU (very slow) or fail.")