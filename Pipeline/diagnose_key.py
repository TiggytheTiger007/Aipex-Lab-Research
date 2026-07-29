#!/usr/bin/env python3
"""
diagnose_key.py — figure out why the Gemini API key is being rejected.

Run it in the same folder as run_experiment.py:

    python diagnose_key.py

It never prints your full key — only its length and first/last few characters.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mask(k: str) -> str:
    if not k:
        return "(empty)"
    if len(k) <= 12:
        return f"{k[:2]}...{k[-2:]}  (suspiciously short)"
    return f"{k[:6]}...{k[-4:]}"


print("=" * 64)
print("GEMINI API KEY DIAGNOSTIC")
print("=" * 64)

# --- 1. Is there a shell export that will override config.txt? -------------
shell_key = os.environ.get("GEMINI_API_KEY")
print("\n[1] Shell environment (this WINS over config.txt)")
if shell_key:
    print(f"    GEMINI_API_KEY is exported in your shell: {mask(shell_key)}  len={len(shell_key)}")
    print("    >>> If this is stale or wrong, editing config.txt will NOT help.")
    print("    >>> Clear it with:   unset GEMINI_API_KEY")
else:
    print("    Not set in shell. Good — config.txt will be used.")

# --- 2. Does config.txt exist and parse? -----------------------------------
print("\n[2] config.txt")
cfg = HERE / "config.txt"
if not cfg.exists():
    print(f"    MISSING: {cfg}")
    print("    >>> Fix:   cp config.example.txt config.txt")
    print("    >>> Then edit config.txt and paste your key.")
    sys.exit(1)

print(f"    Found: {cfg}")
parsed = {}
for i, line in enumerate(cfg.read_text().splitlines(), 1):
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    if "=" not in s:
        print(f"    line {i}: no '=' sign, ignored -> {s!r}")
        continue
    k, _, v = s.partition("=")
    parsed[k.strip()] = v.strip().strip('"').strip("'")

if "GEMINI_API_KEY" not in parsed:
    print("    GEMINI_API_KEY line not found in config.txt.")
    print(f"    Keys present: {list(parsed)}")
    sys.exit(1)

file_key = parsed["GEMINI_API_KEY"]
print(f"    GEMINI_API_KEY in file: {mask(file_key)}  len={len(file_key)}")

# --- 3. Obvious problems with the key value --------------------------------
print("\n[3] Sanity checks on the key")
problems = []
if file_key.startswith("paste_your"):
    problems.append("still the placeholder text from config.example.txt")
if not file_key:
    problems.append("empty value")
if file_key != file_key.strip():
    problems.append("leading/trailing whitespace")
if " " in file_key:
    problems.append("contains a space — likely a broken copy/paste")
if not (file_key.startswith("AQ") or file_key.startswith("AIza")):
    problems.append("does not start with 'AQ' (new auth key) or 'AIza' (legacy key)")
if len(file_key) < 20:
    problems.append(f"only {len(file_key)} chars — looks truncated")

if file_key.startswith("AQ"):
    print("    NOTE: 'AQ' prefix = new-style auth key. These are REJECTED by Google's")
    print("          OpenAI-compatibility endpoint but work on the native API, which")
    print("          is what models.py now uses.")
elif file_key.startswith("AIza"):
    print("    NOTE: 'AIza' prefix = legacy Standard key. Google is retiring these;")
    print("          unrestricted ones are already rejected. Consider making a new key.")

if problems:
    for p in problems:
        print(f"    PROBLEM: {p}")
else:
    print("    Format looks plausible.")

# --- 4. Which key would actually be used? ----------------------------------
effective = shell_key or file_key
print("\n[4] The key that will actually be sent")
print(f"    Source: {'SHELL EXPORT' if shell_key else 'config.txt'}")
print(f"    Value:  {mask(effective)}  len={len(effective)}")
if shell_key and shell_key != file_key:
    print("    >>> WARNING: shell and config.txt disagree. The shell one is being used.")

# --- 5. Live test against the API ------------------------------------------
print("\n[5] Live test against the Gemini API (native endpoint)")
import json as _json
import urllib.error
import urllib.request

model = os.environ.get("GEMINI_MODEL", parsed.get("GEMINI_MODEL", "gemini-2.5-flash"))
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
print(f"    Model: {model}")
payload = _json.dumps({"contents": [{"role": "user",
                                     "parts": [{"text": "Reply with the word OK."}]}]}).encode()
req = urllib.request.Request(url, data=payload, method="POST",
                             headers={"Content-Type": "application/json",
                                      "x-goog-api-key": effective})
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = _json.loads(resp.read().decode())
    text = "".join(p.get("text", "")
                   for p in data["candidates"][0]["content"]["parts"])
    print(f"    SUCCESS — the API replied: {text.strip()[:40]!r}")
    print("\n    Your key works. Run the pipeline normally.")
except urllib.error.HTTPError as e:
    detail = e.read().decode(errors="replace")
    print(f"    FAILED: HTTP {e.code}")
    print(f"    {detail[:400]}")
    low = detail.lower()
    print()
    if e.code == 400 and "api key" in low:
        print("    DIAGNOSIS: Google does not recognise this key.")
        print("      - Create a fresh one at https://aistudio.google.com/apikey")
        print("      - Copy the WHOLE key; check for a stale shell export in [1] above")
    elif e.code == 403:
        print("    DIAGNOSIS: key recognised but not authorised.")
        print("      - The Generative Language API may not be enabled on that project,")
        print("        or the key has API restrictions excluding it.")
        print("      - A key created directly in AI Studio usually avoids this.")
    elif e.code == 429:
        print("    DIAGNOSIS: key is VALID; you hit a rate/quota limit. Wait and retry.")
    elif e.code == 404:
        print(f"    DIAGNOSIS: key may be fine, but model {model!r} was not found.")
        print("      - Set GEMINI_MODEL=<name> in config.txt")
        print("      - Current names: https://ai.google.dev/gemini-api/docs/models")
    else:
        print("    DIAGNOSIS: unrecognised error — paste the text above for help.")
except Exception as e:
    print(f"    FAILED: {e}")
print("=" * 64)
