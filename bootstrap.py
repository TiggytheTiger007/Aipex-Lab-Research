"""
bootstrap.py — startup environment setup.

MUST be imported before anything that imports torch or cadquery. run_experiment.py
imports this first, before every other project module, and the other modules assert
that it ran (see models.require_bootstrap).

Three jobs:
  1. Load config.txt into environment variables, so you never type `export`.
  2. Pin CUDA_VISIBLE_DEVICES to the GPU you chose, before torch can grab all of them.
  3. Preload a modern libstdc++ if conda's cadquery and torch disagree about it.
"""

import os
import sys

READY = False          # set True once setup completes; other modules check this
CONFIG_FILE = "config.txt"


def _load_config_file(path: str = CONFIG_FILE) -> None:
    """Read KEY=value lines from config.txt into os.environ.

    Real shell exports win over config.txt, so you can override a single value
    for one run without editing the file.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(config_path):
        return
    for line in open(config_path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _gpu_device_from_argv():
    """Read --gpu-device straight from sys.argv.

    argparse runs too late: by the time it parses, a module further up the import
    chain may already have imported torch, and CUDA_VISIBLE_DEVICES is only read
    at torch import time.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu-device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--gpu-device="):
            return arg.split("=", 1)[1]
    return None


def _pin_gpu() -> None:
    device = _gpu_device_from_argv() or os.environ.get("GPU_DEVICE")
    if device and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = device


def _patch_libstdcxx() -> None:
    """Preload conda's libstdc++ if it is newer than the one torch would pick up.

    On some conda setups, cadquery/OCP ships an older libstdc++ than torch expects
    and the process dies when both are imported. Preloading fixes it, but requires
    restarting the interpreter, so this is deliberately conservative:

      * skipped for --dry-run / --check-gpu, which never load the generation model;
      * skipped if a debugger is attached, since exec() looks like a crash there;
      * guarded by an env var so it can never restart more than once.
    """
    if "--dry-run" in sys.argv or "--check-gpu" in sys.argv:
        return
    if os.environ.get("_CADPIPE_PATCHED") == "1":
        return
    if sys.gettrace() is not None or "debugpy" in sys.modules:
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if not os.path.exists(lib) or os.environ.get("LD_PRELOAD") == lib:
        return

    print("[bootstrap] Restarting once with LD_PRELOAD set, to avoid a known "
          "cadquery/torch libstdc++ crash. This is expected.", file=sys.stderr)
    os.environ["LD_PRELOAD"] = lib
    os.environ["_CADPIPE_PATCHED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def setup() -> None:
    """Run all startup steps. Safe to call more than once."""
    global READY
    if READY:
        return
    _load_config_file()
    _pin_gpu()
    _patch_libstdcxx()
    READY = True


# Runs on import, so `import bootstrap` is all that's needed.
setup()
