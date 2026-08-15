"""Stage 0: fail in 30 seconds instead of at hour six.

Checks the things that actually stop this pipeline: CUDA, VRAM, free space on
whichever path HF_HOME points at, the Python packages the later stages import,
and the OpenRouter key.  Exits non-zero if anything is a hard blocker.

    python preflight.py
"""

import importlib
import os
import shutil
import sys
import time
from pathlib import Path

import config

BLOCKERS: list[str] = []
WARNINGS: list[str] = []


def gb(n_bytes: float) -> float:
    return n_bytes / 1024**3


def block(msg: str) -> None:
    BLOCKERS.append(msg)
    print(f"  BLOCKER: {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  WARNING: {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_torch_cuda() -> None:
    section("torch / CUDA")
    import torch

    print(f"  torch           {torch.__version__}")
    print(f"  built for CUDA  {torch.version.cuda}")
    print(f"  cuda available  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        block("CUDA is not available -- extract.py cannot run")
        return

    n = torch.cuda.device_count()
    print(f"  device count    {n}")

    total_vram = 0.0
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        free_b, total_b = torch.cuda.mem_get_info(i)
        total_vram += gb(free_b)
        print(
            f"  [{i}] {props.name}  sm_{props.major}{props.minor}  "
            f"{gb(total_b):.1f}GB total, {gb(free_b):.1f}GB free"
        )
        if props.major < 8:
            warn(
                f"device {i} is sm_{props.major}{props.minor}; bf16 needs "
                f"sm_80+ (Ampere or newer)"
            )

    # bf16 weights plus activations and KV cache for a batch of short sequences.
    need_vram = config.APPROX_WEIGHTS_GB * 1.1
    print(f"\n  need ~{need_vram:.0f}GB free VRAM for {config.PROBED_MODEL} in {config.DTYPE}")
    if total_vram < need_vram:
        block(
            f"only {total_vram:.1f}GB free VRAM across {n} device(s), need "
            f"~{need_vram:.0f}GB -- the model will not fit. Use a bigger pod, "
            f"or run the 7B preset: PP_PRESET=7b"
        )


def check_storage() -> None:
    section("storage")
    need_gb = config.APPROX_WEIGHTS_GB * 1.6  # weights + partial downloads + tokenizer

    # HF_HOME may not exist yet on a fresh pod; walk up to a real parent.
    probe = config.HF_HOME
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)

    print(f"  HF_HOME         {config.HF_HOME}")
    print(
        f"  its filesystem  {gb(usage.total):.1f}GB total, "
        f"{gb(usage.free):.1f}GB free"
    )
    print(f"  need            ~{need_gb:.0f}GB for {config.PROBED_MODEL} weights")

    for label, path in (("workspace", config.WORKSPACE), ("/dev/shm", Path("/dev/shm"))):
        if path.exists():
            u = shutil.disk_usage(path)
            print(f"  {label:<15} {gb(u.free):.1f}GB free of {gb(u.total):.1f}GB")

    if gb(usage.free) < need_gb:
        shm = Path("/dev/shm")
        hint = ""
        if shm.exists() and gb(shutil.disk_usage(shm).free) >= need_gb:
            hint = (
                " -- /dev/shm has room: run with HF_HOME=/dev/shm/hf_home "
                "(RAM-backed, lost on pod restart), or resize the /workspace "
                "volume in the Runpod UI to keep the cache persistent"
            )
        block(
            f"{gb(usage.free):.1f}GB free where HF_HOME points, need "
            f"~{need_gb:.0f}GB{hint}"
        )

    # Repo outputs are small (activations are mean-pooled), but not zero.
    u = shutil.disk_usage(config.REPO_ROOT)
    if gb(u.free) < 5:
        warn(f"only {gb(u.free):.1f}GB free where the repo lives ({config.REPO_ROOT})")


def check_ram() -> None:
    section("host RAM")
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = float(rest.strip().split()[0]) / 1024**2  # kB -> GB
        print(f"  total           {info['MemTotal']:.0f}GB")
        print(f"  available       {info['MemAvailable']:.0f}GB")
        if str(config.HF_HOME).startswith("/dev/shm"):
            print(
                f"  NOTE: HF_HOME is on tmpfs, so ~{config.APPROX_WEIGHTS_GB}GB of "
                f"RAM will be held by the weight cache."
            )
        if info["MemAvailable"] < config.APPROX_WEIGHTS_GB:
            warn(
                f"{info['MemAvailable']:.0f}GB RAM available, less than the "
                f"~{config.APPROX_WEIGHTS_GB}GB checkpoint"
            )
    except Exception as exc:  # noqa: BLE001
        warn(f"could not read /proc/meminfo: {exc}")


def check_packages() -> None:
    section("packages")
    required = {
        "transformers": "extract.py",
        "accelerate": "extract.py (device_map)",
        "huggingface_hub": "scenarios.py, extract.py",
        "pandas": "scenarios.py, extract.py, probes.py",
        "pyarrow": "parquet metadata",
        "numpy": "everything",
        "sklearn": "probes.py",
        "matplotlib": "plot.py",
        "httpx": "gen_data.py",
        "tqdm": "progress bars",
    }
    for mod, used_by in required.items():
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "?")
            print(f"  OK    {mod:<18} {version}")
        except ImportError:
            block(f"{mod} not installed (needed by {used_by}) -- pip install -r requirements.txt")


def check_api_key() -> None:
    section("OpenRouter")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        warn(
            "OPENROUTER_API_KEY is not set -- gen_data.py will fail "
            "(stages 5-8 do not need it)"
        )
    else:
        print(f"  OPENROUTER_API_KEY set ({len(key)} chars, ...{key[-4:]})")
    print(f"  generator       {config.GENERATOR_MODEL}")


def main() -> int:
    t0 = time.time()
    print("=== config ===")
    print("  " + config.summary().replace("\n", "\n  "))

    check_torch_cuda()
    check_storage()
    check_ram()
    check_packages()
    check_api_key()

    print(f"\n=== summary ({time.time() - t0:.1f}s) ===")
    if BLOCKERS:
        print(f"\n{'*' * 70}")
        print(f"* {len(BLOCKERS)} BLOCKER(S) -- the pipeline will not complete as configured:")
        for b in BLOCKERS:
            print(f"*   - {b}")
        print(f"{'*' * 70}\n")
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"  - {w}")
    if not BLOCKERS and not WARNINGS:
        print("all clear.")
    elif not BLOCKERS:
        print("\nno blockers.")
    return 1 if BLOCKERS else 0


if __name__ == "__main__":
    sys.exit(main())
