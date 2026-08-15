"""Central configuration for the persona-probes experiment.

Import this module FIRST in every script. It sets HF_HOME before
transformers / huggingface_hub have a chance to read the environment, which is
what keeps the ~65GB of weights on the persistent /workspace volume instead of
the ephemeral container disk.

    import config  # noqa: F401  -- must come before transformers
    from transformers import AutoModelForCausalLM
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths.  Everything lives under /workspace because the container disk is
# ephemeral on Runpod.
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from a gitignored .env, without adding a dependency.

    Only stage 4 needs a secret (OPENROUTER_API_KEY).  A real environment
    variable always wins, so `export`ing on the pod behaves exactly as before
    and this is purely a local-shell convenience -- on Windows `setx` does not
    affect the current process, which makes an exported key easy to lose.
    """
    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# /workspace is the Runpod persistent volume.  Off the pod it does not exist,
# and blindly using it means importing config creates C:\workspace (or /workspace
# on a laptop) as a side effect of `python personas.py`.  Stage 4 and stages 6-8
# run fine anywhere, so fall back to the repo itself -- hf_home/ is already
# gitignored.  PP_WORKSPACE still overrides both.
_ws_env = os.environ.get("PP_WORKSPACE")
if _ws_env:
    WORKSPACE = Path(_ws_env)
else:
    _pod_workspace = Path("/workspace")
    WORKSPACE = _pod_workspace if _pod_workspace.is_dir() else REPO_ROOT

# Where the ~65GB of weights land.  Set before any HF import happens.
#
# This pod has a 20GB /workspace but 2TB of RAM and a 117GB /dev/shm tmpfs, so
# there are two workable choices and they trade persistence against space:
#
#   /workspace/hf_home   persistent across pod restarts, but only if you resize
#                        the volume past ~100GB in the Runpod UI first.
#   /dev/shm/hf_home     RAM-backed, fastest load, no resize needed -- but the
#                        cache is GONE on restart (re-download ~65GB) and it
#                        occupies that much RAM for as long as it exists.
#
# Override per-run without editing this file:  HF_HOME=/dev/shm/hf_home python extract.py
HF_HOME = Path(os.environ.get("HF_HOME") or (WORKSPACE / "hf_home"))
os.environ["HF_HOME"] = str(HF_HOME)
HF_HOME.mkdir(parents=True, exist_ok=True)

DATA_DIR = REPO_ROOT / "data"

# Overridable so a smoke run cannot clobber a real one. Stage 5 on the 32B is
# tens of minutes of GPU time; `extract.py --limit 8` would otherwise overwrite
# activations.npy with 8 rows. Same variable works for every stage at once:
#     PP_OUT=outputs/smoke PP_PRESET=0.5b python extract.py --limit 8
OUT_DIR = Path(os.environ.get("PP_OUT") or (REPO_ROOT / "outputs"))
if not OUT_DIR.is_absolute():
    OUT_DIR = REPO_ROOT / OUT_DIR
FIG_DIR = OUT_DIR / "figures"
GEN_CACHE_DIR = DATA_DIR / "gen_cache"  # one JSON file per completed API call

for _d in (DATA_DIR, OUT_DIR, FIG_DIR, GEN_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Stage outputs.
TOPICS_JSON = DATA_DIR / "topics.json"  # stage 2
GENERATIONS_JSONL = DATA_DIR / "generations.jsonl"  # stage 4
ACTIVATIONS_NPY = OUT_DIR / "activations.npy"  # stage 5
ACTIVATIONS_META = OUT_DIR / "activations_meta.parquet"  # stage 5
ACTIVATIONS_INFO = OUT_DIR / "activations_info.json"  # stage 5
AXIS_NPY = OUT_DIR / "assistant_axis.npy"  # stage 6
AXIS_RANKING_JSON = OUT_DIR / "axis_ranking.json"  # stage 6
PROBE_RESULTS_JSON = OUT_DIR / "probe_results.json"  # stage 7

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

# Data generation.  Goes through the OpenRouter API, no GPU involved.
GENERATOR_MODEL = "qwen/qwen-2.5-72b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Rough OpenRouter list price for GENERATOR_MODEL, USD per 1M tokens.  Used
# only for the running cost estimate printed by gen_data.py -- prices move, so
# treat the number as indicative and update it if you care about accuracy.
GENERATOR_PRICE_IN_PER_MTOK = 0.12
GENERATOR_PRICE_OUT_PER_MTOK = 0.39

# The model we actually probe, run locally from HF weights.
#
# `layers` are residual-stream hook points.  `n_layers` / `hidden` are the
# EXPECTED dims, recorded here for documentation and for the sanity check in
# check_model_dims() -- the real values are always read from the loaded model
# config so that swapping presets can never silently reshape the data.
MODEL_PRESETS = {
    "32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "layers": [36, 37, 38, 39, 40],
        "n_layers": 64,
        "hidden": 5120,
        "approx_bf16_gb": 65,
    },
    # One-line swap for smoke tests on a smaller pod: PP_PRESET=7b
    "7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [14, 15, 16, 17, 18],
        "n_layers": 28,
        "hidden": 3584,
        "approx_bf16_gb": 16,
    },
    # Plumbing only.  Same chat template and architecture as the others, so it
    # exercises every code path in extract.py for ~1GB.  Never report results
    # from this preset.
    "0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers": [10, 11, 12, 13, 14],
        "n_layers": 24,
        "hidden": 896,
        "approx_bf16_gb": 1,
    },
}

PRESET = os.environ.get("PP_PRESET", "32b")
_p = MODEL_PRESETS[PRESET]

PROBED_MODEL = _p["hf_id"]
LAYERS = _p["layers"]
EXPECTED_N_LAYERS = _p["n_layers"]
EXPECTED_HIDDEN = _p["hidden"]
APPROX_WEIGHTS_GB = _p["approx_bf16_gb"]

DTYPE = "bfloat16"

# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

EMOTIONS = [
    "happy",
    "inspired",
    "loving",
    "proud",
    "calm",
    "desperate",
    "angry",
    "guilty",
    "sad",
    "afraid",
    "nervous",
    "surprised",
]

N_TOPICS = 100  # topic strings pulled from the HF dataset
N_SCENARIOS_PER_CELL = 30  # scenarios per (persona, emotion) cell

# Fully crossed design: THE SAME 30 topics are used in every (persona, emotion)
# cell.  If topics varied by cell, a drop in transfer accuracy could be topic
# shift rather than persona shift and the two could not be separated.  Topic
# coverage buys nothing here because no claim is made about topic generality.
#
# The 30 are a random sample of the 100 under this seed -- not hand-curated.
# Some pairings are awkward on purpose ("loving" with a topic about finding
# cash); that forces the emotion to be carried by the persona's framing rather
# than supplied by the scenario.
TOPIC_SAMPLE_SEED = 0
TARGET_WORDS = (100, 150)
GEN_MAX_TOKENS = 300
GEN_TEMPERATURE = 1.0

# Rejection sampling on code-switching (see CLAUDE.md decision 9).
#
# Qwen code-switches into Chinese at temperature 1.0 and the English-only
# instruction did not stop it: 54 rows (1.67%) in run 3, and -- the part that
# matters -- at a rate that RISES with distance from the assistant, 0.3% on
# close_friend to 3.6% on dream_logic.  The language direction in activation
# space dwarfs the emotion directions, so those rows are strong outliers lying
# along the x-axis of the headline figure.
#
# So a draw containing CJK is rejected and redrawn, up to this many times, as a
# rule applied UNIFORMLY to every cell.  The corpus is then a sample from
# P(text | English) identically for every persona.  This does condition on the
# output; the defence is that the rule is uniform across cells rather than
# applied post hoc to a finished corpus.
#
# The redraw does not change the request body -- temperature 1.0 resamples on
# its own -- so cache keys do not move and an unchanged rerun still costs zero.
# Never add a seed or nonce to force variation: that changes every hash and
# invalidates the entire cache.
#
# 8 is a safety net, not an expected path: at the worst persona's 3.6% rate,
# eight consecutive CJK draws has probability ~3e-12.  A row that exhausts the
# cap is cached anyway, flagged, and reported -- an uncached row would be
# silently omitted from generations.jsonl and would break the crossed design,
# which is exactly what "counted, never dropped" exists to prevent.
MAX_CJK_REDRAWS = 8

# Pin the serving provider.  OpenRouter routes a model across several providers
# that differ in quantization and serving stack, and it re-routes freely during a
# run.  Because the grid is built persona-major, a provider that appears or
# breaks mid-run lands on a contiguous block of ONE persona -- an API condition
# that maps straight onto the experiment's x-axis.  That is the same class of
# confound as per-persona emotion leakage, so it is pinned, not left to chance.
#
# As probed on 2026-08-15, qwen-2.5-72b-instruct has exactly two endpoints:
#   DeepInfra  fp8   works
#   Novita     bf16  rejects chat-completions requests outright ("does not
#                    support endpoint: completions") -- the cause of the 192
#                    failures in the first run, all of them clustered in 5 of
#                    the 9 personas.
# So the corpus is generated at fp8; there is no working bf16 endpoint to pick.
# Set to None to restore free routing (and re-introduce the confound).
GENERATOR_PROVIDER = "DeepInfra"
GENERATOR_ALLOW_FALLBACKS = False  # fail loudly rather than silently re-route

# Dispatch order is shuffled under this seed before any call is made, so that a
# time-varying API condition spreads across personas as noise instead of hitting
# one persona's block.  Cache keys are content-based, so shuffling changes
# nothing on disk and generations.jsonl is still written in canonical grid order.
DISPATCH_SEED = 0

# API politeness / robustness (stage 4).
CONCURRENCY = 8
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 120

# Extraction (stage 5).
EXTRACT_BATCH_SIZE = 8
MAX_SEQ_LEN = 1024  # generations are ~150 words; this is a guard, not a target

# Probing (stage 7).
# K-fold cross-validation over TOPICS, not rows.  30 topics / 5 folds = 6 topics
# held out per fold, and every topic is tested exactly once across the run.  A
# single 75/25 split left only ~6-10 test rows per class per persona for a 12-way
# problem -- too noisy to read the headline relationship off.  Folds are cut
# under RANDOM_SEED, so the same topics land in the same fold on every rerun.
N_FOLDS = 5
RANDOM_SEED = 0
LOGREG_MAX_ITER = 2000
LOGREG_C = 1.0

# HF dataset we borrow topic strings from.
TOPICS_DATASET = "ryancodrai/emotion-probes"
TOPICS_DATASET_FILE = "expression/stories.parquet"
TOPICS_COLUMN = "topic"
EMOTION_COLUMN = "emotion"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def check_model_dims(model_config) -> tuple[int, int]:
    """Read the true dims off a loaded HF config and validate LAYERS.

    Returns (num_hidden_layers, hidden_size).  Raises if a configured hook
    layer does not exist on the model that was actually loaded.
    """
    n_layers = model_config.num_hidden_layers
    hidden = model_config.hidden_size

    bad = [i for i in LAYERS if not (0 <= i < n_layers)]
    if bad:
        raise ValueError(
            f"config.LAYERS contains out-of-range layers {bad} for "
            f"{PROBED_MODEL}, which has {n_layers} layers. "
            f"Fix MODEL_PRESETS['{PRESET}']['layers']."
        )
    if n_layers != EXPECTED_N_LAYERS or hidden != EXPECTED_HIDDEN:
        print(
            f"[config] NOTE: loaded model reports {n_layers} layers / hidden "
            f"{hidden}, preset '{PRESET}' expected {EXPECTED_N_LAYERS} / "
            f"{EXPECTED_HIDDEN}. Using the loaded values."
        )
    return n_layers, hidden


def summary() -> str:
    import personas  # local import: keeps config free of import cycles

    n = len(personas.PERSONAS) * len(EMOTIONS) * N_SCENARIOS_PER_CELL
    return (
        f"preset={PRESET}  probed={PROBED_MODEL}  layers={LAYERS}  dtype={DTYPE}\n"
        f"generator={GENERATOR_MODEL}\n"
        f"{len(personas.PERSONAS)} personas x {len(EMOTIONS)} emotions x "
        f"{N_SCENARIOS_PER_CELL} topics = {n} generations\n"
        f"axis anchored on '{personas.DEFAULT_PERSONA_ID}' -> "
        f"'{personas.MOST_DISTANT_PERSONA_ID}', control '{personas.CONTROL_PERSONA_ID}'\n"
        f"HF_HOME={HF_HOME}\n"
        f"data={DATA_DIR}\noutputs={OUT_DIR}"
    )


if __name__ == "__main__":
    print(summary())
