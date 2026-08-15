# persona-probes

Do emotion probes trained on a model's default assistant persona survive the
model roleplaying as something else?

## Hypothesis

A linear probe trained on residual-stream activations can read a model's
expressed emotion accurately when the model is speaking in its default
assistant persona. We hypothesise that this probe degrades as the model is
pushed into personas further from that default, and that the degradation is
systematic rather than random — probe accuracy should fall predictably with
distance along an "assistant axis" defined as the difference of means between
default-persona and most-distant-persona activations. If it holds, an emotion
probe validated on assistant-mode outputs cannot be assumed to work on the same
model under a roleplay system prompt.

## Design

**9 conditions × 12 emotions × 30 topics = 3,240 generations.**

The design is **fully crossed**: the same 30 topics appear in every (persona,
emotion) cell. If topics varied by cell, a drop in transfer accuracy could be
topic shift rather than persona shift and the two could not be separated. The 30
are a seeded random sample of the 100 (`config.TOPIC_SAMPLE_SEED`), not curated —
some pairings are awkward on purpose, which forces the emotion to be carried by
the persona's framing rather than supplied by the scenario.

Eight of the nine conditions are personas, ordered by *intuited* distance from
the default assistant. Stage 6 measures the real distance and is allowed to
disagree; that comparison is one of the results.

The ninth, `bare_template`, is a **control, not a persona**. It sends no system
message, so Qwen's chat template inserts its own — verified as *"You are Qwen,
created by Alibaba Cloud. You are a helpful assistant."* That string is short and
names the vendor, while the written personas are ~50 words. Anchoring the axis
there would risk measuring short-prompt vs long-prompt instead of assistant vs
character.

So the axis is anchored on **`default_assistant`**, an explicit bland assistant
prompt length- and structure-matched to the other personas. `bare_template` is
extracted and projected like everything else, and marked distinctly in every
figure, but never defines the axis:

- lands **near** `default_assistant` → prompt length is not driving the axis.
- lands **far** → the axis is picking up prompt length or template effects, and
  there is a real confound in the headline figure.

Either outcome is a result, for the cost of 360 extra generations.

Every persona prompt uses the same skeleton and the same closing sentence, and
they are matched to ~50 words: identity and voice vary, prose style and
instruction structure do not. Do not lengthen or restructure one for
readability — unmatched length reintroduces exactly the confound above.

> **Anchor length.** `default_assistant` anchors the axis, so it must not be the
> shortest prompt in the set — otherwise "distance from the anchor" is partly
> "longer prompt", the same confound `bare_template` exists to detect, applied to
> the anchor itself. It sits at 48 words inside a 44–50 band. `python personas.py`
> prints the spread; `01_data.ipynb` prints it again at the top.

## Layout

```
config.py                   all paths, models, layers, emotions. Import first.
preflight.py                stage 0 — GPU, VRAM, disk, packages, API key
scenarios.py                stage 2 — 100 topics from HF, and the fixed 30-topic subset
check_dataset_emotions.py   one-off — do our 12 emotions appear in the dataset?
personas.py                 stage 3 — 9 system prompts (data only)
gen_data.py                 stage 4 — OpenRouter generation, content-hash cached
extract.py                  stage 5 — forward hooks, mean-pooled activations
assistant_axis.py           stage 6 — difference-of-means axis + ranking
probes.py                   stage 7 — sklearn logistic regression, 12-way
plot.py                     stage 8 — the figures (+ a .csv twin for each)
notebooks/                  read-only views over each stage's outputs
data/                       topics.json, generations.jsonl, gen_cache/
outputs/                    activations, axis, probe results, figures/
```

Every script runs standalone from the CLI, prints what it is doing, and takes
`--limit N` for a smoke test that exercises the whole path on a handful of
samples before you spend API credits or GPU hours.

## Setup

```bash
cd /workspace/persona-probes
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
```

Versions in `requirements.txt` are **exact pins**, verified working together on
this pod. Two are load-bearing:

- **`transformers==4.57.6`, not 5.x.** transformers ≥ 5 requires torch ≥ 2.5 and
  silently disables PyTorch entirely under the image's torch 2.4.1 —
  `AutoModelForCausalLM` raises `ImportError` at import, with the real cause only
  in a startup log line. Pinning 4.x keeps the image's torch usable, so a fresh
  pod is `pip install -r requirements.txt` and nothing else. Moving to 5.x means
  upgrading torch first and re-verifying `extract.py`'s response-token slicing.
- **`jinja2==3.1.6`.** The image ships 3.1.3; pandas 3.x needs ≥ 3.1.5 for
  `DataFrame.style` and otherwise reports jinja2 as not installed.

torch itself is deliberately unpinned — it ships with the image, CUDA-matched.

### Where the weights go

Qwen2.5-32B-Instruct in bf16 is ~65 GB and must exist as files on a filesystem:
`transformers` mmaps safetensors, and there is no path that streams weights from
the hub straight into VRAM. Default is `/workspace/hf_home`, which needs the
volume resized past ~100 GB to fit.

If you would rather not resize, this pod has 2 TB of RAM and a 117 GB `/dev/shm`
tmpfs:

```bash
HF_HOME=/dev/shm/hf_home python extract.py   # RAM-backed; re-downloads ~65GB after a restart
```

`preflight.py` checks free space on whichever path `HF_HOME` actually points at,
and prints `/workspace` and `/dev/shm` side by side.

## Reproduction

Run in order. Each stage writes to disk and is independently rerunnable.

```bash
# 0. does this pod work at all — fails in ~3s, not at hour six
python preflight.py

# 1. what am I configured to run
python config.py
python personas.py
python scenarios.py --sample            # the fixed 30 topics, seeded
python check_dataset_emotions.py        # expect a report, not silence

# 2. smoke test the whole pipeline before spending anything.
#    PP_OUT sends stage 5-8 outputs somewhere harmless, so a smoke run can
#    never overwrite a real extraction (extract.py warns if you forget).
python gen_data.py --dry-run            # prints an example request + cost ceiling
python gen_data.py --limit 8
PP_OUT=outputs/smoke PP_PRESET=0.5b python extract.py --limit 8

# 3. the real run
python gen_data.py                      # 3,240 OpenRouter calls, resumable
python extract.py                        # local GPU, hooks layers 36-40
python assistant_axis.py
python probes.py
python plot.py                          # -> outputs/figures/
```

`gen_data.py` caches each completed call under a hash of its exact request
content, so a rerun refetches only what is missing and editing one persona's
prompt invalidates only that persona. A run that dies at 90% costs the last 10%.

The serving provider is **pinned** (`config.GENERATOR_PROVIDER`) and recorded on
every row, and dispatch order is **shuffled** (`config.DISPATCH_SEED`). Both
exist because the grid is persona-major, so an API condition that varies with
wall-clock lands on one persona's contiguous block and is indistinguishable from
a persona effect. The first run demonstrated this: 192 rows lost to a provider
that rejects chat-completions requests, concentrated in 5 of the 9 personas (74
in `close_friend`, 0 in the first three). The provider pin is part of the request
body and therefore part of the cache key — a corpus generated under free routing
and one generated on a pinned provider are not interchangeable and will not blend.

Model swap is one variable:

```bash
PP_PRESET=7b   python extract.py   # Qwen2.5-7B-Instruct, layers 14-18
PP_PRESET=0.5b python extract.py   # plumbing only — never report results from this
```

Layer indices and hidden dims come from the loaded model config, never from
hardcoded constants, so a swap cannot silently reshape the data.

## Method notes

- **Pooling.** Mean over assistant *response* tokens only; the system prompt and
  user turn are excluded. The response span is located by measuring the chat
  template's own suffix rather than assuming it. No per-token activations kept.
- **Replay fidelity.** Each message is replayed inside its own persona's
  template, using the same prompt builder `gen_data.py` used, so the system
  prompt a message was generated under is the one it is hooked under.
- **Splits are by topic, not by row, and cross-validated.** The 30 topics are
  cut into `config.N_FOLDS` = 5 folds of 6 and each fold is held out in turn, so
  every topic is tested exactly once and no probe is tested on a topic it trained
  on. A single 75/25 split left ~6–10 test rows per class per persona for a 12-way
  problem — too few to read the headline relationship off. Reported accuracies are
  means over folds; `probe_results.json` keeps the per-fold values, SD and SEM.
- **Transfer is scored on the same held-out topics** as the default-persona
  test, fold by fold, so (a) and (b) differ only in persona. The all-topics
  variant is reported alongside as a secondary, systematically more optimistic
  number.
- **Error bars are ±1 SD across the 5 folds**, in both the headline figure and
  the layer figure. The folds share training rows, so they show the spread of the
  estimate, not a confidence interval.
- **Layer choice.** The best layer is selected on default-persona accuracy
  alone. Selecting it on transfer accuracy would select for the result being
  measured.
- **English only, by instruction and then by rejection sampling.** The
  generation prompt carries one identical `Write in English only.` line in every
  cell. Qwen code-switches into Chinese at temperature 1.0 anyway — 1.67% of run
  3 carried the instruction and still came back with CJK — and the language
  direction in activation space dwarfs the emotion directions, so even a 1–2%
  rate is a strong outlier. Worse, the rate rose with distance from the
  assistant, which is the x-axis of the headline figure. So a draw containing
  CJK is **rejected and redrawn** (`config.MAX_CJK_REDRAWS`), uniformly in every
  cell, making the corpus a sample from P(text | English) identically for every
  persona; the redraw reuses the same request body, so cache keys do not move. A
  row that exhausts the cap is cached and flagged rather than dropped. Rows
  carry `cjk_redraws` / `cjk_residue`, `gen_data.py` prints redraws and CJK per
  persona, and `01_data.ipynb` **raises** if the corpus is not 100% English.
  Never measure this with `str.split()`: CJK has no spaces, so a fully-Chinese
  message counts as 3 "words" and every length check goes blind to it.
- **Emotion leakage is reported, never filtered.** `gen_data.py` and
  `01_data.ipynb` count generations that name their target emotion verbatim,
  **broken down by persona** as well as overall: if leakage varies systematically
  across personas it varies along the x-axis of the headline figure, and the
  result is confounded. Synonyms are not detected, so it is a floor. Whether a
  leaked row is still a valid sample is a call for you, not for the script.

## Notebooks

Read-only views over what the scripts already wrote — nothing is computed in a
notebook that a script does not also compute, so results never depend on cell
execution order.

| notebook | needs |
| --- | --- |
| `notebooks/00_setup.ipynb` | nothing — config, personas, topics, GPU |
| `notebooks/01_data.ipynb` | `gen_data.py` — coverage, lengths, leakage, the text itself |
| `notebooks/02_results.ipynb` | `probes.py`, `assistant_axis.py`, `plot.py` — headline figures, control verdict, intuited vs measured order |

Open them in VS Code Remote and pick the pod's Python interpreter, or run
`jupyter lab --no-browser --port 8888` and forward the port.

## Dataset

Topic strings come from the `topic` column of `expression/stories.parquet` in
[ryancodrai/emotion-probes](https://huggingface.co/datasets/ryancodrai/emotion-probes),
deduplicated. We reuse **only** the topic strings — no stories, labels, or
activations from that dataset enter this pipeline. `scenarios.py` prints which
source it used and falls back to a hardcoded list if the schema has changed
(that fallback is deliberately never cached, so a transient hub failure cannot
poison later runs).

As observed on 2026-08-15: 205,200 rows, 100 unique topics, **171 distinct
emotion values** at 1,200 rows each (171 × 1200 = 205,200 exactly). All 12 of
our emotions appear verbatim — `check_dataset_emotions.py` reverifies this and
exits non-zero on any mismatch.

The dataset has **no stated license**. Reuse of the topic strings here is on
that basis, for non-commercial research; if you plan to redistribute anything
derived from it, check with the dataset author first.

```bibtex
@misc{codrai_emotion_probes,
  author    = {Codrai, Ryan},
  title     = {emotion-probes},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/8303},
  url       = {https://huggingface.co/datasets/ryancodrai/emotion-probes}
}
```

Author name and year are inferred from the hub namespace — check the dataset
card and correct them before this goes anywhere citable.
