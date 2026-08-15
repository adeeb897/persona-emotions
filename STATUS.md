# STATUS

As of 2026-08-15. Read this first if you are picking the project up cold; it is
the state of play, not the design. `README.md` is the design, `CLAUDE.md` is the
list of decisions that must not drift.

## Where things stand

Stages 0–4 are **done and committed**. Stages 5–8 have **never been run on real
data** — they are verified end to end on synthetic activations and on the 0.5b
preset, but no real activations exist yet.

There is **one open decision** blocking extraction. See below.

## What is in the repo

Committed, including things a `.gitignore` would normally exclude, because they
cannot be reproduced by rerunning:

- `data/generations.jsonl` — the corpus, 3,240 rows. **At temperature 1.0 a
  rerun produces a different corpus, not the same one.** This is the only
  artifact in the pipeline that rerunning cannot reproduce.
- `data/gen_cache/` — per-call cache keyed by a content hash of the exact
  request. A rerun refetches only what a change actually invalidates. Keeping
  this means an unchanged rerun costs nothing.
- `data/topics.json` — the seeded 30-topic sample, pinned so a hub outage or a
  schema change cannot silently reshape the design.

Not committed: `outputs/` (activations are ~330 MB and are a pure function of
the corpus plus a GPU) and model weights.

## Run history

Three full generation runs. Only run 3 is in the repo; 1 and 2 were discarded,
and their caches lived in pod-local scratch that is now gone.

| run | config | result |
| --- | --- | --- |
| 1 | free routing, persona-major dispatch | **discarded.** 3,048/3,240; 192 failures clustered in 5 of 9 personas — close_friend 74, dream_logic 46, detective 35, feral_animal 22, peasant 15, and **zero** in the first three. Cause: OpenRouter routed some calls to Novita, which rejects chat-completions requests outright. Provider was not logged, so the successful rows could not be audited retroactively either. |
| 2 | provider pinned to DeepInfra, dispatch shuffled, provider logged | **discarded.** 3,240/3,240 complete; 3 failures, all HTTP 429, refetched. Leakage 0.31%. **CJK 50 rows (1.54%)**, 0.0–3.1% across personas — discovered here. |
| 3 | run 2 + `Write in English only.` in the prompt | **committed.** 3,240/3,240; 17 failures, all HTTP 429, refetched. Leakage 8 rows (0.25%), 0.0–1.1% spread. **CJK 54 rows (1.67%)**, 0.3–3.6% across personas. |

Verified on run 3: 108 cells at exactly 30 rows, one identical topic set across
every cell (crossed design intact), DeepInfra 3,240/3,240 (the pin held), one
returned model string throughout.

## The open decision: code-switching

**The English-only instruction did not work.** Run 3 (with it) has 54 CJK rows;
run 2 (without it) had 50. That is no improvement — if anything the per-persona
gradient sharpened:

| persona | rate |
| --- | --- |
| close_friend | 0.3% |
| default_assistant | 0.6% |
| peasant | 0.6% |
| bare_template | 0.8% |
| tutor | 1.4% |
| resentful_ai | 2.0% |
| feral_animal | 2.8% |
| detective | 3.1% |
| dream_logic | 3.6% |

Most are a stray phrase; 4 rows are more than 30% CJK, the worst at 80%.

Why it matters: the language direction in activation space dwarfs the emotion
directions, so these are strong outliers — and because the rate rises with
distance from the assistant, it is correlated with the x-axis of the headline
figure. "Probe accuracy falls with persona distance" and "the text stops being
English with persona distance" would be hard to tell apart.

`01_data.ipynb` has a **hard gate that raises on this corpus**, by design.
Extraction must not run until this is resolved.

Options, none of them taken yet — this is a methodological call:

1. **Rejection sampling at generation.** Detect CJK in the response inside
   `call_one()` and redraw, capped at N attempts, as a uniform rule applied to
   every cell. The final corpus is then drawn from P(text | English), identically
   for every persona. Cheapest by far: the request body does not change, so cache
   keys do not change — delete the 54 offending cache entries and rerun, ~54
   calls. Note it *is* conditioning on the output; the defence is that the rule
   is uniform across cells rather than applied post hoc to a finished corpus.
   Any rows that never come back clean are a residue needing their own decision.
2. **Lower `GEN_TEMPERATURE`** from 1.0. Code-switching is a sampling artifact.
   Changes corpus diversity everywhere and invalidates every cache key: full
   refetch, ~1 h, ~$0.25.
3. **Accept and document.** Count it as a known confound, like leakage, and
   override the gate deliberately. Costs nothing, leaves a confound correlated
   with the x-axis in the headline result.

Option 1 is the cheapest and the most defensible of the three, but it has not
been chosen.

## Picking it back up

**No GPU is needed for any of this except stage 5.** Stage 4 is pure API:
`gen_data.py` imports only `config`, `personas`, `scenarios` at module level,
with `httpx` and `tqdm` loaded lazily inside `main()`, and `topics.json` is
committed so the `huggingface_hub`/`pandas` download path is never touched.

| stage | needs | where |
| --- | --- | --- |
| 4 `gen_data.py` | Python 3.10+, httpx, tqdm, `OPENROUTER_API_KEY` | any laptop |
| 5 `extract.py` | torch, transformers, ~65 GB weights, 80 GB VRAM | GPU box |
| 6–8 axis / probes / plot | numpy, pandas, sklearn, matplotlib | any laptop |

So: resolve the CJK decision and iterate stage 4 locally, then rent a GPU **once**
for a single extraction against a corpus you are happy with.

```bash
export OPENROUTER_API_KEY=sk-or-...
python gen_data.py --dry-run      # confirms what a change actually invalidates
python gen_data.py                # refetches only that
jupyter lab notebooks/01_data.ipynb   # the CJK gate must pass before extraction
```

Stage 5 needs `HF_HOME` pointed somewhere with ~65 GB free — on the original pod
`/workspace` was a 20 GB volume, so this meant either resizing it or using the
117 GB `/dev/shm` tmpfs and re-downloading after any restart. `preflight.py`
checks whichever path `HF_HOME` actually points at.

```bash
python preflight.py
PP_OUT=outputs/smoke PP_PRESET=0.5b python extract.py --limit 8   # plumbing check
python extract.py && python assistant_axis.py && python probes.py && python plot.py
```
