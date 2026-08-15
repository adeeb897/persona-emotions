# STATUS

As of 2026-08-15. Read this first if you are picking the project up cold; it is
the state of play, not the design. `README.md` is the design, `CLAUDE.md` is the
list of decisions that must not drift.

## Where things stand

Stages 0–4 are **done**. The CJK decision that was blocking extraction is
**resolved** — the researcher chose rejection sampling, it is implemented, and
the corpus (run 4) is 100% English with the crossed design intact. Stages 5–8
run end to end on the **real 3,240-row corpus** at the 0.5b preset; the only
thing left is one 32B extraction on a GPU.

All of it is committed on the branch `run4-cjk-rejection-sampling` and **not
pushed**. See the bottom of this file.

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
| 3 | run 2 + `Write in English only.` in the prompt | **superseded by run 4.** 3,240/3,240; 17 failures, all HTTP 429, refetched. Leakage 8 rows (0.25%), 0.0–1.1% spread. **CJK 54 rows (1.67%)**, 0.3–3.6% across personas — the instruction alone did not work. |
| 4 | run 3 + CJK rejection sampling (`config.MAX_CJK_REDRAWS`) | **current.** Only the 54 CJK cache entries were deleted and refetched; the other 3,186 rows are byte-identical to run 3. 54/54 fetched, 0 failures, 45 s. **CJK 0**, 0 residue, 3 redraws total (resentful_ai 1, dream_logic 2). Leakage unchanged at 8 rows (0.25%). |

Verified on run 4, by diff against the run-3 corpus: exactly 54 rows changed,
every one of them a row that had CJK, none of the replacements has CJK, and no
unchanged row had CJK. 108 cells at exactly 30 rows, one identical topic set
across every cell (crossed design intact), DeepInfra 3,240/3,240 (the pin held),
one returned model string throughout. `01_data.ipynb` executed headless and its
hard gate **passes**: *"0 of 3240 generations contain CJK characters — corpus is
100% English; safe to extract."*

## The CJK decision, as resolved

The researcher chose **rejection sampling** (option 1 of the three that were on
the table; the others were lowering `GEN_TEMPERATURE`, and accepting the
confound and documenting it).

What it does: a draw containing CJK is rejected and redrawn, up to
`config.MAX_CJK_REDRAWS` = 8, as a rule applied **uniformly in every cell**. The
corpus is then a sample from P(text | English) identically for every persona —
including the 3,186 rows that were already clean, since "keep if clean, redraw if
not" is what rejection sampling *is*. It does condition on the output; the
defence is that the rule is uniform across cells rather than applied post hoc to
a finished corpus.

Three properties worth keeping in mind before touching `gen_data.py`:

- **The redraw reuses the same request body**, so cache keys do not move and an
  unchanged rerun still costs zero. Never add a seed or nonce to force
  variation — that changes every hash and invalidates the whole cache.
- **The redraw loop is outside the transport-retry loop.** A 429 retry is not a
  redraw and a redraw does not burn the 429 backoff budget; the two failures are
  different animals and are counted separately.
- **A row that exhausts the cap is cached anyway**, flagged `cjk_residue: true`,
  and reported. Leaving it uncached would drop it from `generations.jsonl` and
  break the 108-cells-of-30 design — the thing decision 9's "counted, never
  dropped" exists to prevent. At the worst persona's 3.6% rate this is a safety
  net, not an expected path (P ≈ 3e-12), and run 4 hit it zero times.

Every accepted row carries `cjk_redraws` (int) and `cjk_residue` (bool);
`gen_data.py` prints a cumulative per-persona redraw table read back off the
cache, so the audit trail survives into later reruns. Rows generated before this
existed carry no field and count as 0 — which is why only 54 rows have it.

The redraw rate still carries the persona gradient the CJK rate did (2 of 3
redraws on `dream_logic`). That is expected and is *not* a confound in the
activations — every accepted row is English either way — but it is on the record
rather than assumed away.

## The full-corpus 0.5b rehearsal

Stages 5–8 have now been run **on the real 3,240-row corpus**, on a local RTX
5070, at `PP_PRESET=0.5b` into `PP_OUT=outputs/smoke`. This is a plumbing result
and nothing else — *never report a number from the 0.5b preset* — but it means
the only thing the 32B run has left to prove is the 32B.

What it demonstrated, on real text rather than synthetic activations:

- Extraction: 3,240 rows, **0 skipped** — no record exceeds `MAX_SEQ_LEN`, and
  the response slice is non-empty for every one. 28.6 s of forward passes at
  113 rows/s, peak 1.9 GB.
- The measured suffix logic works on the real template: `'<|im_end|>\n'`, 2
  tokens.
- Stages 6, 7, 8 all complete, and `02_results.ipynb` executes headless against
  their outputs. Three figures and their `.csv` twins are written.
- All three notebooks execute headless, `00_setup` and `01_data` included.

Two things that showed up in the rehearsal and are worth watching on the 32B —
both are *the experiment working*, not bugs:

- **`bare_template` lands at distance 0.479**, mid-pack, nearest `detective` /
  `peasant`. Per README that is the "far" branch: the axis may be picking up
  prompt length or template effects, not persona alone. Whether it replicates at
  32B is one of the results.
- **`feral_animal` projects further than `dream_logic`** (1.30 vs the anchor's
  1.00), so the intuited "most distant" persona is not the measured one, and 7
  of 9 personas moved from their intuited rank. Also a result, and also to be
  re-read at 32B.

## Picking it back up

**No GPU is needed for any of this except stage 5.** Stage 4 is pure API:
`gen_data.py` imports only `config`, `personas`, `scenarios` at module level,
with `httpx` and `tqdm` loaded lazily inside `main()`, and `topics.json` is
committed so the `huggingface_hub`/`pandas` download path is never touched.

| stage | needs | where | status |
| --- | --- | --- | --- |
| 4 `gen_data.py` | Python 3.10+, httpx, tqdm, `OPENROUTER_API_KEY` | any laptop | **done** (run 4) |
| 5 `extract.py` | torch, transformers, ~65 GB weights, 80 GB VRAM | GPU box | **the only thing left** |
| 6–8 axis / probes / plot | numpy, pandas, sklearn, matplotlib | any laptop | rehearsed on the real corpus at 0.5b |

So: rent a GPU **once**, for a single 32B extraction against a corpus that is
already final.

```bash
python preflight.py
PP_OUT=outputs/smoke PP_PRESET=0.5b python extract.py --limit 8   # plumbing check
python extract.py && python assistant_axis.py && python probes.py && python plot.py
```

Stage 5 needs `HF_HOME` pointed somewhere with ~65 GB free — on the original pod
`/workspace` was a 20 GB volume, so this meant either resizing it or using the
117 GB `/dev/shm` tmpfs and re-downloading after any restart. `preflight.py`
checks whichever path `HF_HOME` actually points at.

`pip install hf_xet` before the download: without it `huggingface_hub` prints
*"Xet Storage is enabled for this repo, but the 'hf_xet' package is not
installed. Falling back to regular HTTP download"* and takes the slow path for
all 65 GB.

## Running off the pod

The repo now runs on a Windows laptop as well as on Linux. Three things made
that true, and all three matter on the pod too:

- **`config.WORKSPACE` falls back to the repo** when `/workspace` does not
  exist. It used to be an unconditional `/workspace`, so merely importing
  `config` — which `python personas.py` does — created `C:\workspace\hf_home`.
  On the pod `/workspace` exists, so behaviour there is unchanged, and
  `PP_WORKSPACE` still overrides.
- **`config` reads a gitignored `.env`** for `OPENROUTER_API_KEY`, without a
  dependency. A real environment variable always wins, so `export` on the pod
  behaves exactly as before; this exists because Windows `setx` does not affect
  the current process and an exported key is easy to lose.
- **The cache write in `gen_data.py` now passes `encoding="utf-8"`.** It is
  written with `ensure_ascii=False`, so it can contain non-ASCII, while every
  read already specified utf-8. On Windows the bare `write_text` used cp1252 —
  a CJK residue row would have raised `UnicodeEncodeError`, and anything else
  non-ASCII would have been written as mojibake and read back mangled. The Linux
  pod hid this by having a UTF-8 locale. **This bug was latent in the run-3
  corpus too**; it just never fired, because that corpus was written on Linux.

Local environment, for reference — this is *not* what the results come from:

| | |
| --- | --- |
| Python | 3.13.5 (Anaconda) |
| torch | 2.8.0+cu129, RTX 5070 12 GB (sm_120) |
| transformers | **4.56.2**, not the pinned 4.57.6 — stage 5 ran fine on it, including the `dtype=` kwarg. The pin exists for the pod's torch 2.4.1 and should stay. |
| nbconvert | 7.16.6 — all three notebooks execute headless |

`preflight.py` runs here too. It correctly reports one blocker (10.8 GB free
VRAM against the ~72 GB the 32B needs) and skips the `/proc/meminfo` RAM check
as Linux-only rather than warning about a missing file.

## What was committed

On branch **`run4-cjk-rejection-sampling`**, not on `main`, and **not pushed**.
Fast-forward `main` onto it when you are happy with it.

| commit | what |
| --- | --- |
| `38208d9` | `gen_data: reject and redraw code-switched generations` — the redraw loop, `config.MAX_CJK_REDRAWS`, decision 9 in `CLAUDE.md`, the `README.md` method note, the utf-8 cache write, and the off-pod fixes in `config.py` / `preflight.py` |
| `5b7449c` | `data: commit run-4 corpus (CJK 0, 54 rows refetched)` — `generations.jsonl` and the 54 changed `gen_cache/` entries |
| `f7c44e1` | `plot: keep labels off other personas' markers` |

The intended split had the off-pod fixes as their own commit, but `config.py`
carries changes belonging to both it and the rejection-sampling commit, and
separating them would have meant `git add -p` through reformatted files. They
are squashed into the first commit instead.

The corpus commit is the one that matters — at temperature 1.0 a rerun produces
a different corpus, not the same one, and the 54 refetched rows are as
unreproducible as the original 3,186.

**The Python diffs are much larger than the edits.** A `PostToolUse` hook runs a
formatter on every file that is written, so `config.py`, `gen_data.py`,
`plot.py` and `preflight.py` came back reformatted whole — mostly multi-line
`print(f"...")` calls being re-wrapped. It is non-semantic, and all four were
run after formatting. Separating the reformat into its own commit would mean
reconstructing a reformat-only state on top of HEAD, which is more work than the
tidier history is worth; `git diff -w` is the cheaper way to read these.

Everything else in the repo is unchanged: `personas.py`, `scenarios.py`,
`extract.py`, `assistant_axis.py`, `probes.py`, the notebooks, and
`data/topics.json` were not touched.

## What is left

1. Rent the GPU. `preflight.py`, then the 0.5b `--limit 8` plumbing check into
   `PP_OUT=outputs/smoke`, then the real run.
2. `python extract.py && python assistant_axis.py && python probes.py &&
   python plot.py`. Extraction is the only slow part; at 0.5b the 3,240 forward
   passes took 29 s, so budget on the order of tens of minutes at 32B, not
   hours.
3. Read `02_results.ipynb` against the 32B outputs, and specifically re-check
   the two things the rehearsal flagged: where `bare_template` lands, and
   whether the measured persona order matches the intuited one.
