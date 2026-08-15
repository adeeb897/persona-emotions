# CLAUDE.md

Hackathon experiment: do emotion probes trained on a model's default assistant
persona still work as the model roleplays personas further from that default?
See `README.md` for the design and the run order.

## Fixed design decisions

These are settled. They are not defaults waiting to be improved, and several of
them look like arbitrary choices until you know what they are protecting against
— which is why they are written down here. Do not change one to make a number
look better, to simplify code, or because a stage would be tidier without it. If
a change genuinely requires revisiting one, stop and ask first.

1. **The assistant axis is anchored on `default_assistant`.** Not on
   `bare_template`, not on a pooled "all personas" mean. The anchor is one
   explicit, written, length-matched prompt.

2. **`bare_template` is a control, never used to define the axis.** It sends no
   system message, so the chat template supplies its own short vendor string. It
   is generated, extracted and projected like everything else, and marked
   distinctly in every figure — but it never enters the difference-of-means that
   defines the axis. Near the anchor → prompt length is not driving the axis. Far
   → there is a real confound in the headline figure. Either way it is a result.

3. **Condition (b) always scores on the same held-out topics as (a).** Transfer
   is evaluated fold by fold on exactly the topics the fold held out, so (a) and
   (b) differ only in persona. The all-topics variant stays as a stored secondary
   metric; it is systematically more optimistic and is never the headline.

4. **The best layer is selected on (a) only** — default-persona accuracy. Never
   on transfer accuracy, which would select for the result being measured.

5. **Splits are over topics, cross-validated, 5 folds** (`config.N_FOLDS`). Six
   topics per fold, each topic tested exactly once, never train and test on the
   same topic. Never split by row: rows within a topic are not independent.
   Reported accuracies are means over folds; per-fold values, SD and SEM are all
   kept in `probe_results.json`, and figures show ±1 SD across folds.

6. **Topic sample seed is 0** (`config.TOPIC_SAMPLE_SEED`), and the same 30
   topics appear in every (persona, emotion) cell — the design is fully crossed.
   If topics varied by cell, a drop in transfer accuracy could be topic shift
   rather than persona shift and the two could not be separated. The 30 are a
   seeded random sample, not curated; awkward pairings are the point.

7. **Emotion leakage is counted, never filtered.** Reported overall and broken
   down by persona, because leakage that varies across personas varies along the
   x-axis of the headline figure. Whether a leaked row is still a valid sample is
   the researcher's call, not the script's.

8. **Dispatch order is shuffled (`config.DISPATCH_SEED` = 0), and the serving
   provider is pinned (`config.GENERATOR_PROVIDER`) and logged per row.** Same
   class of confound as per-persona leakage, arrived at the hard way. The grid is
   built persona-major, so a contiguous block of 360 calls *is* one persona: any
   time-varying API condition — a provider entering or leaving the routing pool,
   rate limiting, a model version flip — lands on one persona and masquerades as
   a persona effect. The first run lost 192 rows to a provider that rejects
   chat-completions requests, and they fell in 5 of the 9 personas, 74 in
   `close_friend` and 0 in the first three. Shuffling spreads such windows across
   personas as noise; pinning removes the routing variance; logging the provider
   makes what remains auditable instead of assumed. `gen_data.py` prints a
   provider-by-persona table, which should be one column and flat — if it isn't,
   the pin did not hold and the corpus is heterogeneous in quantization.
   Failure modes (`http_4xx` vs `api_error_200` vs `transient`) are counted
   separately: the first two look identical in a progress bar and were what
   identified the cause.

9. **Persona prompts are length-matched (44–50 words), including the anchor.**
   `default_assistant` must not be the shortest prompt in the set — otherwise
   "distance from the anchor" is partly "longer prompt", which is the confound
   `bare_template` exists to detect, applied to the anchor itself. Same skeleton
   and same closing line for all of them. Do not lengthen or restructure one for
   readability. `python personas.py` prints the spread.

## Working agreements

- **Methodological calls belong to the researcher.** Implement the architecture;
  if a scientific question comes up mid-task — what to measure, how to score it,
  what counts as a valid sample — stop and ask rather than deciding.
- Every stage runs standalone from the CLI, prints what it is doing, and takes
  `--limit N` for a smoke test. Keep it that way.
- Notebooks are read-only views over what the scripts already wrote. Nothing is
  computed in a notebook that a script does not also compute, so results never
  depend on cell execution order.
- `PP_OUT` redirects stage 5–8 outputs; `PP_PRESET` swaps the probed model.
  Smoke runs use both so they cannot clobber a real extraction.
- Layer indices and hidden dims are read from the loaded model config, never
  hardcoded, so a preset swap cannot silently reshape the data.
