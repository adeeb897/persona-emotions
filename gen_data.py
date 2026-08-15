"""Stage 4: generate the corpus via OpenRouter.

9 personas x 12 emotions x 30 topics = 3,240 short first-person messages in
which the persona is experiencing the emotion without naming it.

No GPU involved -- this stage only talks to the API.  It is resumable: every
completed call is written to its own file in data/gen_cache/ keyed by a hash of
the exact request content, and a rerun skips anything already cached.  A run
that dies at 90% costs you the last 10%, not the whole thing.

Draws that code-switch into Chinese are rejected and redrawn, as a uniform rule
in every cell, so the corpus is a sample from P(text | English) identically for
every persona (config.MAX_CJK_REDRAWS).  The redraw reuses the same request
body, so cache keys do not move.

Two things here exist to keep API conditions from correlating with persona,
because the grid is built persona-major and a 360-row block IS one persona:
the serving provider is PINNED (config.GENERATOR_PROVIDER) and recorded per
row, and dispatch order is SHUFFLED (config.DISPATCH_SEED) so an outage window
spreads across personas as noise.  The first run had neither, and lost 192 rows
concentrated in 5 of the 9 personas.

    python gen_data.py --limit 8      # smoke test, ~8 calls
    python gen_data.py                # the real run
    python gen_data.py --dry-run      # print the grid and cost estimate, call nothing
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time

import config
import personas
import scenarios

USER_PROMPT_TEMPLATE = (
    "Situation: {topic}\n\n"
    "Write about this situation in the first person, in about {lo}-{hi} words. "
    "As you write, you are feeling {emotion}. Do not name that feeling, and do "
    "not use any obvious synonym for it -- let it show only through what you "
    "say and how you say it. Write in English only. Reply with the message only."
)

# Qwen code-switches into Chinese at temperature 1.0.  Run 2 (no English
# instruction) produced 50 such rows, 1.54%, varying 0.0-3.1% across personas.
# That matters more than the rate suggests: the language direction in activation
# space dwarfs the emotion directions, so a handful of code-switched rows are
# strong outliers, and a rate that varies by persona contaminates both the axis
# and transfer accuracy.  Hence the instruction above, and the checks below.
#
# The instruction alone did not work -- run 3 carried it and still produced 54
# CJK rows (1.67%), 0.3-3.6% across personas, if anything a sharper gradient.
# So call_one() also REJECTS a draw containing CJK and redraws, uniformly in
# every cell (config.MAX_CJK_REDRAWS).  These predicates are what it rejects on.
#
# Unicode ranges, not str.split() artifacts: CJK text has no spaces, so
# `len(text.split())` reports a 200-character Chinese message as "3 words" and
# the length checks are blinded to it.  That is how run 2 hid this.
CJK_RE = re.compile(r"[一-鿿぀-ヿ㐀-䶿豈-﫿]")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def cjk_fraction(text: str) -> float:
    return len(CJK_RE.findall(text)) / max(len(text), 1)


def count_words(text: str) -> int:
    """Word count that does not go blind on CJK.

    Whitespace tokens that contain no CJK, plus one per CJK character (the
    conventional approximation -- a CJK character is roughly a word).  Using
    str.split() alone reports a fully-Chinese generation as 3 words, which reads
    as "suspiciously short" instead of "not English".
    """
    cjk = len(CJK_RE.findall(text))
    latin = sum(1 for tok in text.split() if not CJK_RE.search(tok))
    return latin + cjk


def build_grid(limit: int | None = None) -> list[dict]:
    """Every (persona, emotion, scenario) cell.  The same 30 topics in each."""
    chosen = scenarios.sample_scenarios()
    grid = [
        {
            "persona_id": p["id"],
            "emotion": emotion,
            "scenario_id": sid,
            "topic": topic,
        }
        for p in personas.PERSONAS
        for emotion in config.EMOTIONS
        for sid, topic in chosen
    ]
    if limit is not None:
        # Deterministic shuffle before truncating, so a smoke test touches
        # several personas and emotions rather than 8 rows of one cell.
        random.Random(config.RANDOM_SEED).shuffle(grid)
        grid = grid[:limit]
    return grid


def user_prompt_for(item: dict) -> str:
    """The user turn for a grid item.

    extract.py imports this so the prompt a message was generated under is
    byte-identical to the prompt it is replayed under.  Change it here and both
    stages move together (and every content hash correctly invalidates).
    """
    return USER_PROMPT_TEMPLATE.format(
        topic=item["topic"],
        emotion=item["emotion"],
        lo=config.TARGET_WORDS[0],
        hi=config.TARGET_WORDS[1],
    )


def request_for(item: dict) -> dict:
    """The exact API request body for a grid item.

    Includes the provider pin when one is configured.  That makes the pin part
    of the content hash, which is correct: a corpus generated under free routing
    and one generated on a pinned provider are not interchangeable, and the
    cache must not blend them.
    """
    body = {
        "model": config.GENERATOR_MODEL,
        "messages": personas.build_messages(item["persona_id"], user_prompt_for(item)),
        "temperature": config.GEN_TEMPERATURE,
        "max_tokens": config.GEN_MAX_TOKENS,
    }
    if config.GENERATOR_PROVIDER:
        body["provider"] = {
            "order": [config.GENERATOR_PROVIDER],
            "allow_fallbacks": config.GENERATOR_ALLOW_FALLBACKS,
        }
    return body


def content_hash(body: dict) -> str:
    """Hash the request that will actually be sent.

    Anything that changes the output -- model, system prompt, user prompt,
    temperature, max_tokens -- changes the hash, so editing a persona prompt
    correctly invalidates only that persona's cache.  Nothing else does.
    """
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def cache_path(h: str):
    return config.GEN_CACHE_DIR / f"{h}.json"


def note_failure(stats: dict, mode: str, item: dict, detail: str) -> None:
    """Record how a call failed, not just that it did.

    The first run's failures arrived in two shapes that look identical in a
    progress bar and are diagnosed completely differently:

      http_4xx        HTTP 400 from OpenRouter itself -- permanent, not retried
      api_error_200   HTTP 200 whose BODY carries an error -- retried to
                      exhaustion before being given up on, so it burns the full
                      backoff budget while being just as permanent
      transient       anything else (timeout, 5xx, 429) that survived retries

    Both of the first two were the same underlying cause (a provider rejecting
    the endpoint), and telling them apart is what identified it.
    """
    stats["fail_modes"][mode] = stats["fail_modes"].get(mode, 0) + 1
    stats["fail_by_persona"][item["persona_id"]] = (
        stats["fail_by_persona"].get(item["persona_id"], 0) + 1
    )
    stats["failed"] += 1
    if mode not in stats["fail_examples"]:
        stats["fail_examples"][mode] = detail[:200]


async def call_one(client, item: dict, body: dict, h: str, stats: dict) -> dict | None:
    """One accepted draw for a grid item, redrawing on code-switching.

    Wraps `draw_once` -- which owns transport retry/backoff -- in a rejection
    loop, so a CJK redraw never burns the 429 budget and a 429 retry is never
    counted as a redraw.  The two failure kinds are genuinely different: one is
    the API misbehaving, the other is the model sampling Chinese.

    The request body is identical on every redraw, so all draws for an item
    share one cache key and `--dry-run` still reports the true amount of work.
    Returns the accepted record (with `cjk_redraws` recorded), the last draw
    flagged `cjk_residue` if the cap is exhausted, or None if the call failed
    permanently -- in which case the item is left uncached for the next rerun.
    """
    for n_redraws in range(config.MAX_CJK_REDRAWS + 1):
        record = await draw_once(client, item, body, h, stats)
        if record is None:
            return None  # transport failure: report it, do not redraw
        if not has_cjk(record["text"]):
            record["cjk_redraws"] = n_redraws
            record["cjk_residue"] = False
            if n_redraws:
                stats["redraws"][item["persona_id"]] = (
                    stats["redraws"].get(item["persona_id"], 0) + n_redraws
                )
            return record
        stats["cjk_draws"] += 1

    # Cap exhausted.  Cache the last draw anyway: leaving it uncached would drop
    # the row from generations.jsonl and break the 108-cells-of-30 design, which
    # is worse than one flagged row the notebook gate will raise on.
    record["cjk_redraws"] = config.MAX_CJK_REDRAWS
    record["cjk_residue"] = True
    stats["redraws"][item["persona_id"]] = (
        stats["redraws"].get(item["persona_id"], 0) + config.MAX_CJK_REDRAWS
    )
    stats["cjk_residue"] += 1
    print(
        f"\n[gen] CJK residue: {item['persona_id']}/{item['emotion']}/"
        f"{item['scenario_id']} still contains CJK after "
        f"{config.MAX_CJK_REDRAWS} redraws -- cached and flagged, not dropped"
    )
    return record


async def draw_once(client, item: dict, body: dict, h: str, stats: dict) -> dict | None:
    """One API call with retry/backoff.  Returns the record, or None if it
    failed permanently (the item is simply left uncached for the next rerun)."""
    url = f"{config.OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "X-Title": "persona-probes",
    }

    for attempt in range(config.MAX_RETRIES):
        try:
            resp = await client.post(
                url, headers=headers, json=body, timeout=config.REQUEST_TIMEOUT_SECONDS
            )
            if resp.status_code == 200:
                payload = resp.json()
                if "error" in payload:
                    raise RuntimeError(f"api error: {payload['error']}")
                text = payload["choices"][0]["message"]["content"].strip()
                if not text:
                    raise RuntimeError("empty completion")

                usage = payload.get("usage") or {}
                stats["in_tok"] += usage.get("prompt_tokens", 0)
                stats["out_tok"] += usage.get("completion_tokens", 0)

                # Which provider actually served this row.  OpenRouter reports
                # it per response; without persisting it, provider
                # heterogeneity across the corpus is unauditable after the
                # fact -- which is exactly what went wrong on the first run.
                served_by = payload.get("provider")
                stats["providers"][served_by] = stats["providers"].get(served_by, 0) + 1

                return {
                    **item,
                    "text": text,
                    "generator_model": config.GENERATOR_MODEL,
                    "generator_provider": served_by,
                    "generator_provider_pinned": config.GENERATOR_PROVIDER,
                    "returned_model": payload.get("model"),
                    "content_hash": h,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }

            # 429 and 5xx are worth retrying; a 4xx is our fault and will not
            # fix itself.
            if resp.status_code != 429 and resp.status_code < 500:
                print(
                    f"\n[gen] permanent {resp.status_code} for "
                    f"{item['persona_id']}/{item['emotion']}/{item['scenario_id']}: "
                    f"{resp.text[:200]}"
                )
                note_failure(
                    stats, "http_4xx", item, f"HTTP {resp.status_code}: {resp.text}"
                )
                return None
            raise RuntimeError(f"http {resp.status_code}")

        except Exception as exc:  # noqa: BLE001 -- retry anything transient
            if attempt == config.MAX_RETRIES - 1:
                print(
                    f"\n[gen] gave up on {item['persona_id']}/{item['emotion']}/"
                    f"{item['scenario_id']} after {config.MAX_RETRIES}: {exc}"
                )
                # An error carried inside a 200 body is a different animal from
                # a timeout, even though both land here after retries.
                mode = (
                    "api_error_200" if str(exc).startswith("api error") else "transient"
                )
                note_failure(stats, mode, item, str(exc))
                return None
            await asyncio.sleep(config.BACKOFF_BASE_SECONDS * (2**attempt))
    return None


def estimated_cost(stats: dict) -> float:
    return (
        stats["in_tok"] / 1e6 * config.GENERATOR_PRICE_IN_PER_MTOK
        + stats["out_tok"] / 1e6 * config.GENERATOR_PRICE_OUT_PER_MTOK
    )


async def run(todo: list[tuple[dict, dict, str]], stats: dict) -> None:
    import httpx
    from tqdm import tqdm

    sem = asyncio.Semaphore(config.CONCURRENCY)
    bar = tqdm(total=len(todo), desc="generating", unit="call")
    t0 = time.time()

    async with httpx.AsyncClient() as client:

        async def worker(item, body, h):
            async with sem:
                record = await call_one(client, item, body, h, stats)
                if record is not None:
                    # Write per-item so a crash cannot lose completed work.
                    #
                    # encoding is explicit because ensure_ascii=False means this
                    # file can contain non-ASCII, and write_text() otherwise uses
                    # the platform default -- cp1252 on Windows, which either
                    # raises on a CJK residue row or writes mojibake that the
                    # utf-8 readers below silently mangle.  The Linux pod hid
                    # this by having a UTF-8 locale.
                    cache_path(h).write_text(
                        json.dumps(record, ensure_ascii=False), encoding="utf-8"
                    )
                    stats["done"] += 1
                bar.update(1)
                if stats["done"] and stats["done"] % 50 == 0:
                    rate = stats["done"] / max(time.time() - t0, 1e-6)
                    bar.set_postfix_str(
                        f"${estimated_cost(stats):.2f} est, {rate:.1f}/s, "
                        f"{stats['failed']} failed"
                    )

        await asyncio.gather(*(worker(i, b, h) for i, b, h in todo))
    bar.close()


def write_jsonl(grid: list[dict]) -> int:
    """Rebuild generations.jsonl from the cache, in grid order.

    Always regenerated from cache rather than appended to, so the file is
    consistent even after a partial run.
    """
    n = 0
    with open(config.GENERATIONS_JSONL, "w", encoding="utf-8") as fh:
        for item in grid:
            h = content_hash(request_for(item))
            p = cache_path(h)
            if p.exists():
                fh.write(p.read_text(encoding="utf-8").rstrip() + "\n")
                n += 1
    return n


def report_redraws(grid: list[dict]) -> None:
    """Per-persona rejection-sampling history, read back off the cache.

    The in-run table only covers calls this invocation made; this one survives
    into every later rerun, so the corpus carries its own audit trail.  A
    persona gradient here is expected and is not a confound -- the accepted rows
    are English either way -- but "how much resampling did each persona need"
    is exactly the kind of thing that has to be on the record rather than
    reconstructed later from memory.  Rows generated before rejection sampling
    existed carry no field and count as 0.
    """
    redraws, residue, seen = {}, {}, {}
    for item in grid:
        p = cache_path(content_hash(request_for(item)))
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        pid = rec["persona_id"]
        seen[pid] = seen.get(pid, 0) + 1
        redraws[pid] = redraws.get(pid, 0) + int(rec.get("cjk_redraws", 0) or 0)
        if rec.get("cjk_residue"):
            residue[pid] = residue.get(pid, 0) + 1

    if not sum(redraws.values()) and not residue:
        return

    print(
        f"\n[gen] CJK rejection sampling, cumulative over the cache "
        f"(cap {config.MAX_CJK_REDRAWS} per item):"
    )
    print(f"    {'rank':<5} {'persona':<18} {'redraws':>8} {'residue':>8}")
    for pp in personas.PERSONAS:
        pid = pp["id"]
        if not seen.get(pid):
            continue
        tag = "  <- CONTROL" if pp["is_control"] else ""
        print(
            f"    {pp['distance_rank']:<5} {pid:<18} {redraws.get(pid, 0):>8} "
            f"{residue.get(pid, 0):>8}{tag}"
        )
    print(
        f"    total: {sum(redraws.values())} redraws, {sum(residue.values())} residue"
    )


def report_cjk(grid: list[dict]) -> int:
    """Count code-switched generations per persona.  Returns the total.

    Unlike leakage, this one is expected to be ZERO: draws containing CJK are
    rejected and redrawn at generation time (config.MAX_CJK_REDRAWS), uniformly
    in every cell, so a non-zero count here means items exhausted the cap. Those
    are still counted rather than filtered -- dropping rows would break the fully
    crossed design (108 cells x exactly 30) and would do it unevenly across
    personas, reintroducing the imbalance the provider pin removed.
    """
    counts, totals = {}, {}
    worst = []
    for item in grid:
        p = cache_path(content_hash(request_for(item)))
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        pid = rec["persona_id"]
        totals[pid] = totals.get(pid, 0) + 1
        if has_cjk(rec["text"]):
            counts[pid] = counts.get(pid, 0) + 1
            worst.append((cjk_fraction(rec["text"]), rec))

    total = sum(counts.values())
    n = sum(totals.values())
    if not total:
        print(
            f"\n[gen] CJK check: clean -- 0 of {n} generations contain CJK characters."
        )
        return 0

    print(
        f"\n[gen] WARNING: {total} of {n} generation(s) contain CJK characters "
        f"({total / n:.2%}) despite the English-only instruction:"
    )
    print(f"    {'rank':<5} {'persona':<18} {'cjk':>5} {'of':>6} {'rate':>7}")
    for pp in personas.PERSONAS:
        pid = pp["id"]
        if not totals.get(pid):
            continue
        c = counts.get(pid, 0)
        tag = "  <- CONTROL" if pp["is_control"] else ""
        print(
            f"    {pp['distance_rank']:<5} {pid:<18} {c:>5} {totals[pid]:>6} "
            f"{c / totals[pid]:>6.1%}{tag}"
        )
    worst.sort(key=lambda t: -t[0])
    f, rec = worst[0]
    print(
        f"    worst: {rec['persona_id']}/{rec['emotion']}/{rec['scenario_id']} "
        f"at {f:.0%} CJK"
    )
    print(
        "    NOT dropped -- filtering would break the crossed design unevenly "
        "across personas. This needs a decision, not a filter."
    )
    return total


def report_leakage(grid: list[dict]) -> None:
    """Flag rows where the target emotion appears verbatim.

    Deliberately reports rather than filters -- whether a leaked row is still
    a valid sample is a call for you, not for this script.

    Broken down BY PERSONA as well as by emotion, and printed in axis order.
    An overall rate is not enough: if leakage varies systematically across
    personas it varies along the x-axis of the headline figure, and "probe
    accuracy falls with distance" becomes indistinguishable from "the emotion
    word stops appearing in the text with distance".  The spread is the number
    to look at, not the mean.
    """
    leaked, total = [], {}
    for item in grid:
        p = cache_path(content_hash(request_for(item)))
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        total[rec["persona_id"]] = total.get(rec["persona_id"], 0) + 1
        if re.search(rf"\b{re.escape(rec['emotion'])}\b", rec["text"], re.IGNORECASE):
            leaked.append(rec)

    n_cached = sum(total.values())
    if not leaked:
        print(
            f"[gen] no generation names its target emotion verbatim (0 of {n_cached})."
        )
        return

    by_persona, by_emotion = {}, {}
    for rec in leaked:
        by_persona[rec["persona_id"]] = by_persona.get(rec["persona_id"], 0) + 1
        by_emotion[rec["emotion"]] = by_emotion.get(rec["emotion"], 0) + 1

    print(
        f"\n[gen] WARNING: {len(leaked)} of {n_cached} generation(s) name the "
        f"target emotion verbatim ({len(leaked) / n_cached:.2%}, NOT dropped "
        f"-- your call):"
    )

    print(f"\n    by persona (axis order; a trend here is a confound, not a nuisance)")
    print(f"    {'rank':<5} {'persona':<18} {'leaked':>7} {'of':>6} {'rate':>7}")
    rates = []
    for p in personas.PERSONAS:
        pid = p["id"]
        if not total.get(pid):
            continue
        n, n_tot = by_persona.get(pid, 0), total[pid]
        rate = n / n_tot
        rates.append(rate)
        tag = "  <- CONTROL" if p["is_control"] else ""
        print(
            f"    {p['distance_rank']:<5} {pid:<18} {n:>7} {n_tot:>6} {rate:>6.1%}{tag}"
        )
    if rates:
        print(
            f"    spread across personas: {min(rates):.1%} - {max(rates):.1%} "
            f"({(max(rates) - min(rates)) * 100:.1f}pp)"
        )

    print(f"\n    by emotion")
    for emotion, n in sorted(by_emotion.items(), key=lambda kv: -kv[1]):
        print(f"    {n:4d}  {emotion}")
    print("\n    (synonyms are not detected -- this is a floor, not a ceiling)")


def report_providers(grid: list[dict]) -> None:
    """Which provider served each persona's rows.

    Reported for the same reason leakage is: OpenRouter's providers differ in
    quantization and serving stack, so if provider assignment varies across
    personas it varies along the x-axis of the headline figure.  Pinning is
    supposed to make this table trivial -- one column, flat.  If it is not
    trivial, the pin did not hold and the corpus is heterogeneous.
    """
    counts, missing = {}, 0
    for item in grid:
        p = cache_path(content_hash(request_for(item)))
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        prov = rec.get("generator_provider")
        if prov is None:
            missing += 1
        counts.setdefault(rec["persona_id"], {})
        counts[rec["persona_id"]][prov] = counts[rec["persona_id"]].get(prov, 0) + 1

    if not counts:
        return
    provs = sorted(
        {p for row in counts.values() for p in row}, key=lambda p: (p is None, str(p))
    )

    print(
        f"\n[gen] serving provider by persona "
        f"(pinned: {config.GENERATOR_PROVIDER or 'NONE -- free routing'})"
    )
    header = f"    {'persona':<18}" + "".join(f"{str(p):>14}" for p in provs)
    print(header)
    for pp in personas.PERSONAS:
        row = counts.get(pp["id"])
        if not row:
            continue
        print(f"    {pp['id']:<18}" + "".join(f"{row.get(p, 0):>14}" for p in provs))

    if len(provs) > 1:
        print(
            "    WARNING: more than one provider served this corpus. Provider "
            "differs in quantization, so this is a systematic difference "
            "between rows that is NOT persona."
        )
    if missing:
        print(
            f"    NOTE: {missing} row(s) carry no provider field -- generated "
            f"before provider logging existed. Treat them as unknown, not as "
            f"the pinned provider."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke test: only N calls, spread across the grid",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the grid and cost estimate, make no calls",
    )
    args = ap.parse_args()

    grid = build_grid(args.limit)
    print(
        f"[gen] grid: {len(grid)} items "
        f"({len(personas.PERSONAS)} personas x {len(config.EMOTIONS)} emotions "
        f"x {config.N_SCENARIOS_PER_CELL} topics)"
        + (f"  [--limit {args.limit}]" if args.limit else "")
    )
    print(f"[gen] generator: {config.GENERATOR_MODEL}")

    todo = []
    cached = 0
    for item in grid:
        body = request_for(item)
        h = content_hash(body)
        if cache_path(h).exists():
            cached += 1
        else:
            todo.append((item, body, h))

    # Shuffle DISPATCH order only.  The grid stays canonical for write_jsonl and
    # the leakage report; what changes is the sequence calls go out in, so a
    # provider outage or rate-limit window mid-run spreads across personas as
    # noise instead of landing on one persona's contiguous 360-row block. Cache
    # keys are content-based, so this invalidates nothing.
    random.Random(config.DISPATCH_SEED).shuffle(todo)

    print(f"[gen] {cached} already cached, {len(todo)} to fetch")
    print(
        f"[gen] provider: "
        + (
            f"pinned to {config.GENERATOR_PROVIDER} "
            f"(fallbacks {'on' if config.GENERATOR_ALLOW_FALLBACKS else 'OFF'})"
            if config.GENERATOR_PROVIDER
            else "free routing (NOT pinned)"
        )
    )
    print(f"[gen] dispatch order shuffled under DISPATCH_SEED={config.DISPATCH_SEED}")

    if args.dry_run or not todo:
        est_in = len(todo) * 260
        est_out = len(todo) * config.GEN_MAX_TOKENS
        cost = (
            est_in / 1e6 * config.GENERATOR_PRICE_IN_PER_MTOK
            + est_out / 1e6 * config.GENERATOR_PRICE_OUT_PER_MTOK
        )
        if args.dry_run:
            print(
                f"[gen] dry run: would cost <= ${cost:.2f} at list price "
                f"(assumes every call maxes out at {config.GEN_MAX_TOKENS} tokens)"
            )
            ex = request_for(grid[0])
            print(
                f"\n--- example request ({grid[0]['persona_id']} / "
                f"{grid[0]['emotion']}) ---"
            )
            for m in ex["messages"]:
                print(f"[{m['role']}] {m['content']}\n")
            return 0

    if todo:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("[gen] ERROR: OPENROUTER_API_KEY is not set")
            return 1
        stats = {
            "in_tok": 0,
            "out_tok": 0,
            "done": 0,
            "failed": 0,
            "providers": {},
            "fail_modes": {},
            "fail_by_persona": {},
            "fail_examples": {},
            "redraws": {},  # persona -> rejected draws, for the audit table
            "cjk_draws": 0,  # draws rejected for containing CJK
            "cjk_residue": 0,  # items still CJK after MAX_CJK_REDRAWS
        }
        t0 = time.time()
        asyncio.run(run(todo, stats))
        print(
            f"[gen] {stats['done']} fetched, {stats['failed']} failed, "
            f"{time.time() - t0:.0f}s"
        )
        print(
            f"[gen] tokens: {stats['in_tok']:,} in / {stats['out_tok']:,} out "
            f"-> ${estimated_cost(stats):.2f} estimated this run"
        )

        if stats["providers"]:
            print(
                "[gen] served by: "
                + ", ".join(
                    f"{p} x{n}"
                    for p, n in sorted(
                        stats["providers"].items(), key=lambda kv: -kv[1]
                    )
                )
            )
        # Rejection-sampling audit.  The redraw rate is expected to carry the
        # same persona gradient the CJK rate did -- rising with distance from
        # the assistant -- which is precisely why the rule has to be uniform.
        # Printed so that gradient is on the record rather than assumed away.
        if stats["cjk_draws"]:
            print(
                f"\n[gen] CJK rejection sampling: {stats['cjk_draws']} draw(s) "
                f"rejected and redrawn (cap {config.MAX_CJK_REDRAWS} per item)"
            )
            print(f"    {'rank':<5} {'persona':<18} {'redraws':>8}")
            for p in personas.PERSONAS:
                n = stats["redraws"].get(p["id"], 0)
                if n:
                    tag = "  <- CONTROL" if p["is_control"] else ""
                    print(f"    {p['distance_rank']:<5} {p['id']:<18} {n:>8}{tag}")
            if stats["cjk_residue"]:
                print(
                    f"    WARNING: {stats['cjk_residue']} item(s) exhausted the cap "
                    f"and are cached with cjk_residue=true. They are NOT dropped; "
                    f"the notebook gate will raise on them, which is your call."
                )

        if stats["failed"]:
            print(
                f"\n[gen] {stats['failed']} item(s) left uncached -- rerun to retry "
                f"just those"
            )
            print("    failure modes:")
            for mode, n in sorted(stats["fail_modes"].items(), key=lambda kv: -kv[1]):
                print(f"      {n:5d}  {mode}")
                print(f"             e.g. {stats['fail_examples'].get(mode, '')[:150]}")
            print(
                "    failures by persona (should be flat; a spike on one "
                "persona is the confound):"
            )
            for p in personas.PERSONAS:
                n = stats["fail_by_persona"].get(p["id"], 0)
                if n:
                    print(f"      {n:5d}  {p['id']}")

    n = write_jsonl(grid)
    print(f"[gen] wrote {n}/{len(grid)} records -> {config.GENERATIONS_JSONL}")
    report_providers(grid)
    report_redraws(grid)
    report_cjk(grid)
    report_leakage(grid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
