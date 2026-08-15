"""Stage 4: generate the corpus via OpenRouter.

9 personas x 12 emotions x 30 topics = 3,240 short first-person messages in
which the persona is experiencing the emotion without naming it.

No GPU involved -- this stage only talks to the API.  It is resumable: every
completed call is written to its own file in data/gen_cache/ keyed by a hash of
the exact request content, and a rerun skips anything already cached.  A run
that dies at 90% costs you the last 10%, not the whole thing.

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
    "say and how you say it. Reply with the message only."
)


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
    """The exact API request body for a grid item."""
    return {
        "model": config.GENERATOR_MODEL,
        "messages": personas.build_messages(item["persona_id"], user_prompt_for(item)),
        "temperature": config.GEN_TEMPERATURE,
        "max_tokens": config.GEN_MAX_TOKENS,
    }


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


async def call_one(client, item: dict, body: dict, h: str, stats: dict) -> dict | None:
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
            resp = await client.post(url, headers=headers, json=body,
                                     timeout=config.REQUEST_TIMEOUT_SECONDS)
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

                return {
                    **item,
                    "text": text,
                    "generator_model": config.GENERATOR_MODEL,
                    "content_hash": h,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }

            # 429 and 5xx are worth retrying; a 4xx is our fault and will not
            # fix itself.
            if resp.status_code != 429 and resp.status_code < 500:
                print(f"\n[gen] permanent {resp.status_code} for "
                      f"{item['persona_id']}/{item['emotion']}/{item['scenario_id']}: "
                      f"{resp.text[:200]}")
                stats["failed"] += 1
                return None
            raise RuntimeError(f"http {resp.status_code}")

        except Exception as exc:  # noqa: BLE001 -- retry anything transient
            if attempt == config.MAX_RETRIES - 1:
                print(f"\n[gen] gave up on {item['persona_id']}/{item['emotion']}/"
                      f"{item['scenario_id']} after {config.MAX_RETRIES}: {exc}")
                stats["failed"] += 1
                return None
            await asyncio.sleep(config.BACKOFF_BASE_SECONDS * (2 ** attempt))
    return None


def estimated_cost(stats: dict) -> float:
    return (stats["in_tok"] / 1e6 * config.GENERATOR_PRICE_IN_PER_MTOK
            + stats["out_tok"] / 1e6 * config.GENERATOR_PRICE_OUT_PER_MTOK)


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
                    cache_path(h).write_text(json.dumps(record, ensure_ascii=False))
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
        print(f"[gen] no generation names its target emotion verbatim "
              f"(0 of {n_cached}).")
        return

    by_persona, by_emotion = {}, {}
    for rec in leaked:
        by_persona[rec["persona_id"]] = by_persona.get(rec["persona_id"], 0) + 1
        by_emotion[rec["emotion"]] = by_emotion.get(rec["emotion"], 0) + 1

    print(f"\n[gen] WARNING: {len(leaked)} of {n_cached} generation(s) name the "
          f"target emotion verbatim ({len(leaked) / n_cached:.2%}, NOT dropped "
          f"-- your call):")

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
        print(f"    {p['distance_rank']:<5} {pid:<18} {n:>7} {n_tot:>6} "
              f"{rate:>6.1%}{tag}")
    if rates:
        print(f"    spread across personas: {min(rates):.1%} - {max(rates):.1%} "
              f"({(max(rates) - min(rates)) * 100:.1f}pp)")

    print(f"\n    by emotion")
    for emotion, n in sorted(by_emotion.items(), key=lambda kv: -kv[1]):
        print(f"    {n:4d}  {emotion}")
    print("\n    (synonyms are not detected -- this is a floor, not a ceiling)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only N calls, spread across the grid")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and cost estimate, make no calls")
    args = ap.parse_args()

    grid = build_grid(args.limit)
    print(f"[gen] grid: {len(grid)} items "
          f"({len(personas.PERSONAS)} personas x {len(config.EMOTIONS)} emotions "
          f"x {config.N_SCENARIOS_PER_CELL} topics)"
          + (f"  [--limit {args.limit}]" if args.limit else ""))
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

    print(f"[gen] {cached} already cached, {len(todo)} to fetch")

    if args.dry_run or not todo:
        est_in = len(todo) * 260
        est_out = len(todo) * config.GEN_MAX_TOKENS
        cost = (est_in / 1e6 * config.GENERATOR_PRICE_IN_PER_MTOK
                + est_out / 1e6 * config.GENERATOR_PRICE_OUT_PER_MTOK)
        if args.dry_run:
            print(f"[gen] dry run: would cost <= ${cost:.2f} at list price "
                  f"(assumes every call maxes out at {config.GEN_MAX_TOKENS} tokens)")
            ex = request_for(grid[0])
            print(f"\n--- example request ({grid[0]['persona_id']} / "
                  f"{grid[0]['emotion']}) ---")
            for m in ex["messages"]:
                print(f"[{m['role']}] {m['content']}\n")
            return 0

    if todo:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("[gen] ERROR: OPENROUTER_API_KEY is not set")
            return 1
        stats = {"in_tok": 0, "out_tok": 0, "done": 0, "failed": 0}
        t0 = time.time()
        asyncio.run(run(todo, stats))
        print(f"[gen] {stats['done']} fetched, {stats['failed']} failed, "
              f"{time.time() - t0:.0f}s")
        print(f"[gen] tokens: {stats['in_tok']:,} in / {stats['out_tok']:,} out "
              f"-> ${estimated_cost(stats):.2f} estimated this run")
        if stats["failed"]:
            print(f"[gen] {stats['failed']} item(s) left uncached -- rerun to retry "
                  f"just those")

    n = write_jsonl(grid)
    print(f"[gen] wrote {n}/{len(grid)} records -> {config.GENERATIONS_JSONL}")
    report_leakage(grid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
