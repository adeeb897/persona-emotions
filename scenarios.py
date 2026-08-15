"""Stage 2: the 100 topic strings.

Pulls `topic` from ryancodrai/emotion-probes (expression/stories.parquet),
deduplicates while preserving order, caches to data/topics.json, and falls back
to a hardcoded list if the fetch fails or the schema has changed.  The source
actually used is always printed and is recorded in the cache file.

    python scenarios.py            # load (from cache if present)
    python scenarios.py --refresh  # ignore cache, re-fetch from the hub
    python scenarios.py --limit 5  # smoke test
"""

import argparse
import json
import random

import config

# --------------------------------------------------------------------------
# Fallback.  Used only if the hub fetch fails or the schema does not match.
# These are ours, not the dataset's.
# --------------------------------------------------------------------------

FALLBACK_TOPICS = [
    "a job interview that went badly",
    "moving to a new city alone",
    "a childhood friend getting back in touch",
    "losing a set of keys",
    "the last day at an old job",
    "a long delayed flight",
    "cooking a meal for someone important",
    "a hospital waiting room",
    "finding an old photograph",
    "a neighbour's late night party",
    "the first day of school",
    "a car breaking down on a motorway",
    "an unexpected medical bill",
    "a wedding speech",
    "a pet going missing",
    "a promotion announced to the team",
    "a difficult phone call with a parent",
    "running a race and finishing last",
    "an exam result arriving by post",
    "a stranger returning a lost wallet",
    "a power cut during a storm",
    "a first date at a busy restaurant",
    "an argument with a sibling",
    "a package that never arrived",
    "planting a garden in spring",
    "a funeral on a rainy afternoon",
    "footsteps in an empty house",
    "a surprise party",
    "a bank account overdrawn",
    "teaching someone to drive",
    "a train journey through the mountains",
    "a broken promise",
    "a rejected loan application",
    "reading a letter from years ago",
    "a dog waiting by the door",
    "a house move on a hot day",
    "a missed deadline at work",
    "an unexpected inheritance",
    "a snowed-in weekend",
    "a public speaking engagement",
    "a lie told to protect someone",
    "the last piece of a puzzle",
    "an empty seat at a dinner table",
    "an early morning swim",
    "a mistake discovered in a report",
    "a locked door with no key",
    "a reunion after ten years",
    "a stray cat on the porch",
    "a leaking roof",
    "a birthday no one remembered",
    "a difficult diagnosis",
    "a new baby in the family",
    "a night shift in an empty office",
    "a burnt dinner for guests",
    "a marathon training run",
    "a court summons in the mail",
    "an unfinished painting",
    "a boat on a still lake",
    "a cancelled holiday",
    "a fight with a best friend",
    "winning a small local competition",
    "a broken phone screen",
    "a long queue in the rain",
    "an apology that came too late",
    "a coach's final team talk",
    "an old car sold to a stranger",
    "a candle burning in a window",
    "a child's first steps",
    "a rumour spreading at work",
    "a failed recipe attempt",
    "a scholarship offer",
    "a delayed test result",
    "a wallet stolen abroad",
    "a quiet library afternoon",
    "an unexpected visitor at midnight",
    "a promotion given to someone else",
    "a homemade gift",
    "a bridge crossing at dusk",
    "a difficult conversation with a landlord",
    "a sudden thunderstorm at a picnic",
    "an audition for a small role",
    "a message left on read",
    "a garden shed cleared out",
    "a farm at harvest time",
    "a mislaid passport before a trip",
    "a lighthouse in a storm",
    "a class reunion invitation",
    "a bee sting at a summer fair",
    "a first day as team leader",
    "an argument overheard through a wall",
    "a plant that finally flowered",
    "a wrong turn on a night drive",
    "a school play performance",
    "a doctor running late",
    "a letter of resignation",
    "a bicycle stolen from a rack",
    "a first paycheck",
    "a village festival at night",
    "an unexplained noise in the attic",
    "a goodbye at a train station",
]


def _fetch_from_hub(n: int) -> list[str]:
    """Download the parquet and pull the topic column.  Raises on any problem."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    print(f"[scenarios] downloading {config.TOPICS_DATASET}:{config.TOPICS_DATASET_FILE}")
    path = hf_hub_download(
        repo_id=config.TOPICS_DATASET,
        filename=config.TOPICS_DATASET_FILE,
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    if config.TOPICS_COLUMN not in df.columns:
        raise KeyError(
            f"column '{config.TOPICS_COLUMN}' not in parquet; columns are "
            f"{list(df.columns)}"
        )

    seen, topics = set(), []
    for value in df[config.TOPICS_COLUMN].tolist():
        s = str(value).strip()
        if s and s not in seen:
            seen.add(s)
            topics.append(s)

    print(f"[scenarios] {len(df)} rows -> {len(topics)} unique topics")
    if len(topics) < n:
        raise ValueError(f"only {len(topics)} unique topics, need {n}")
    return topics[:n]


def load_topics(n: int = config.N_TOPICS, refresh: bool = False) -> tuple[list[str], str]:
    """Return (topics, source).  source is one of: cache, huggingface, fallback."""
    if not refresh and config.TOPICS_JSON.exists():
        blob = json.loads(config.TOPICS_JSON.read_text())
        topics = blob["topics"][:n]
        print(
            f"[scenarios] SOURCE=cache ({config.TOPICS_JSON.name}, originally "
            f"'{blob.get('source')}') -- {len(topics)} topics"
        )
        return topics, "cache"

    try:
        topics = _fetch_from_hub(n)
        source = "huggingface"
    except Exception as exc:  # noqa: BLE001 -- any failure means fall back
        print(f"[scenarios] hub fetch FAILED: {type(exc).__name__}: {exc}")
        topics = FALLBACK_TOPICS[:n]
        source = "fallback"

    if source == "fallback":
        # Deliberately NOT cached: caching a fallback would mean the hub is
        # never retried once a transient failure has happened.
        print(f"[scenarios] SOURCE=fallback -- {len(topics)} topics, not cached")
        return topics, source

    config.TOPICS_JSON.write_text(
        json.dumps(
            {
                "source": source,
                "dataset": config.TOPICS_DATASET,
                "file": config.TOPICS_DATASET_FILE,
                "column": config.TOPICS_COLUMN,
                "n": len(topics),
                "topics": topics,
            },
            indent=2,
        )
    )
    print(f"[scenarios] SOURCE={source} -- {len(topics)} topics -> {config.TOPICS_JSON}")
    return topics, source


def sample_scenarios(
    n: int = config.N_SCENARIOS_PER_CELL,
    seed: int = config.TOPIC_SAMPLE_SEED,
    refresh: bool = False,
) -> list[tuple[int, str]]:
    """The fixed topic subset used in EVERY (persona, emotion) cell.

    Returns [(scenario_id, topic), ...] where scenario_id indexes into the full
    100-topic list, so ids stay meaningful if n ever changes.  Random sample
    under `seed`, sorted by id for stable output; deliberately not curated.
    """
    topics, _ = load_topics(refresh=refresh)
    ids = sorted(random.Random(seed).sample(range(len(topics)), n))
    return [(i, topics[i]) for i in ids]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true", help="show the fixed 30-topic subset")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-fetch")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only N topics")
    args = ap.parse_args()

    if args.sample:
        chosen = sample_scenarios(refresh=args.refresh)
        print(
            f"\nfixed subset: {len(chosen)} of {config.N_TOPICS} topics, "
            f"seed={config.TOPIC_SAMPLE_SEED}, used in every cell:"
        )
        for sid, topic in chosen[: args.limit or len(chosen)]:
            print(f"  {sid:3d}  {topic}")
        raise SystemExit(0)

    topics, source = load_topics(refresh=args.refresh)
    show = topics[: args.limit] if args.limit else topics
    for i, t in enumerate(show):
        print(f"  {i:3d}  {t}")
    print(f"\n{len(topics)} topics loaded (source={source}), showing {len(show)}")
    if source == "fallback":
        print("WARNING: using the hardcoded fallback list, NOT the published dataset.")
