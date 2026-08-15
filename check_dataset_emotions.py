"""One-off check: do our 12 emotion strings appear verbatim in the dataset's
`emotion` column?

This deliberately does not fix anything or fall back to anything.  It prints
the mismatch and exits non-zero so a discrepancy is impossible to miss.

    python check_dataset_emotions.py
"""

import sys

import config


def main() -> int:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    print(f"[check] {config.TOPICS_DATASET}:{config.TOPICS_DATASET_FILE}")
    try:
        path = hf_hub_download(
            repo_id=config.TOPICS_DATASET,
            filename=config.TOPICS_DATASET_FILE,
            repo_type="dataset",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[check] FAILED to download: {type(exc).__name__}: {exc}")
        return 2

    df = pd.read_parquet(path)
    print(f"[check] {len(df)} rows, columns: {list(df.columns)}")

    if config.EMOTION_COLUMN not in df.columns:
        print(f"[check] MISMATCH: no '{config.EMOTION_COLUMN}' column in this file.")
        return 1

    values = sorted({str(v).strip() for v in df[config.EMOTION_COLUMN].tolist()})
    counts = df[config.EMOTION_COLUMN].astype(str).str.strip().value_counts()

    print(f"\n[check] dataset has {len(values)} distinct emotion values:")
    for v in values:
        print(f"    {counts[v]:6d}  {v}")

    ours = set(config.EMOTIONS)
    theirs = set(values)
    present = [e for e in config.EMOTIONS if e in theirs]
    missing = [e for e in config.EMOTIONS if e not in theirs]
    extra = sorted(theirs - ours)

    print(f"\n[check] our 12 emotions, verbatim match:")
    for e in config.EMOTIONS:
        mark = "OK  " if e in theirs else "MISS"
        n = int(counts[e]) if e in theirs else 0
        print(f"    {mark}  {e:<12} {n}")

    print(f"\n[check] {len(present)}/{len(config.EMOTIONS)} of ours present verbatim")
    if missing:
        print(f"[check] MISSING from dataset: {missing}")
    if extra:
        print(f"[check] in dataset but not in our list: {extra}")

    if missing:
        print(
            "\n[check] MISMATCH -- decide what you want to do about it "
            "(rename ours, or accept that the topic strings were written "
            "against a different emotion set)."
        )
        return 1

    print("\n[check] all 12 present verbatim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
