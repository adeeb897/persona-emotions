"""Stage 7: 12-way emotion probes, cross-validated over topics.

(a) Train and test on the default persona, held-out topics.
(b) Train on the default persona, test on each other persona separately.

Both are reported per layer, so the best layer can be chosen on (a) -- i.e. on
default-persona performance alone, never on transfer performance, which would
leak the result being measured.

Splits are over TOPICS, not rows, and there are `config.N_FOLDS` of them: the 30
topics are cut into 5 folds of 6, and each fold is held out in turn.  Every
topic is therefore tested exactly once across the run, and no probe is ever
tested on a topic it trained on.  A single 75/25 split left ~6-10 test rows per
class per persona for a 12-way problem -- too few to read the headline
relationship off, hence the CV.

Cross-persona transfer is scored on the SAME held-out topics as (a), fold by
fold, so a drop in (b) cannot be explained by the probe having seen those topics
during training -- (a) and (b) differ only in persona.  The all-topics variant
is kept alongside as a secondary number because it costs nothing; it is
systematically the more optimistic of the two.

Every accuracy is the MEAN over folds, and is stored with its per-fold values,
SD and SEM so the figures can show the spread.

    python probes.py --limit 8    # smoke test (will warn: too few samples)
    python probes.py
"""

import argparse
import json
import math
import sys

import config
import personas


def make_folds(scenario_ids, n_folds, seed):
    """Cut the topics into `n_folds` disjoint held-out sets.

    Seeded, so the same topics land in the same fold on every rerun, and sorted
    within a fold purely for readable output.  Each topic appears in exactly one
    fold, which is what makes "tested exactly once" true.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(scenario_ids))
    return [np.sort(chunk) for chunk in np.array_split(shuffled, n_folds)
            if len(chunk)]


def agg(values):
    """(mean, SD, SEM) over the folds that produced a number.

    Folds with nothing to score contribute None and are dropped rather than
    counted as zero.  SD/SEM are None below two folds -- reporting 0.0 there
    would draw an error bar that means "one sample", not "no variance".
    """
    import numpy as np

    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return mean, None, None
    sd = float(np.std(vals, ddof=1))
    return mean, sd, sd / math.sqrt(len(vals))


def fmt(mean, sd, width=9):
    if mean is None:
        return f"{'n/a':>{width}}"
    s = f"{mean:.3f}" + (f" +/-{sd:.3f}" if sd is not None else "")
    return f"{s:>{width}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only N rows")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not config.ACTIVATIONS_NPY.exists():
        print(f"[probes] ERROR: {config.ACTIVATIONS_NPY} not found -- run extract.py first")
        return 1

    acts = np.load(config.ACTIVATIONS_NPY)
    meta = pd.read_parquet(config.ACTIVATIONS_META)
    info = json.loads(config.ACTIVATIONS_INFO.read_text())
    layers = info["layers"]

    if args.limit is not None:
        acts, meta = acts[: args.limit], meta.iloc[: args.limit].reset_index(drop=True)
    print(f"[probes] acts {acts.shape}, layers {layers}")

    default = personas.DEFAULT_PERSONA_ID
    persona_ids = meta["persona_id"].to_numpy()
    scenario_ids = meta["scenario_id"].to_numpy()
    y = meta["emotion"].to_numpy()
    is_default = persona_ids == default

    if not is_default.any():
        print(f"[probes] ERROR: no rows for the default persona '{default}'.")
        return 1

    # Fixed class list, taken from ALL default-persona rows rather than from one
    # fold's training split, so per-fold confusion matrices are the same shape
    # and can be pooled.
    classes = sorted(np.unique(y[is_default]).tolist())
    n_classes = len(classes)
    chance = 1.0 / n_classes
    if n_classes < 2:
        print("[probes] ERROR: fewer than 2 emotion classes in the default-persona "
              "rows -- nothing to fit. Expected under a small --limit.")
        return 1
    if n_classes < len(config.EMOTIONS):
        print(f"[probes] WARNING: only {n_classes} of {len(config.EMOTIONS)} emotions "
              f"present in the default-persona rows -- accuracies are not comparable "
              f"to a full run.")

    # --- topic-level folds ------------------------------------------------
    all_scenarios = np.array(sorted(meta["scenario_id"].unique()))
    n_folds = min(config.N_FOLDS, len(all_scenarios))
    if n_folds < config.N_FOLDS:
        print(f"[probes] WARNING: only {len(all_scenarios)} topic(s) present -- "
              f"using {n_folds} fold(s) instead of {config.N_FOLDS}.")
    folds = make_folds(all_scenarios, n_folds, config.RANDOM_SEED)

    print(f"[probes] {len(all_scenarios)} topics -> {len(folds)} folds "
          f"({', '.join(str(len(f)) for f in folds)} topics held out); "
          f"each topic tests exactly once")
    for k, f in enumerate(folds):
        print(f"[probes]   fold {k} test topics: {f.tolist()}")

    other_personas = [p["id"] for p in personas.PERSONAS
                      if p["id"] != default and (persona_ids == p["id"]).any()]

    # --- per layer, per fold ----------------------------------------------
    results = {}
    skipped_reported = False
    for li, layer in enumerate(layers):
        X = acts[:, li, :]

        fold_default = []
        fold_transfer = {pid: {"held": [], "all": []} for pid in other_personas}
        cm_total = np.zeros((n_classes, n_classes), dtype=int)
        n_default_scored = 0
        n_transfer_scored = {pid: {"held": 0, "all": 0} for pid in other_personas}
        n_used = 0

        for k, test_topics in enumerate(folds):
            in_fold = np.isin(scenario_ids, test_topics)
            train_mask = is_default & ~in_fold

            if train_mask.sum() < 2 or len(np.unique(y[train_mask])) < 2:
                if not skipped_reported:
                    print(f"[probes] WARNING: fold {k} has too little default-persona "
                          f"training data to fit -- skipped. Expected under --limit.")
                fold_default.append(None)
                for pid in other_personas:
                    fold_transfer[pid]["held"].append(None)
                    fold_transfer[pid]["all"].append(None)
                continue
            n_used += 1

            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=config.LOGREG_MAX_ITER, C=config.LOGREG_C),
            )
            clf.fit(X[train_mask], y[train_mask])

            # (a) same persona, held-out topics of this fold.
            m_def = is_default & in_fold
            if m_def.any():
                pred = clf.predict(X[m_def])
                fold_default.append(float(accuracy_score(y[m_def], pred)))
                cm_total += confusion_matrix(y[m_def], pred, labels=classes)
                n_default_scored += int(m_def.sum())
            else:
                fold_default.append(None)

            # (b) other personas, scored on THIS fold's held-out topics.
            for pid in other_personas:
                m_held = (persona_ids == pid) & in_fold
                m_all = persona_ids == pid
                fold_transfer[pid]["held"].append(
                    float(accuracy_score(y[m_held], clf.predict(X[m_held])))
                    if m_held.any() else None)
                fold_transfer[pid]["all"].append(
                    float(accuracy_score(y[m_all], clf.predict(X[m_all])))
                    if m_all.any() else None)
                n_transfer_scored[pid]["held"] += int(m_held.sum())
                n_transfer_scored[pid]["all"] += int(m_all.sum())

        skipped_reported = True

        if n_used == 0:
            print("[probes] ERROR: no fold had enough default-persona training data "
                  "to fit a probe. This is expected under a small --limit; run the "
                  "full pipeline for real numbers.")
            return 1

        mean_d, sd_d, sem_d = agg(fold_default)
        if mean_d is None:
            print("[probes] ERROR: no fold produced held-out default-persona rows, so "
                  "the layer cannot be selected on (a).")
            return 1

        transfer = {}
        for pid in other_personas:
            m_h, sd_h, sem_h = agg(fold_transfer[pid]["held"])
            m_a, sd_a, sem_a = agg(fold_transfer[pid]["all"])
            transfer[pid] = {
                # primary: same held-out topics as (a), so only persona differs
                "acc_heldout_scenarios": m_h,
                "acc_heldout_scenarios_sd": sd_h,
                "acc_heldout_scenarios_sem": sem_h,
                "acc_heldout_scenarios_folds": fold_transfer[pid]["held"],
                "n_heldout": n_transfer_scored[pid]["held"],
                # secondary: includes topics the probe trained on (other persona)
                "acc_all_scenarios": m_a,
                "acc_all_scenarios_sd": sd_a,
                "acc_all_scenarios_sem": sem_a,
                "acc_all_scenarios_folds": fold_transfer[pid]["all"],
                "n_all": n_transfer_scored[pid]["all"],
            }

        results[str(layer)] = {
            "acc_default_heldout": mean_d,
            "acc_default_heldout_sd": sd_d,
            "acc_default_heldout_sem": sem_d,
            "acc_default_heldout_folds": fold_default,
            "n_default_heldout": n_default_scored,
            "n_folds_used": n_used,
            "classes": classes,
            # Pooled over folds: every default-persona row is predicted exactly
            # once, by the one probe that did not train on its topic.
            "confusion_matrix": cm_total.tolist(),
            "transfer": transfer,
        }

    # --- best layer, chosen on (a) only -----------------------------------
    best_layer = max(results, key=lambda k: results[k]["acc_default_heldout"])
    print(f"\n[probes] per-layer accuracy on the default persona, CV mean over "
          f"{len(folds)} folds +/- SD (chance = {chance:.3f}):")
    for layer in layers:
        r = results[str(layer)]
        mark = "  <- best" if str(layer) == best_layer else ""
        print(f"    layer {layer:<4} "
              f"{fmt(r['acc_default_heldout'], r['acc_default_heldout_sd'], 14)}{mark}")

    br = results[best_layer]
    print(f"\n[probes] transfer at best layer {best_layer}, CV mean +/- SD "
          f"(primary = held-out topics, the same folds as (a)):")
    print(f"    {'persona':<18} {'held-out':>14} {'all topics':>14}")
    print(f"    {default:<18} "
          f"{fmt(br['acc_default_heldout'], br['acc_default_heldout_sd'], 14)} "
          f"{'(trained)':>14}")
    for pid in other_personas:
        t = br["transfer"][pid]
        tag = "  <- CONTROL" if personas.PERSONAS_BY_ID[pid]["is_control"] else ""
        print(f"    {pid:<18} "
              f"{fmt(t['acc_heldout_scenarios'], t['acc_heldout_scenarios_sd'], 14)} "
              f"{fmt(t['acc_all_scenarios'], t['acc_all_scenarios_sd'], 14)}{tag}")

    config.PROBE_RESULTS_JSON.write_text(json.dumps({
        "model": info["model"],
        "layers": layers,
        "best_layer": int(best_layer),
        "best_layer_selected_on": "acc_default_heldout (CV mean over folds)",
        "chance": chance,
        "n_classes": n_classes,
        "default_persona": default,
        "control_persona": personas.CONTROL_PERSONA_ID,
        "cv": {
            "scheme": "k-fold over topics",
            "n_folds": len(folds),
            "seed": config.RANDOM_SEED,
            "folds": [f.tolist() for f in folds],
            "note": "each topic is held out in exactly one fold; transfer is "
                    "scored on the same held-out topics as the default persona, "
                    "fold by fold.",
        },
        "aggregation": "every accuracy is the mean over folds; *_sd and *_sem are "
                       "the spread of the per-fold values, and *_folds keeps them.",
        "primary_metric": "acc_heldout_scenarios",
        "per_layer": results,
    }, indent=2))
    print(f"\n[probes] -> {config.PROBE_RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
