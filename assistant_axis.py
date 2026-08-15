"""Stage 6: the assistant axis.

Difference of means between default-persona and most-distant-persona
activations, unit-normalised, computed independently per hooked layer.  Every
persona's mean activation is then projected onto it and the ranking saved.

The axis is anchored on `default_assistant`, NEVER on `bare_template`.
bare_template carries no system message, so its prompt is both much shorter
than the personas' and written by the chat template rather than by us --
anchoring there would risk measuring prompt length instead of persona.  It is
projected like everything else and reported as a control.

    python assistant_axis.py
"""

import argparse
import json
import sys

import config
import personas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="accepted for pipeline symmetry; the axis always uses all rows")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    if not config.ACTIVATIONS_NPY.exists():
        print(f"[axis] ERROR: {config.ACTIVATIONS_NPY} not found -- run extract.py first")
        return 1

    acts = np.load(config.ACTIVATIONS_NPY)          # (n_rows, n_layers, hidden)
    meta = pd.read_parquet(config.ACTIVATIONS_META)
    info = json.loads(config.ACTIVATIONS_INFO.read_text())
    layers = info["layers"]
    print(f"[axis] acts {acts.shape}, layers {layers}")

    anchor, far = personas.DEFAULT_PERSONA_ID, personas.MOST_DISTANT_PERSONA_ID
    is_anchor = (meta["persona_id"] == anchor).to_numpy()
    is_far = (meta["persona_id"] == far).to_numpy()

    for name, mask in ((anchor, is_anchor), (far, is_far)):
        if not mask.any():
            print(f"[axis] ERROR: no rows for '{name}' -- cannot define the axis. "
                  f"Personas present: {sorted(meta['persona_id'].unique())}")
            return 1
    print(f"[axis] anchor '{anchor}' n={int(is_anchor.sum())}, "
          f"far '{far}' n={int(is_far.sum())}")

    present = [p["id"] for p in personas.PERSONAS if (meta["persona_id"] == p["id"]).any()]

    axis = np.zeros((len(layers), acts.shape[2]), dtype=np.float32)
    per_layer = {}

    for li, layer in enumerate(layers):
        mu_anchor = acts[is_anchor, li].mean(axis=0)
        mu_far = acts[is_far, li].mean(axis=0)
        raw = mu_anchor - mu_far
        norm = float(np.linalg.norm(raw))
        axis[li] = raw / norm                       # unit-normalised

        # Project each persona's mean activation onto the unit axis.
        proj = {
            pid: float(acts[(meta["persona_id"] == pid).to_numpy(), li].mean(axis=0) @ axis[li])
            for pid in present
        }

        # Rescale so the anchor sits at 0 and the far endpoint at 1, giving a
        # readable "distance from the default assistant" for the x-axis of the
        # headline plot.  Larger = further from the assistant persona.
        p_anchor, p_far = proj[anchor], proj[far]
        span = p_anchor - p_far
        distance = {pid: float((p_anchor - v) / span) for pid, v in proj.items()}

        per_layer[str(layer)] = {
            "diff_of_means_norm": norm,
            "projection": proj,
            "distance": distance,
            "ranking": sorted(distance, key=distance.get),
        }

        print(f"\n[axis] layer {layer}  ||mu_anchor - mu_far|| = {norm:.2f}")
        print(f"    {'persona':<18} {'distance':>9}  {'projection':>11}")
        for pid in sorted(distance, key=distance.get):
            tag = "  <- CONTROL" if personas.PERSONAS_BY_ID[pid]["is_control"] else ""
            print(f"    {pid:<18} {distance[pid]:>9.3f}  {proj[pid]:>11.2f}{tag}")

    np.save(config.AXIS_NPY, axis)
    config.AXIS_RANKING_JSON.write_text(json.dumps({
        "anchor_persona": anchor,
        "far_persona": far,
        "control_persona": personas.CONTROL_PERSONA_ID,
        "layers": layers,
        "note": "distance is rescaled so anchor=0 and far=1; larger is further "
                "from the default assistant. The control never defines the axis.",
        "per_layer": per_layer,
    }, indent=2))

    print(f"\n[axis] axis {axis.shape} -> {config.AXIS_NPY}")
    print(f"[axis] ranking -> {config.AXIS_RANKING_JSON}")

    # The control is the whole point of stage 6 -- say something about it.
    ctrl = personas.CONTROL_PERSONA_ID
    if ctrl in present:
        d = [per_layer[str(l)]["distance"][ctrl] for l in layers]
        mean_d = sum(d) / len(d)
        print(f"\n[axis] CONTROL '{ctrl}' sits at mean distance {mean_d:.3f} "
              f"(anchor=0, {far}=1)")
        if mean_d < 0.25:
            print("       -> close to the anchor: prompt length is not driving the axis.")
        else:
            print("       -> NOT close to the anchor: the axis may be picking up "
                  "prompt length / template effects, not just persona. Real confound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
