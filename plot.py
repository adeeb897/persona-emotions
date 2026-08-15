"""Stage 8: the figures.

  1. accuracy_vs_distance.png  -- the key figure: probe accuracy against measured
     distance along the assistant axis, one point per persona. bare_template is
     drawn in a different colour AND a different marker AND its own legend entry,
     so it reads as a control rather than as a ninth persona. Points are means
     over the topic folds; error bars are +/-1 SD across those folds.
  2. confusion_matrix.png      -- 12x12 for the default-persona probe, pooled
     over folds (each row predicted once, by the probe that skipped its topic).
  3. accuracy_by_layer.png     -- how the layer choice was made.

Each figure also writes a .csv twin, so no value is reachable only by reading a
colour off the page.

    python plot.py
    python plot.py --layer 38    # override the layer chosen by probes.py
"""

import argparse
import csv
import json
import sys

import config
import personas

# Palette roles (light surface). Categorical slots 1 and 2 -- the documented
# all-pairs-safe subset -- carry persona vs control; everything else is ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_PERSONA = "#2a78d6"  # slot 1, blue
SERIES_CONTROL = "#eb6834"  # slot 2, orange
# Sequential blue ramp, light -> dark (steps 100..700).
SEQ_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def style(plt):
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # solid hairline; never dashed
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"[plot] table -> {path}")


def fig_accuracy_vs_distance(plt, probe, axis, layer):
    """The key figure."""
    per_layer = probe["per_layer"][str(layer)]
    distance = axis["per_layer"][str(layer)]["distance"]
    default = probe["default_persona"]
    control = probe["control_persona"]
    n_folds = probe["cv"]["n_folds"]

    # Every accuracy here is a mean over the topic folds; the error bar is the
    # SD of those per-fold values -- the spread of the thing being plotted, not
    # a confidence interval. SEM is in probe_results.json if you want it, but
    # the folds share training rows, so neither is a clean standard error.
    points = []
    for pid in [p["id"] for p in personas.PERSONAS]:
        if pid not in distance:
            continue
        if pid == default:
            acc, sd = (
                per_layer["acc_default_heldout"],
                per_layer["acc_default_heldout_sd"],
            )
        else:
            t = per_layer["transfer"].get(pid, {})
            acc, sd = t.get("acc_heldout_scenarios"), t.get("acc_heldout_scenarios_sd")
        if acc is None:
            continue
        points.append((pid, distance[pid], acc, sd or 0.0, pid == control))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for is_control, colour, marker, label in (
        (False, SERIES_PERSONA, "o", "persona"),
        (True, SERIES_CONTROL, "D", f"{control} (control)"),
    ):
        sel = [p for p in points if p[4] is is_control]
        if not sel:
            continue
        ax.errorbar(
            [p[1] for p in sel],
            [p[2] for p in sel],
            yerr=[p[3] for p in sel],
            fmt="none",
            ecolor=colour,
            elinewidth=1.4,
            capsize=3,
            capthick=1.4,
            alpha=0.7,
            zorder=2,
        )
        ax.scatter(
            [p[1] for p in sel],
            [p[2] for p in sel],
            s=90 if not is_control else 80,
            c=colour,
            marker=marker,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=2,  # 2px surface ring, not a border
            label=label,
        )

    # Personas cluster hard near the anchor, so labels must be placed, not just
    # offset by a constant. Greedy: try above, then below, then further out,
    # taking the first slot that does not collide with an already-placed label.
    ax.set_ylim(0, 1.15)  # headroom for labels on points at 1.0
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    x_lo, x_hi = min(p[1] for p in points), max(p[1] for p in points)
    x_pad = (x_hi - x_lo) * 0.08 or 0.05
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)

    def frac(x, y):
        (a, b), (c, d) = ax.get_xlim(), ax.get_ylim()
        return (x - a) / (b - a), (y - c) / (d - c)

    AXES_H_POINTS = 5.0 * 72 * 0.78  # figure height in, minus chrome
    AXES_W_POINTS = 7.2 * 72 * 0.78  # ... and width, for label half-widths
    LABEL_HALF_H = 0.019  # half a line of 8.5pt text, in axes frac
    MARKER_R = 0.014  # marker radius + its surface ring

    def half_w(text):
        return len(text) * 4.6 / 2 / AXES_W_POINTS  # 8.5pt is ~4.6pt per char

    # Markers and their error bars are obstacles too, not just other labels.
    # Without this a label finds a slot that is empty of text and lands squarely
    # on someone else's point -- which is what "peasant" did to bare_template's
    # marker the first time this figure was drawn on real data.
    obstacles = []
    for pid, x, acc, sd, _ in points:
        fx, fy_lo = frac(x, acc - sd)
        _, fy_hi = frac(x, acc + sd)
        obstacles.append((pid, fx, fy_lo - MARKER_R, fy_hi + MARKER_R))

    placed = []
    for pid, x, acc, sd, _ in sorted(points, key=lambda p: p[1]):
        w = half_w(pid)
        for dy in (0.035, -0.050, 0.080, -0.095, 0.125, -0.140, 0.170):
            # Hang the label off the end of the error bar, not off the marker,
            # so a wide bar cannot grow into its own label.
            anchor = acc + sd if dy > 0 else acc - sd
            fx, fy = frac(x, anchor)
            cy = fy + dy
            hits_label = any(
                abs(fx - px) < w + pw and abs(cy - py) < 2 * LABEL_HALF_H
                for px, py, pw in placed
            )
            hits_point = any(
                opid != pid
                and abs(fx - ofx) < w + MARKER_R
                and cy + LABEL_HALF_H > o_lo
                and cy - LABEL_HALF_H < o_hi
                for opid, ofx, o_lo, o_hi in obstacles
            )
            if not (hits_label or hits_point):
                break
        placed.append((fx, fy + dy, w))
        # A label pushed past its first slot is no longer obviously attached to
        # its own point -- with two points a hair apart on x, "peasant" above
        # "detective" reads as either. A hairline leader removes the ambiguity;
        # points that got their default slot are cleaner without one.
        leader = (
            dict(
                arrowstyle="-",
                linewidth=0.6,
                color=MUTED,
                shrinkA=1.5,
                shrinkB=3.0,
                alpha=0.85,
            )
            if abs(dy) > 0.05
            else None
        )
        ax.annotate(
            pid,
            (x, anchor),
            textcoords="offset points",
            xytext=(0, dy * AXES_H_POINTS),
            ha="center",
            fontsize=8.5,
            color=INK_SECONDARY,
            arrowprops=leader,
        )

    chance = probe["chance"]
    ax.axhline(chance, color=MUTED, linewidth=0.8, zorder=1)
    ax.annotate(
        f"chance ({chance:.2f})",
        (ax.get_xlim()[1], chance),
        textcoords="offset points",
        xytext=(-4, 5),
        ha="right",
        fontsize=8.5,
        color=MUTED,
    )

    ax.set_xlabel(
        f"distance along the assistant axis "
        f"({default} = 0, {probe.get('far_persona', axis['far_persona'])} = 1)"
    )
    ax.set_ylabel(f"probe accuracy, held-out topics (mean of {n_folds} folds)")
    ax.set_title(
        f"Emotion probe transfer across personas\n"
        f"trained on {default}, layer {layer}\n"
        f"error bars: +/-1 SD across the {n_folds} topic folds",
        color=INK,
        fontsize=12,
        loc="left",
        pad=14,
    )
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()

    out = config.FIG_DIR / "accuracy_vs_distance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")

    def sem_of(pid):
        if pid == default:
            return per_layer["acc_default_heldout_sem"]
        return per_layer["transfer"].get(pid, {}).get("acc_heldout_scenarios_sem")

    write_csv(
        config.FIG_DIR / "accuracy_vs_distance.csv",
        [
            "persona_id",
            "is_control",
            "axis_distance",
            "accuracy_heldout_scenarios",
            "sd_across_folds",
            "sem_across_folds",
            "n_folds",
        ],
        [
            [
                pid,
                int(ctl),
                f"{x:.4f}",
                f"{acc:.4f}",
                f"{sd:.4f}",
                "" if sem_of(pid) is None else f"{sem_of(pid):.4f}",
                n_folds,
            ]
            for pid, x, acc, sd, ctl in points
        ],
    )


def fig_confusion(plt, probe, layer):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    per_layer = probe["per_layer"][str(layer)]
    classes = per_layer["classes"]
    cm = np.array(per_layer["confusion_matrix"], dtype=float)
    row_totals = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, row_totals, out=np.zeros_like(cm), where=row_totals > 0)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.grid(False)
    im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(classes)), classes, fontsize=8.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        f"Default-persona probe, layer {layer}\n"
        f"cell = count; shade = fraction of true class\n"
        f"pooled over {probe['cv']['n_folds']} folds; each row predicted once",
        color=INK,
        fontsize=12,
        loc="left",
        pad=14,
    )

    # Values on every cell: for a confusion matrix the annotated grid IS the
    # table view, so the number is never reachable by colour alone.
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] == 0:
                continue
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if frac[i, j] > 0.55 else INK,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("fraction of true class", color=INK_SECONDARY, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    fig.tight_layout()
    out = config.FIG_DIR / "confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")

    write_csv(
        config.FIG_DIR / "confusion_matrix.csv",
        ["true"] + [f"pred_{c}" for c in classes],
        [[classes[i]] + [int(v) for v in cm[i]] for i in range(len(classes))],
    )


def fig_accuracy_by_layer(plt, probe):
    layers = probe["layers"]
    accs = [probe["per_layer"][str(l)]["acc_default_heldout"] for l in layers]
    sds = [probe["per_layer"][str(l)]["acc_default_heldout_sd"] or 0.0 for l in layers]
    best = probe["best_layer"]
    n_folds = probe["cv"]["n_folds"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(
        layers,
        accs,
        yerr=sds,
        fmt="none",
        ecolor=SERIES_PERSONA,
        elinewidth=1.4,
        capsize=3,
        capthick=1.4,
        alpha=0.7,
        zorder=2,
    )
    ax.plot(
        layers,
        accs,
        color=SERIES_PERSONA,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=3,
    )
    ax.axhline(probe["chance"], color=MUTED, linewidth=0.8, zorder=1)

    bi = layers.index(best)
    ax.annotate(
        f"best: layer {best} ({accs[bi]:.3f})",
        (best, accs[bi]),
        textcoords="offset points",
        xytext=(0, 14),
        ha="center",
        fontsize=9,
        color=INK_SECONDARY,
    )

    ax.set_xlabel("layer")
    ax.set_ylabel(f"accuracy, held-out topics (mean of {n_folds} folds)")
    ax.set_title(
        "Default-persona probe accuracy by layer\n"
        "the layer is chosen here, never on transfer performance\n"
        f"error bars: +/-1 SD across the {n_folds} topic folds",
        color=INK,
        fontsize=12,
        loc="left",
        pad=14,
    )
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.15)  # headroom: the "best" label and the top error
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # bar both sit above a 1.0 point
    fig.tight_layout()

    out = config.FIG_DIR / "accuracy_by_layer.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")

    write_csv(
        config.FIG_DIR / "accuracy_by_layer.csv",
        ["layer", "acc_default_heldout", "sd_across_folds", "n_folds"],
        [[l, f"{a:.4f}", f"{s:.4f}", n_folds] for l, a, s in zip(layers, accs, sds)],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="override the best layer chosen by probes.py",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="accepted for pipeline symmetry; plots whatever exists",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for path, stage in (
        (config.PROBE_RESULTS_JSON, "probes.py"),
        (config.AXIS_RANKING_JSON, "assistant_axis.py"),
    ):
        if not path.exists():
            print(f"[plot] ERROR: {path} not found -- run {stage} first")
            return 1

    probe = json.loads(config.PROBE_RESULTS_JSON.read_text())
    axis = json.loads(config.AXIS_RANKING_JSON.read_text())
    layer = args.layer if args.layer is not None else probe["best_layer"]
    if str(layer) not in probe["per_layer"]:
        print(f"[plot] ERROR: layer {layer} not in {probe['layers']}")
        return 1

    style(plt)
    print(
        f"[plot] layer {layer}"
        + (" (override)" if args.layer is not None else " (best, from probes.py)")
    )

    fig_accuracy_vs_distance(plt, probe, axis, layer)
    fig_confusion(plt, probe, layer)
    fig_accuracy_by_layer(plt, probe)
    print(f"\n[plot] figures in {config.FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
