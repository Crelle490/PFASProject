#!/usr/bin/env python
"""
Plot a Gantt-style timing map from system_device_activity_vectors.npz

- X-axis: time [s]
- Y-axis: devices (pumps, valves, sensors)
- Color:
    * Pump: dark blue ~ +1, light blue ~ 0, red ~ -1
    * Valve: orange when 1
    * Sensor: green when 1

Usage:
    python plots_device_activity.py
    python plots_device_activity.py path/to/file.npz
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.transforms import blended_transform_factory

# ---------- Manual phase boundaries (hard-coded) ----------
# These are the times [s] where the background color should change.
MANUAL_PHASE_BOUNDARIES = [
    31.0,
    58.0,
    81.0,
    258.0,
    285.0,
    299.0,
    322.0,
    497.0,
    523.0,
    536.0,
    561.0,
]

# Subtitles for each segment (between successive edges)
# Use '\n' for multi-line labels.
SUB_LABELS = [
    "prime\nlines",
    "create\nsolution",
    "reactor",
    "sample",
    "recircle",
    "read",
    "reactor w/ catalyst",          # <-- one line
    "sample",
    "recircle",
    "read",
    "empty tubes and chambers",     # <-- one line
    "",  # last (after final boundary) left blank
]

# Major sections (big titles)
MAJOR_SECTIONS = [
    ("Initialize", 0, 2),   # segments 0–1
    ("Cycle 1",   2, 6),    # segments 2–5
    ("Cycle 2",   6, 10),   # segments 6–9
    ("Exit",      10, 12),  # segments 10–11
]


# ---------- Helpers for categorisation & labels ----------

def categorize_device(name: str) -> str:
    """
    Return 'pump', 'valve', or 'sensor' based on name.
    """
    lname = name.lower()

    if lname in {"valve1", "valve2"}:
        return "valve"

    if lname.startswith("sensor_") or lname in {"ph", "fluoride"}:
        return "sensor"

    return "pump"


def pretty_label(name: str) -> str:
    """
    Human-friendly y-axis labels.
    """
    cat = categorize_device(name)
    lname = name.lower()

    if cat == "pump":
        if lname == "pump_mix":
            return "Pump reactor"
        if lname == "pump_holding_to_valves":
            return "Pump holding to valves"
        return f"Pump {name}"
    elif cat == "sensor":
        if lname == "sensor_fluoride":
            return "sensor fluoride"
        return name
    else:
        return name


# ---------- Color mapping ----------

def pump_color(value: float):
    """
    Map pump value in [-1, 1] to RGB:
        - negative -> red scale (light at 0, dark at -1)
        - positive -> blue scale (light at 0, dark at +1)
    """
    v = max(-1.0, min(1.0, float(value)))

    if v >= 0:
        light = np.array([0.776, 0.859, 0.937])   # ~ #c6dbef
        dark  = np.array([0.031, 0.318, 0.612])   # ~ #08519c
        rgb = light + (dark - light) * v
    else:
        light = np.array([0.988, 0.733, 0.631])   # ~ #fcbba1
        dark  = np.array([0.796, 0.094, 0.114])   # ~ #cb181d
        w = abs(v)
        rgb = light + (dark - light) * w

    return tuple(rgb)


VALVE_COLOR = "#fdae6b"   # orange
SENSOR_COLOR = "#31a354"  # green


# ---------- Loading ----------

def load_device_activity(npz_path: Path):
    data = np.load(npz_path)

    if "time_s" not in data.files:
        raise KeyError(f"'time_s' not found in {npz_path}; keys: {data.files}")

    t_axis = data["time_s"]
    device_names = [k for k in data.files if k != "time_s"]
    device_names.sort()

    vectors = {name: data[name] for name in device_names}
    return t_axis, device_names, vectors


# ---------- Plotting ----------

def plot_timing_map(
    t_axis,
    device_names,
    vectors,
    out_path: Path,
    activity_eps: float = 1e-3,
):
    """
    Build the timing map using the stored device vectors.
    """
    # Filter out buffer, water, and pH sensor
    exclude = {"buffer", "water", "sensor_pH"}
    plot_devices = [d for d in device_names if d not in exclude]
    n_dev = len(plot_devices)

    t_min, t_max = float(t_axis[0]), float(t_axis[-1])
    dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 1.0
    t_end = t_max + dt

    fig_height = 0.6 * n_dev + 2
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # --- Background phase bands ---
    phase_edges = [t_min]
    for b in MANUAL_PHASE_BOUNDARIES:
        if t_min < b < t_end:
            phase_edges.append(b)
    phase_edges.append(t_end)
    phase_edges = sorted(set(phase_edges))

    if len(SUB_LABELS) != len(phase_edges) - 1:
        raise ValueError(
            f"SUB_LABELS has {len(SUB_LABELS)} entries but there are "
            f"{len(phase_edges) - 1} segments."
        )

    for i in range(len(phase_edges) - 1):
        start = phase_edges[i]
        end = phase_edges[i + 1]
        if i % 2 == 0:
            color = (0.93, 0.96, 1.0)
        else:
            color = (1.0, 0.94, 0.95)
        ax.axvspan(start, end, color=color, alpha=0.5, zorder=0)

    # --- Draw bars per device ---
    for row, dev in enumerate(plot_devices):
        vec = np.asarray(vectors[dev], dtype=float)
        cat = categorize_device(dev)

        active = np.abs(vec) > activity_eps
        if not active.any():
            continue

        changes = np.where(np.diff(active.astype(int)) != 0)[0]
        seg_starts, seg_ends = [], []

        if active[0]:
            seg_starts.append(0)

        for c in changes:
            if not active[c] and active[c + 1]:
                seg_starts.append(c + 1)
            elif active[c] and not active[c + 1]:
                seg_ends.append(c + 1)

        if active[-1]:
            seg_ends.append(len(vec))

        assert len(seg_starts) == len(seg_ends)

        for s, e in zip(seg_starts, seg_ends):
            t_start = t_axis[s]
            t_end_seg = t_axis[e - 1] + dt
            width = t_end_seg - t_start
            if width <= 0:
                continue

            segment_values = vec[s:e]
            val = float(np.median(segment_values))

            if cat == "pump":
                color = pump_color(val)
            elif cat == "valve":
                color = VALVE_COLOR
            else:
                color = SENSOR_COLOR

            y_center = row
            height = 0.6
            rect = Rectangle(
                (t_start, y_center - height / 2),
                width,
                height,
                facecolor=color,
                edgecolor="k",
                linewidth=0.6,
                zorder=3,
            )
            ax.add_patch(rect)

            # numeric value on bar
            ax.text(
                t_start + width / 2.0,
                y_center,
                f"{val:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                zorder=4,
            )

    # --- Titles & sub-titles with boxes + leader lines ---

    # transform: x in data coords, y in axes coords
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # Subtitles (shifted one segment to the right, and zig-zag in height)
    n_segments = len(phase_edges) - 1
    y_base = 1.06
    y_amp = 0.03  # amplitude for -_-_ pattern

    for i in range(n_segments):
        label = SUB_LABELS[i]
        if not label:
            continue

        # Shift one subsegment to the right: use segment i+1
        j = i + 1
        if j >= n_segments:
            continue

        start2 = phase_edges[j]
        end2 = phase_edges[j + 1]
        x_mid = 0.5 * (start2 + end2)

        # pattern -_-_ : high / low / high / low ...
        # First subtitle low → flip pattern
        y_box = y_base + (-y_amp if i % 2 == 0 else +y_amp)

        # text box
        ax.text(
            x_mid,
            y_box,
            label,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white",
                      ec="black",
                      lw=0.6),
            zorder=5,
        )

        # leader line from top of axes to box
        ax.plot(
            [x_mid, x_mid],
            [1.0, y_box],
            transform=trans,
            color="black",
            linewidth=0.6,
            alpha=0.7,
            zorder=4,
        )

    # Major section titles
    y_title = 1.14
    for title, seg_start, seg_end in MAJOR_SECTIONS:
        seg_start = max(0, seg_start)
        seg_end = min(n_segments, seg_end)
        if seg_end <= seg_start:
            continue
        x_left = phase_edges[seg_start]
        x_right = phase_edges[seg_end]
        x_mid = 0.5 * (x_left + x_right)

        ax.text(
            x_mid,
            y_title,
            title,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            zorder=5,
        )

    # Dashed vertical lines at big section boundaries (each shifted one segment right)
    section_boundaries_segments = [2, 6, 10]
    for seg_idx in section_boundaries_segments:
        shifted = seg_idx + 1
        if 0 <= shifted < len(phase_edges):
            x = phase_edges[shifted]
            ax.axvline(
                x,
                color="k",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                zorder=2,
            )

    # --- Decorate axes ---
    ax.set_ylim(-1, n_dev - 0.0)
    ax.set_yticks(range(n_dev))
    pretty_labels = [pretty_label(name) for name in plot_devices]
    ax.set_yticklabels(pretty_labels, fontsize=14)

    ax.set_xlim(t_min, t_end)
    ax.set_xlabel("Time [s]", fontsize=14)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Move x-label slightly down to leave room for legend
    ax.xaxis.set_label_coords(0.5, -0.12)

    # --- Legend in the left gap under the axes (above axis title) ---
    pump_patch_pos = Patch(facecolor=pump_color(0.9), edgecolor="k", label="Pump (+PWM)")
    pump_patch_neg = Patch(facecolor=pump_color(-0.9), edgecolor="k", label="Pump (-PWM)")
    valve_patch = Patch(facecolor=VALVE_COLOR, edgecolor="k", label="Valve (on)")
    sensor_patch = Patch(facecolor=SENSOR_COLOR, edgecolor="k", label="Sensor (on)")

    fig.legend(
        handles=[pump_patch_pos, pump_patch_neg, valve_patch, sensor_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),   # much farther below
        frameon=True,
        fontsize=11,
        ncol=4
    )


    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved timing map to {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        npz_path = Path(sys.argv[1])
    else:
        npz_path = script_dir / "system_device_activity_vectors.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found.")

    t_axis, device_names, vectors = load_device_activity(npz_path)
    print(f"Loaded {len(device_names)} devices, {len(t_axis)} time steps.")

    out_path = npz_path.with_name("system_device_timing_map.png")
    plot_timing_map(t_axis, device_names, vectors, out_path)


if __name__ == "__main__":
    main()
