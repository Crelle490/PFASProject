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
    python tests/plot_device_timing_map.py
    python tests/plot_device_timing_map.py path/to/file.npz
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch


# ---------- Helpers for categorisation ----------

def categorize_device(name: str) -> str:
    """
    Return 'pump', 'valve', or 'sensor' based on name.

    We only treat exact valve names as valves; everything else that
    isn't a known sensor is considered a pump, so names like
    'pump_holding_to_valves' do NOT get misclassified as valves.
    """
    lname = name.lower()

    # explicit valve names
    if lname in {"valve1", "valve2"}:
        return "valve"

    # sensor naming convention
    if lname.startswith("sensor_") or lname in {"ph", "fluoride"}:
        return "sensor"

    # default: treat as pump
    return "pump"


def pretty_label(name: str) -> str:
    """
    Build a human-friendly label for the Y-axis.

    - Pumps get 'Pump ...'
        * pump_mix -> 'Pump reactor'
        * pump_holding_to_valves -> 'Pump holding to valves'
        * other pumps: 'Pump <name>'
    - Valves/sensors keep their original name.
    """
    cat = categorize_device(name)
    lname = name.lower()

    if cat == "pump":
        if lname == "pump_mix":
            return "Pump reactor"
        if lname == "pump_holding_to_valves":
            return "Pump holding to valves"
        return f"Pump {name}"
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
        # light blue -> dark blue
        light = np.array([0.776, 0.859, 0.937])   # ~ #c6dbef
        dark  = np.array([0.031, 0.318, 0.612])   # ~ #08519c
        rgb = light + (dark - light) * v
    else:
        # light red -> dark red
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
    Build a bar-style timing map, similar to your original system timing plot,
    but only using the per-device activity vectors.
    """
    n_dev = len(device_names)
    t_min, t_max = float(t_axis[0]), float(t_axis[-1])

    fig_height = 0.6 * n_dev + 2
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # --- Background phase bands based on *valve* state changes ---
    valve_names = [dev for dev in device_names if categorize_device(dev) == "valve"]

    t_bounds = None
    if valve_names and len(t_axis) > 1:
        # Build (T, n_valves) matrix of 0/1 valve states
        valve_states = []
        for name in valve_names:
            vec = np.asarray(vectors[name], dtype=float)
            active = (np.abs(vec) > activity_eps).astype(int)
            valve_states.append(active)
        valve_mat = np.vstack(valve_states).T  # shape (T, n_valves)

        # find indices where any valve changes state
        change_idx = np.where(np.any(np.diff(valve_mat, axis=0) != 0, axis=1))[0] + 1

        if change_idx.size > 0:
            idx_bounds = np.concatenate(([0], change_idx, [len(t_axis) - 1]))
            t_bounds = t_axis[idx_bounds]

    # Draw background spans if we have meaningful valve-based boundaries
    if t_bounds is not None and len(t_bounds) > 1:
        for i in range(len(t_bounds) - 1):
            start = float(t_bounds[i])
            end = float(t_bounds[i + 1])
            if end <= start:
                continue
            # alternate light blue / light pink
            if i % 2 == 0:
                color = (0.93, 0.96, 1.0)   # very light blue
            else:
                color = (1.0, 0.94, 0.95)   # very light pink
            ax.axvspan(start, end, color=color, alpha=0.5, zorder=0)

    # --- Draw bars per device ---
    # y coordinate: each device at integer row index
    for row, dev in enumerate(device_names):
        vec = np.asarray(vectors[dev], dtype=float)
        cat = categorize_device(dev)

        # find contiguous segments where |value| > activity_eps
        active = np.abs(vec) > activity_eps
        if not active.any():
            continue

        # indices where state changes (active<->inactive)
        changes = np.where(np.diff(active.astype(int)) != 0)[0]

        # segment start indices
        seg_starts = []
        seg_ends = []

        # first segment
        if active[0]:
            seg_starts.append(0)

        for c in changes:
            if not active[c] and active[c + 1]:
                seg_starts.append(c + 1)
            elif active[c] and not active[c + 1]:
                seg_ends.append(c + 1)

        # if still active at end
        if active[-1]:
            seg_ends.append(len(vec))

        assert len(seg_starts) == len(seg_ends)

        # draw each active segment
        for s, e in zip(seg_starts, seg_ends):
            t_start = t_axis[s]
            t_end = t_axis[e - 1] + (t_axis[1] - t_axis[0])  # assume uniform step
            width = t_end - t_start
            if width <= 0:
                continue

            segment_values = vec[s:e]
            val = float(np.median(segment_values))

            if cat == "pump":
                color = pump_color(val)
            elif cat == "valve":
                color = VALVE_COLOR
            else:  # sensor
                color = SENSOR_COLOR

            # bar centered on row index
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

    # --- Decorate axes ---
    ax.set_ylim(-1, n_dev - 0.0)
    ax.set_yticks(range(n_dev))

    # pretty labels with "Pump ..." etc.
    pretty_labels = [pretty_label(name) for name in device_names]
    ax.set_yticklabels(pretty_labels, fontsize=14)

    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Time [s]", fontsize=14)

    # bigger tick labels
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # --- Legend (bigger font) ---
    pump_patch_pos = Patch(facecolor=pump_color(0.9), edgecolor="k", label="Pump (+PWM)")
    pump_patch_neg = Patch(facecolor=pump_color(-0.9), edgecolor="k", label="Pump (-PWM)")
    valve_patch = Patch(facecolor=VALVE_COLOR, edgecolor="k", label="Valve (on)")
    sensor_patch = Patch(facecolor=SENSOR_COLOR, edgecolor="k", label="Sensor (on)")
    ax.legend(
        handles=[pump_patch_pos, pump_patch_neg, valve_patch, sensor_patch],
        loc="upper right",
        frameon=True,
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved timing map to {out_path}")


def main():
    if len(sys.argv) > 1:
        npz_path = Path(sys.argv[1])
    else:
        npz_path = Path("system_device_activity_vectors.npz")

    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found. Run run_controller_init.py first, or pass a path.")

    t_axis, device_names, vectors = load_device_activity(npz_path)
    print(f"Loaded {len(device_names)} devices, {len(t_axis)} time steps.")

    out_path = npz_path.with_name("system_device_timing_map.png")
    plot_timing_map(t_axis, device_names, vectors, out_path)


if __name__ == "__main__":
    main()
