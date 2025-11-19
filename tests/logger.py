import time
import matplotlib.pyplot as plt


# ================================================================
# Timeline Logger
# ================================================================
class TimelineLogger:
    def __init__(self):
        self.t0 = time.monotonic()
        # (t_rel, channel, value, volume_ml)
        self.events = []

        # High-level events (functions)
        self._event_active = {}    # name -> start_time
        self.event_intervals = []  # (name, t_start, t_end)

        # Cycle markers
        self.cycle_marks = []      # (t_rel, label_str)

    # --------- logging pumps/valves/sensors ----------
    def log(self, channel: str, value: float, volume_ml: float = None):
        """
        Pumps:
            value      = PWM in [0,100]
            volume_ml  = volume pushed during this instant (optional)

        Valves/sensors:
            value      = 0/1
            volume_ml  ignored
        """
        t_rel = time.monotonic() - self.t0
        self.events.append((t_rel, channel, float(value), volume_ml))

    # --------- high-level function events ----------
    def start_event(self, name: str):
        t_rel = time.monotonic() - self.t0
        self._event_active[name] = t_rel

    def end_event(self, name: str):
        t_rel = time.monotonic() - self.t0
        t0 = self._event_active.pop(name, None)
        if t0 is not None:
            self.event_intervals.append((name, t0, t_rel))

    # --------- cycle division ----------
    def mark_cycle(self, cycle_idx: int | str):
        """
        Mark the start of a new cycle for plotting.
        Example: mark_cycle(1), mark_cycle(2), ...
        """
        t_rel = time.monotonic() - self.t0
        self.cycle_marks.append((t_rel, str(cycle_idx)))

    # --------- convert logs to intervals ----------
    def to_component_intervals(self):
        """
        Returns:
            dict[channel] = list[
                    (start, duration, value, volume_ml)
            ]
        """
        intervals = {}
        grouped = {}

        # Sort by time
        for t, ch, v, vol in sorted(self.events, key=lambda e: e[0]):
            grouped.setdefault(ch, []).append((t, v, vol))

        for ch, seq in grouped.items():
            intervals[ch] = []
            active_val = 0.0
            active_vol = None
            start_t = None

            for t, v, vol in seq:
                if active_val == 0.0 and v != 0.0:
                    # OFF -> ON
                    active_val = v
                    active_vol = vol
                    start_t = t

                elif active_val != 0.0 and v != active_val:
                    # value changed
                    intervals[ch].append((start_t, t - start_t, active_val, active_vol))
                    if v == 0.0:
                        active_val = 0.0
                        active_vol = None
                        start_t = None
                    else:
                        active_val = v
                        active_vol = vol
                        start_t = t

            # still active at end
            if active_val != 0.0 and start_t is not None:
                last_t = seq[-1][0]
                intervals[ch].append((start_t, last_t - start_t, active_val, active_vol))

        return intervals


# ================================================================
# Plot timeline
# ================================================================
def plot_timeline(logger: TimelineLogger, title="System Timing",
                  filename=None, show=True):
    comp_intervals = logger.to_component_intervals()
    channels = list(comp_intervals.keys())

    # --- nice ordering: sensors, valves, pumps ---
    preferred_order = [
        "sensor_fluoride", "sensor_ph",
        "valve_1", "valve_2",
        "pump_1", "pump_2", "pump_3",
        "pump_4", "pump_5", "pump_6", "pump_7",
    ]
    ordered_channels = [ch for ch in preferred_order if ch in channels]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.82, left=0.15, right=0.98, bottom=0.12)

    ymax = len(ordered_channels) - 0.5
    height = 0.6

    # ---------- shaded function/event bands ----------
    if logger.event_intervals:
        colors = ["#fff3f3", "#eef5ff"]
        short_name = {
            "init_reservoirs": "init",
            "create_mixture": "mix",
            "reactor_circulation": "react",
            "sensor_sample": "sample",
            "sensor_flush": "flush",
            "catalyst_dose": "dose",
        }
        for i, (name, t0, t1) in enumerate(logger.event_intervals):
            ax.axvspan(t0, t1,
                       color=colors[i % len(colors)], alpha=0.5, zorder=0)
            label_y = ymax + 0.3 + 0.25 * (i % 2)
            xmid = 0.5 * (t0 + t1)
            ax.text(xmid, label_y, short_name.get(name, name),
                    ha="center", va="bottom", fontsize=8)

    # ---------- cycle division lines ----------
    for t, label in logger.cycle_marks:
        ax.axvline(t, color="black", linestyle="-", linewidth=1.0,
                   alpha=0.9, zorder=3)
        ax.text(t, ymax + 0.9, f"Cycle {label}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ---------- helper: style by component type ----------
    def style_for_channel(ch: str):
        if ch.startswith("pump_"):
            # color by PWM (blue colormap)
            def color_for_val(v):
                norm = max(0.0, min(v, 100.0)) / 100.0
                return plt.cm.Blues(norm if norm > 0 else 0.2)
            return {"color_func": color_for_val,
                    "hatch": None,
                    "edgecolor": "black"}
        elif ch.startswith("valve_"):
            # valves: orange + hatch
            def color_for_val(_v):
                return "tab:orange"
            return {"color_func": color_for_val,
                    "hatch": "//",
                    "edgecolor": "black"}
        elif ch.startswith("sensor_"):
            # sensors: green, no hatch
            def color_for_val(_v):
                return "tab:green"
            return {"color_func": color_for_val,
                    "hatch": None,
                    "edgecolor": "black"}
        else:
            # fallback
            def color_for_val(_v):
                return "tab:gray"
            return {"color_func": color_for_val,
                    "hatch": None,
                    "edgecolor": "black"}

    # ---------- component bars ----------
    yticks = []
    ylabels = []

    for i, ch in enumerate(ordered_channels):
        yticks.append(i)
        ylabels.append(ch)

        style = style_for_channel(ch)
        for start, dur, val, vol in comp_intervals.get(ch, []):
            color = style["color_func"](val)

            ax.broken_barh(
                [(start, dur)],
                (i - height/2, height),
                facecolors=color,
                edgecolors=style["edgecolor"],
                linewidth=0.3,
                hatch=style["hatch"],
                zorder=2,
            )

            # volume text only for pumps, and only if interval is not tiny
            if ch.startswith("pump_") and vol is not None and dur > 0.03:
                xmid = start + dur / 2
                ax.text(
                    xmid, i,
                    f"{vol:.1f} mL",
                    ha="center", va="center",
                    fontsize=7, color="black",
                )

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time [s]")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax.set_ylim(-1, len(ordered_channels) + 1.5)

    # ---------- simple legend ----------
    import matplotlib.patches as mpatches
    pump_patch = mpatches.Patch(color=plt.cm.Blues(0.7), label="Pump (PWM-colored)")
    valve_patch = mpatches.Patch(facecolor="tab:orange", hatch="//",
                                 edgecolor="black", label="Valve")
    sensor_patch = mpatches.Patch(color="tab:green", label="Sensor")
    ax.legend(handles=[pump_patch, valve_patch, sensor_patch],
              loc="upper right", fontsize=8, framealpha=0.9)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()




# ================================================================
# Fake cycle test
# ================================================================
def simulate_fake_cycle(logger: TimelineLogger):
    def wait(t): time.sleep(t)

    # ----- Init reservoirs -----
    logger.start_event("init_reservoirs")
    logger.log("pump_2", 60, volume_ml=4.0)
    logger.log("pump_3", 60, volume_ml=3.2)
    logger.log("pump_7", 40, volume_ml=2.1)
    wait(0.15)
    logger.log("pump_2", 0)
    logger.log("pump_3", 0)
    logger.log("pump_7", 0)
    logger.end_event("init_reservoirs")

    # ----- Create mixture -----
    logger.start_event("create_mixture")
    logger.log("pump_4", 70, volume_ml=5.5)
    logger.log("pump_5", 80, volume_ml=12.0)
    wait(0.2)
    logger.log("pump_4", 0)
    logger.log("pump_5", 0)

    logger.log("pump_1", 90, volume_ml=10.0)
    wait(0.25)
    logger.log("pump_1", 0)
    logger.end_event("create_mixture")

    # ----- Reactor circulation -----
    logger.start_event("reactor_circulation")
    logger.log("pump_1", 75, volume_ml=20.0)
    logger.log("valve_1", 1)
    logger.log("valve_2", 1)
    wait(0.3)
    logger.log("pump_1", 0)
    logger.log("valve_1", 0)
    logger.log("valve_2", 0)
    logger.end_event("reactor_circulation")

    # ----- Sensor sample -----
    logger.start_event("sensor_sample")
    logger.log("pump_6", 50, volume_ml=2.0)
    logger.log("valve_1", 1)
    logger.log("sensor_fluoride", 1)
    logger.log("sensor_ph", 1)
    wait(0.15)
    logger.log("sensor_fluoride", 0)
    logger.log("sensor_ph", 0)
    logger.log("pump_6", 0)
    logger.log("valve_1", 0)
    logger.end_event("sensor_sample")

    # ----- Sensor flush -----
    logger.start_event("sensor_flush")
    logger.log("pump_6", 40, volume_ml=1.0)
    wait(0.12)
    logger.log("pump_6", 0)
    logger.end_event("sensor_flush")

    # ----- Catalyst dosing -----
    logger.start_event("catalyst_dose")
    logger.log("pump_2", 30, volume_ml=0.8)
    wait(0.08)
    logger.log("pump_2", 0)

    logger.log("pump_3", 50, volume_ml=1.1)
    wait(0.08)
    logger.log("pump_3", 0)

    logger.log("pump_7", 70, volume_ml=1.9)
    wait(0.08)
    logger.log("pump_7", 0)
    logger.end_event("catalyst_dose")


if __name__ == "__main__":
    logger = TimelineLogger()

    # mark cycle boundaries
    logger.mark_cycle(1)
    simulate_fake_cycle(logger)

    logger.mark_cycle(2)
    simulate_fake_cycle(logger)

    plot_timeline(logger, filename="demo_timeline.png", show=False)
    print("Saved demo_timeline.png")
