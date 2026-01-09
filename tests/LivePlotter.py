from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
import time

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotConfig:
    title: str = "PFAS MPC Live"
    max_points: int = 600          # rolling window length
    refresh_hz: float = 10.0       # UI refresh rate (avoid redrawing every call)
    show_pred: bool = True
    show_states: bool = True

    # Labels
    meas_label: str = "F (ppm)"
    meas2_label: str = "F (M)"     # optional second measurement
    u_labels: tuple[str, str] = ("u_SO3 (M)", "u_Cl (M)")

    # If you want to plot specific state indices (e.g., PFAS, F-, etc.)
    state_indices: Optional[Sequence[int]] = None
    state_labels: Optional[Sequence[str]] = None


class LivePlotter:
    """
    Create once; call update() each control loop.

    update(
        t=timestamp_or_seconds,
        y=F_ppm,
        y2=F_M,
        u=[u_so3, u_cl],
        x=state_vector (optional),
        pred=dict or array (optional)
    )

    pred can be:
      - dict with keys: "t", "y", "x", "u" (any subset), or
      - array-like for y-horizon (length N)
    """

    def __init__(self, cfg: PlotConfig = PlotConfig()):
        self.cfg = cfg
        self._last_draw_t = 0.0

        # buffers (rolling)
        self.t: list[float] = []
        self.y: list[float] = []
        self.y2: list[float] = []
        self.u1: list[float] = []
        self.u2: list[float] = []
        self.x_hist: list[np.ndarray] = []  # store full x vectors (optional)

        # matplotlib objects
        self._fig = None
        self._ax_y = None
        self._ax_u = None
        self._ax_x = None

        self._line_y = None
        self._line_y2 = None
        self._line_u1 = None
        self._line_u2 = None
        self._lines_x = []  # multiple state traces

        self._pred_line_y = None
        self._pred_line_y2 = None

        self._init_plot()

    # ----------------- public -----------------

    def update(
        self,
        *,
        t: Optional[float] = None,
        y: Optional[float] = None,
        y2: Optional[float] = None,
        u: Optional[Sequence[float]] = None,
        x: Optional[Sequence[float]] = None,
        pred: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add one sample and refresh the plot at cfg.refresh_hz.
        """
        if t is None:
            t = time.time()
        t = float(t)

        if u is None:
            u = (np.nan, np.nan)
        u = np.asarray(u, dtype=float).ravel()
        if u.size < 2:
            u = np.pad(u, (0, 2 - u.size), constant_values=np.nan)

        # append data
        self.t.append(t)
        self.y.append(np.nan if y is None else float(y))
        self.y2.append(np.nan if y2 is None else float(y2))
        self.u1.append(float(u[0]))
        self.u2.append(float(u[1]))

        if x is not None:
            self.x_hist.append(np.asarray(x, dtype=float).ravel())
        else:
            self.x_hist.append(np.array([], dtype=float))

        # roll window
        self._trim_buffers()

        # update lines (cheap) + maybe redraw (throttled)
        self._update_lines(pred=pred, meta=meta)
        self._maybe_draw()

    def close(self):
        try:
            plt.close(self._fig)
        except Exception:
            pass

    # ----------------- internals -----------------

    def _init_plot(self):
        plt.ion()
        show_states = bool(self.cfg.show_states)

        if show_states:
            self._fig, (self._ax_y, self._ax_u, self._ax_x) = plt.subplots(3, 1, sharex=True)
        else:
            self._fig, (self._ax_y, self._ax_u) = plt.subplots(2, 1, sharex=True)
            self._ax_x = None

        self._fig.suptitle(self.cfg.title)

        # measurement axis
        (self._line_y,) = self._ax_y.plot([], [], label=self.cfg.meas_label)
        (self._line_y2,) = self._ax_y.plot([], [], label=self.cfg.meas2_label)
        self._ax_y.set_ylabel("Measurement")
        self._ax_y.grid(True)
        self._ax_y.legend(loc="upper right")

        # prediction overlay lines (optional)
        (self._pred_line_y,) = self._ax_y.plot([], [], linestyle="--", label="pred-y")
        (self._pred_line_y2,) = self._ax_y.plot([], [], linestyle="--", label="pred-y2")
        self._pred_line_y.set_visible(False)
        self._pred_line_y2.set_visible(False)

        # input axis
        (self._line_u1,) = self._ax_u.plot([], [], label=self.cfg.u_labels[0])
        (self._line_u2,) = self._ax_u.plot([], [], label=self.cfg.u_labels[1])
        self._ax_u.set_ylabel("Inputs")
        self._ax_u.grid(True)
        self._ax_u.legend(loc="upper right")

        # state axis (optional)
        if self._ax_x is not None:
            self._ax_x.set_ylabel("States")
            self._ax_x.set_xlabel("time (s)")
            self._ax_x.grid(True)

        plt.show(block=False)
        plt.pause(0.001)

    def _trim_buffers(self):
        n = len(self.t)
        if n <= self.cfg.max_points:
            return
        k0 = n - self.cfg.max_points

        self.t = self.t[k0:]
        self.y = self.y[k0:]
        self.y2 = self.y2[k0:]
        self.u1 = self.u1[k0:]
        self.u2 = self.u2[k0:]
        self.x_hist = self.x_hist[k0:]

    def _update_lines(self, *, pred: Optional[Any], meta: Optional[Dict[str, Any]]):
        # convert time to relative seconds for nicer axis
        t0 = self.t[0] if self.t else time.time()
        tt = np.asarray(self.t, dtype=float) - float(t0)

        self._line_y.set_data(tt, np.asarray(self.y, dtype=float))
        self._line_y2.set_data(tt, np.asarray(self.y2, dtype=float))
        self._line_u1.set_data(tt, np.asarray(self.u1, dtype=float))
        self._line_u2.set_data(tt, np.asarray(self.u2, dtype=float))

        # states
        if self._ax_x is not None:
            self._update_state_lines(tt)

        # prediction overlay
        self._update_pred_lines(tt, pred)

        # rescale axes
        self._autoscale(self._ax_y)
        self._autoscale(self._ax_u)
        if self._ax_x is not None:
            self._autoscale(self._ax_x)

        # small text annotation (optional)
        if meta and self._ax_y is not None:
            # put meta in title of measurement plot
            txt = " | ".join([f"{k}:{v}" for k, v in meta.items()])
            self._ax_y.set_title(txt)

    def _update_state_lines(self, tt: np.ndarray):
        # Determine which x to plot
        xs = [x for x in self.x_hist if x.size > 0]
        if not xs:
            # no state data
            for ln in self._lines_x:
                ln.set_data([], [])
            return

        # choose indices
        x_dim = int(xs[-1].size)
        if self.cfg.state_indices is None:
            idx = list(range(min(3, x_dim)))  # default plot first 3 states
            labels = [f"x[{i}]" for i in idx]
        else:
            idx = [int(i) for i in self.cfg.state_indices if 0 <= int(i) < x_dim]
            labels = list(self.cfg.state_labels) if self.cfg.state_labels else [f"x[{i}]" for i in idx]

        # ensure enough lines exist
        while len(self._lines_x) < len(idx):
            (ln,) = self._ax_x.plot([], [], label=f"x[{len(self._lines_x)}]")
            self._lines_x.append(ln)
            self._ax_x.legend(loc="upper right")

        # build matrix with NaN for missing
        X = np.full((len(self.x_hist), x_dim), np.nan, dtype=float)
        for k, x in enumerate(self.x_hist):
            if x.size == x_dim:
                X[k, :] = x
            elif x.size > 0:
                # mismatched dimension; best-effort fill
                m = min(x.size, x_dim)
                X[k, :m] = x[:m]

        for j, i in enumerate(idx):
            self._lines_x[j].set_label(labels[j] if j < len(labels) else f"x[{i}]")
            self._lines_x[j].set_data(tt, X[:, i])

        # hide extra lines
        for j in range(len(idx), len(self._lines_x)):
            self._lines_x[j].set_data([], [])

        # refresh legend labels
        self._ax_x.legend(loc="upper right")

    def _update_pred_lines(self, tt: np.ndarray, pred: Optional[Any]):
        if not self.cfg.show_pred:
            self._pred_line_y.set_visible(False)
            self._pred_line_y2.set_visible(False)
            return

        if pred is None:
            self._pred_line_y.set_visible(False)
            self._pred_line_y2.set_visible(False)
            return

        # supported:
        # - dict: {"t": [...], "y": [...], "y2": [...]}
        # - array-like: y horizon
        if isinstance(pred, dict):
            t_pred = pred.get("t", None)
            y_pred = pred.get("y", None)
            y2_pred = pred.get("y2", None)

            if t_pred is None:
                # assume horizon relative to last time
                kN = len(y_pred) if y_pred is not None else 0
                t_last = tt[-1] if tt.size else 0.0
                t_pred = t_last + np.arange(kN)

            t_pred = np.asarray(t_pred, dtype=float)
            if y_pred is not None:
                self._pred_line_y.set_data(t_pred, np.asarray(y_pred, dtype=float))
                self._pred_line_y.set_visible(True)
            else:
                self._pred_line_y.set_visible(False)

            if y2_pred is not None:
                self._pred_line_y2.set_data(t_pred, np.asarray(y2_pred, dtype=float))
                self._pred_line_y2.set_visible(True)
            else:
                self._pred_line_y2.set_visible(False)

            return

        # assume array-like y horizon
        y_pred = np.asarray(pred, dtype=float).ravel()
        if y_pred.size == 0:
            self._pred_line_y.set_visible(False)
            self._pred_line_y2.set_visible(False)
            return

        t_last = tt[-1] if tt.size else 0.0
        t_pred = t_last + np.arange(y_pred.size)
        self._pred_line_y.set_data(t_pred, y_pred)
        self._pred_line_y.set_visible(True)
        self._pred_line_y2.set_visible(False)

    def _autoscale(self, ax):
        ax.relim()
        ax.autoscale_view(True, True, True)

    def _maybe_draw(self):
        now = time.time()
        min_dt = 1.0 / max(1e-3, float(self.cfg.refresh_hz))
        if (now - self._last_draw_t) < min_dt:
            return
        self._last_draw_t = now
        self._fig.canvas.draw_idle()
        plt.pause(0.001)
