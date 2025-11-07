# live_plotter.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import advance_one_control_step

# ----------------------------------------------------------
# 1) --- Prediction rollout helpers ------------------------
# ----------------------------------------------------------

def predict_horizon_old(ctx, rk_cell, xk_flat, Uplan, substeps, Ts):
    """
    Legacy numeric rollout using advance_one_control_step.
    """
    x = np.asarray(xk_flat, dtype=float).copy().reshape(1, 1, -1)
    Xpred = [x.copy()]
    N = int(ctx["N"])
    for uk in Uplan:
        x = advance_one_control_step(rk_cell, x, uk, int(substeps))
        Xpred.append(x.copy())
    Xpred = np.array(Xpred, dtype=float).reshape(N + 1, -1)
    Z_pred = np.sum(Xpred[:, :7], axis=1)
    t_pred = np.arange(N + 1, dtype=float) * float(Ts)
    t_u_pred = np.arange(N, dtype=float) * float(Ts)
    return t_pred, Z_pred, t_u_pred, Xpred


# ----------------------------------------------------------
# 2) --- Live plotter class --------------------------------
# ----------------------------------------------------------

class LiveMPCPlot:
    """
    Interactive visualizer for MPC:
    (1) ΣPFAS (measured + predicted)
    (2) Inputs (past + plan) — rendered as ZERO-ORDER HOLD (steps-post)
    (3) OPTIONAL: a 4×2 panel with each state (hist + pred)
    (4) OPTIONAL: scatter of measured fluoride (F⁻) on the F⁻ panel
    """

    def __init__(self, Ts, t_max, z0, u_max, x0=None, make_state_grid=True):
        self.Ts = float(Ts)
        self.t_max = float(t_max)
        self.z0 = float(z0)
        self.u_max = np.asarray(u_max, dtype=float).reshape(-1)
        self.x0 = None if x0 is None else np.asarray(x0, dtype=float).reshape(-1)

        # --- Matplotlib setup ---
        plt.ion()

        # ========== Main figure: ΣPFAS + inputs ==========
        self.fig, (self.axZ, self.axU) = plt.subplots(2, 1, figsize=(10, 8))

        # ΣPFAS
        (self.line_Z_hist,) = self.axZ.plot([], [], lw=2, label="Σ PFAS (measured)")
        (self.line_Z_pred,) = self.axZ.plot([], [], "--", lw=2, label="Σ PFAS (predicted)")
        self.axZ.set_ylabel("Σ PFAS [M]")
        self.axZ.set_title("MPC – Total PFAS and prediction horizon")
        self.axZ.grid(True)
        self.axZ.legend(loc="best")
        self.axZ.set_xlim(0.0, self.t_max)
        z_ylim_max = self.z0 * 1.05 if self.z0 > 0 else 1.0
        self.axZ.set_ylim(0.0, z_ylim_max)

        # Inputs (ZOH via steps-post)
        (self.line_U_so3_hist,) = self.axU.plot([], [], lw=2, label="SO₃ (hist)", drawstyle="steps-post")
        (self.line_U_cl_hist,)  = self.axU.plot([], [], lw=2, label="Cl⁻ (hist)", drawstyle="steps-post")
        (self.line_U_so3_pred,) = self.axU.plot([], [], "--", lw=2, label="SO₃ (plan)", drawstyle="steps-post")
        (self.line_U_cl_pred,)  = self.axU.plot([], [], "--", lw=2, label="Cl⁻ (plan)", drawstyle="steps-post")
        self.axU.set_ylabel("Dose [M]")
        self.axU.set_xlabel("Time [s]")
        self.axU.grid(True)
        self.axU.legend(loc="best")
        self.axU.set_xlim(0.0, self.t_max)
        u_ylim_max = float(np.max(self.u_max)) * 1.05 if self.u_max.size and np.max(self.u_max) > 0 else 1.0
        self.axU.set_ylim(0.0, u_ylim_max)
        if self.u_max.size >= 1:
            self.axU.axhline(self.u_max[0], linestyle=":", linewidth=1)
        if self.u_max.size >= 2:
            self.axU.axhline(self.u_max[1], linestyle="--", linewidth=1)
        self.fig.tight_layout()

        # ========== Optional 4×2 state grid ==========
        self.make_state_grid = bool(make_state_grid)
        self.state_labels = [f"PFAS{i+1}" for i in range(7)] + ["F⁻"]
        self.state_lines_hist = []
        self.state_lines_pred = []
        self.ax_states = None
        self.scatter_F = None  # fluoride scatter artist

        if self.make_state_grid:
            self.figX, self.ax_states = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
            self.ax_states = self.ax_states.flatten()
            for i in range(8):
                (lh,) = self.ax_states[i].plot([], [], lw=1.5, alpha=0.95, label=f"{self.state_labels[i]} (hist)")
                (lp,) = self.ax_states[i].plot([], [], "--", lw=1.5, alpha=0.95, label=f"{self.state_labels[i]} (pred)")
                self.ax_states[i].set_title(self.state_labels[i])
                self.ax_states[i].set_ylabel("Concentration [M]")
                self.ax_states[i].grid(True)
                self.state_lines_hist.append(lh)
                self.state_lines_pred.append(lp)
            # add fluoride scatter on the last panel (F⁻)
            self.scatter_F = self.ax_states[7].scatter([], [], s=18, label="F⁻ (meas)", zorder=3)
            self.ax_states[7].legend(loc="best")
            for ax in self.ax_states[-2:]:
                ax.set_xlabel("Time [s]")
            self.figX.tight_layout()
            self._sync_state_ylim_from_total()

    # ---------- helpers ----------
    def _sync_state_ylim_from_total(self):
        if self.ax_states is None:
            return
        ymin, ymax = self.axZ.get_ylim()
        for ax in self.ax_states[:-1]:
            ax.set_ylim(ymin, ymax)

    # ---------- public API ----------
    def update(
        self,
        t_hist,
        Z_hist,
        t_u_hist,
        U_hist,
        t0_abs,
        t_pred_rel,
        Z_pred,
        t_u_pred_rel,
        U_plan,
        X_hist=None,
        X_pred=None,
        F_meas_t=None,   # <-- NEW: times of measured fluoride
        F_meas=None,     # <-- NEW: measured fluoride values
    ):
        """Update all plots. Inputs are TRUE ZOH with continuous hand-off.

        If provided, (F_meas_t, F_meas) will be scattered on the F⁻ panel.
        """
        # ----- ΣPFAS -----
        self.line_Z_hist.set_data(t_hist, Z_hist)
        t_pred_abs = t0_abs + np.asarray(t_pred_rel, dtype=float)
        self.line_Z_pred.set_data(t_pred_abs, Z_pred)

        # auto-raise ΣPFAS y-limit if needed
        z_max_now = max(
            float(np.max(Z_hist)) if len(Z_hist) else 0.0,
            float(np.max(Z_pred)) if len(Z_pred) else 0.0,
        )
        ymin, ymax = self.axZ.get_ylim()
        if z_max_now > 0.95 * ymax:
            self.axZ.set_ylim(0.0, 1.1 * z_max_now)

        # ----- Inputs (history) — ZOH with no gap -----
        U_hist = np.asarray(U_hist, dtype=float)
        if U_hist.size > 0:
            t_u_hist = np.asarray(t_u_hist, dtype=float)
            # Extend by one step so steps-post holds through [t_last, t_last+Ts)
            t_hist_ext = np.concatenate([t_u_hist, [t_u_hist[-1] + self.Ts]])
            so3_ext    = np.concatenate([U_hist[:, 0], [U_hist[-1, 0]]])
            cl_ext     = np.concatenate([U_hist[:, 1], [U_hist[-1, 1]]])

            self.line_U_so3_hist.set_data(t_hist_ext, so3_ext)
            self.line_U_cl_hist.set_data(t_hist_ext, cl_ext)
        else:
            self.line_U_so3_hist.set_data([], [])
            self.line_U_cl_hist.set_data([], [])

        # ----- Inputs (planned) — ZOH, start exactly at last history value -----
        U_plan = np.asarray(U_plan, dtype=float)
        if U_plan.size > 0:
            # Hand-off time = right after last applied action (ZOH boundary)
            t0_pred_abs = (t_u_hist[-1] + self.Ts) if len(t_u_hist) else t0_abs
            t_plan_abs  = t0_pred_abs + np.asarray(t_u_pred_rel, dtype=float)

            # Last historical value; if none, start from first planned
            if len(t_u_hist) and U_hist.size:
                u_last_so3 = float(U_hist[-1, 0])
                u_last_cl  = float(U_hist[-1, 1])
            else:
                u_last_so3 = float(U_plan[0, 0])
                u_last_cl  = float(U_plan[0, 1])

            # Prepend a point at hand-off time with the last historical value
            t_plan_abs = np.concatenate([[t0_pred_abs], t_plan_abs])
            so3_plan   = np.concatenate([[u_last_so3], U_plan[:, 0]])
            cl_plan    = np.concatenate([[u_last_cl],  U_plan[:, 1]])

            self.line_U_so3_pred.set_data(t_plan_abs, so3_plan)
            self.line_U_cl_pred.set_data(t_plan_abs, cl_plan)
        else:
            self.line_U_so3_pred.set_data([], [])
            self.line_U_cl_pred.set_data([], [])

        # ----- Optional per-state panels -----
        if self.ax_states is not None and (X_hist is not None) and (X_pred is not None):
            X_hist = np.asarray(X_hist, dtype=float)
            X_pred = np.asarray(X_pred, dtype=float)
            for i in range(8):
                self.state_lines_hist[i].set_data(t_hist, X_hist[:, i])
                self.state_lines_pred[i].set_data(t_pred_abs, X_pred[:, i])

            # sync PFAS1–7 to ΣPFAS y-lims
            self._sync_state_ylim_from_total()

            # fluoride axis scaling (include measured points if provided)
            f_hist = X_hist[:, 7] if X_hist.shape[1] >= 8 else np.array([])
            f_pred = X_pred[:, 7] if X_pred.shape[1] >= 8 else np.array([])
            f_all_parts = []
            if f_hist.size: f_all_parts.append(f_hist)
            if f_pred.size: f_all_parts.append(f_pred)
            if (F_meas_t is not None) and (F_meas is not None) and len(F_meas):
                f_all_parts.append(np.asarray(F_meas, dtype=float))

                # update scatter
                try:
                    offs = np.column_stack([np.asarray(F_meas_t, float), np.asarray(F_meas, float)])
                    self.scatter_F.set_offsets(offs)
                except Exception:
                    # fall back to re-creating the scatter if needed
                    self.scatter_F.remove()
                    self.scatter_F = self.ax_states[7].scatter(np.asarray(F_meas_t, float),
                                                              np.asarray(F_meas, float),
                                                              s=18, label="F⁻ (meas)", zorder=3)
                    self.ax_states[7].legend(loc="best")
            else:
                # nothing measured yet -> clear
                self.scatter_F.set_offsets(np.empty((0, 2)))

            if f_all_parts:
                f_all = np.concatenate(f_all_parts)
                f_min, f_max = float(np.min(f_all)), float(np.max(f_all))
                if f_max == f_min:
                    pad = max(abs(f_max) * 0.1, 1e-15)
                    self.ax_states[7].set_ylim(f_max - pad, f_max + pad)
                else:
                    f_margin = 0.1 * (f_max - f_min)
                    self.ax_states[7].set_ylim(f_min - f_margin, f_max + f_margin)

        # keep x-lims fixed
        self.axZ.set_xlim(0.0, self.t_max)
        self.axU.set_xlim(0.0, self.t_max)
        if self.ax_states is not None:
            for ax in self.ax_states:
                ax.set_xlim(0.0, self.t_max)

        plt.pause(0.5)

""" LIVEPLotter without CasADi dependency 
# live_plotter.py
# ----------------------------------------------------------
# Live plotting utilities for PFAS MPC simulations (no CasADi dependency).
#
# Prediction horizon uses (in order of preference):
#   1) advance_one_control_step(rk_cell, xk, uk, substeps) if ctx_adi["rk_cell"] present
#   2) ctx_adi["step_fn"](x,u) -> x_next   (pure Python callable you provide)
#   3) ctx_adi["Phi"](x,u) -> x_next       (pure Python callable, NOT CasADi)
#
# Plotting:
#   - ΣPFAS history + prediction
#   - Inputs (hist + planned)
#   - Optional 4×2 state grid: PFAS1–7 share ΣPFAS y-scale; F⁻ auto-scales
#
# Author: Nichlas Kondrup (2025) + patch by ChatGPT
# ----------------------------------------------------------

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from helper_functions import advance_one_control_step  # your plant stepper


# ----------------------------------------------------------
# 1) --- Prediction rollout helper (numeric only) ----------
# ----------------------------------------------------------

import numpy as np
from helper_functions import advance_one_control_step
# live_plotter.py
import numpy as np
from helper_functions import advance_one_control_step

def predict_horizon(ctx_adi, xk_flat, Uplan, Ts, uscale):

    N        = int(ctx_adi["N"])
    rk_cell  = ctx_adi["rk_cell"]
    substeps = int(ctx_adi.get("substeps", 1))
    NX       = 8  # adjust if different

    # inputs / scaling
    x0 = np.asarray(xk_flat, dtype=float).reshape(-1)          # (NX,)
    Uplan  = np.asarray(Uplan,  dtype=float).reshape(N, -1)    # (N,2)
    uscale = np.asarray(uscale, dtype=float).reshape(1, -1)    # (1,2)
    U_phys = np.maximum(Uplan * uscale, 0.0)                   # clamp ≥ 0

    # helper: force 2-D (1, NX)
    def as_2d(x):
        a = np.asarray(x, dtype=float)
        return a.reshape(1, -1) if a.ndim == 1 else a

    states = as_2d(x0)  # (1, NX)
    X = [x0.copy()]

    for uk in U_phys:
        so3, cl = float(uk[0]), float(uk[1])

        # Expand substeps manually: call helper with substeps=1 repeatedly.
        # This way we can fix shape between calls without editing the helper.
        for _ in range(substeps):
            states = as_2d(states)  # ensure (1, NX) before each call
            states = advance_one_control_step(rk_cell, states, [so3, cl], 1)
            states = as_2d(states)  # normalize output to (1, NX)

        x_next = np.asarray(states, float).reshape(-1)  # (NX,)
        if x_next.size > NX:
            x_next = x_next[-NX:]  # safety if the cell returns extra entries
        X.append(x_next.copy())

    Xpred  = np.stack(X, axis=0)                  # (N+1, NX)
    Z_pred = np.sum(Xpred[:, :7], axis=1)         # ΣPFAS
    t_pred   = np.arange(N + 1, dtype=float) * float(Ts)
    t_u_pred = np.arange(N,     dtype=float) * float(Ts)
    return t_pred, Z_pred, t_u_pred, Xpred

# ----------------------------------------------------------
# 2) --- Live plotter class --------------------------------
# ----------------------------------------------------------

class LiveMPCPlot:

    def __init__(self, Ts, t_max, z0, u_max, x0=None, make_state_grid=True):
        self.Ts    = float(Ts)
        self.t_max = float(t_max)
        self.z0    = float(z0)
        self.u_max = np.asarray(u_max, dtype=float).reshape(-1)
        self.x0    = None if x0 is None else np.asarray(x0, dtype=float).reshape(-1)

        plt.ion()

        # ---------- ΣPFAS + Inputs ----------
        self.fig, (self.axZ, self.axU) = plt.subplots(2, 1, figsize=(10, 8))

        # ΣPFAS
        (self.line_Z_hist,) = self.axZ.plot([], [], lw=2, label="Σ PFAS (measured)")
        (self.line_Z_pred,) = self.axZ.plot([], [], "--", lw=2, label="Σ PFAS (predicted)")
        self.axZ.set_ylabel("Σ PFAS [M]")
        self.axZ.set_title("MPC – Total PFAS and prediction horizon")
        self.axZ.grid(True)
        self.axZ.legend(loc="best")
        self.axZ.set_xlim(0.0, self.t_max)
        z_ylim_max = self.z0 * 1.05 if self.z0 > 0 else 1.0
        self.axZ.set_ylim(0.0, z_ylim_max)

        # Inputs
        (self.line_U_so3_hist,) = self.axU.plot([], [], lw=2, label="SO₃ (hist)")
        (self.line_U_cl_hist,)  = self.axU.plot([], [], lw=2, label="Cl⁻ (hist)")
        (self.line_U_so3_pred,) = self.axU.plot([], [], "--", lw=2, label="SO₃ (plan)")
        (self.line_U_cl_pred,)  = self.axU.plot([], [], "--", lw=2, label="Cl⁻ (plan)")
        self.axU.set_ylabel("Dose [M]")
        self.axU.set_xlabel("Time [s]")
        self.axU.grid(True)
        self.axU.legend(loc="best")
        self.axU.set_xlim(0.0, self.t_max)
        u_ylim_max = float(np.max(self.u_max)) * 1.05 if self.u_max.size and np.max(self.u_max) > 0 else 1.0
        self.axU.set_ylim(0.0, u_ylim_max)
        if self.u_max.size >= 1:
            self.axU.axhline(self.u_max[0], linestyle=":", linewidth=1)
        if self.u_max.size >= 2:
            self.axU.axhline(self.u_max[1], linestyle="--", linewidth=1)

        self.fig.tight_layout()

        # ---------- Individual states ----------
        self.make_state_grid = bool(make_state_grid)
        self.state_labels = [f"PFAS{i+1}" for i in range(7)] + ["F⁻"]
        self.state_lines_hist, self.state_lines_pred = [], []
        self.ax_states = None

        if self.make_state_grid:
            self.figX, self.ax_states = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
            self.ax_states = self.ax_states.flatten()
            for i in range(8):
                (lh,) = self.ax_states[i].plot([], [], lw=1.5, alpha=0.95,
                                               label=f"{self.state_labels[i]} (hist)")
                (lp,) = self.ax_states[i].plot([], [], "--", lw=1.5, alpha=0.95,
                                               label=f"{self.state_labels[i]} (pred)")
                self.ax_states[i].set_title(self.state_labels[i])
                self.ax_states[i].set_ylabel("Concentration [M]")
                self.ax_states[i].grid(True)
                self.state_lines_hist.append(lh)
                self.state_lines_pred.append(lp)
            for ax in self.ax_states[-2:]:
                ax.set_xlabel("Time [s]")
            self.figX.tight_layout()
            self._sync_pfas_y_with_total()

    # ---------- helpers ----------
    def _sync_pfas_y_with_total(self):
        if self.ax_states is None:
            return
        ymin, ymax = self.axZ.get_ylim()
        for i, ax in enumerate(self.ax_states):
            if i < 7:
                ax.set_ylim(ymin, ymax)

    @staticmethod
    def _stairs(t0_abs, t_rel, y_vals, Ts):
        y_vals = np.asarray(y_vals, dtype=float)
        if y_vals.size == 0:
            return np.array([]), np.array([])
        t = t0_abs + np.asarray(t_rel, dtype=float)
        t_step = np.repeat(t, 2)
        y_step = np.repeat(y_vals, 2)
        t_step = np.concatenate([t_step, [t_step[-1] + Ts]])
        y_step = np.concatenate([y_step, [y_step[-1]]])
        return t_step, y_step

    # ---------- update ----------
    def update(self, t_hist, Z_hist, t_u_hist, U_hist,
               t0_abs, t_pred_rel, Z_pred, t_u_pred_rel, U_plan,
               X_hist=None, X_pred=None):
        # ΣPFAS
        self.line_Z_hist.set_data(t_hist, Z_hist)
        t_pred_abs = t0_abs + np.asarray(t_pred_rel, dtype=float)
        self.line_Z_pred.set_data(t_pred_abs, Z_pred)

        # expand y if needed and mirror to PFAS₁–₇
        z_max_now = max(np.max(Z_hist) if len(Z_hist) else 0,
                        np.max(Z_pred) if len(Z_pred) else 0)
        ymin, ymax = self.axZ.get_ylim()
        if z_max_now > 0.95 * ymax:
            self.axZ.set_ylim(0.0, 1.1 * z_max_now)
        self._sync_pfas_y_with_total()

        # inputs
        U_hist = np.asarray(U_hist, dtype=float)
        if U_hist.size > 0:
            self.line_U_so3_hist.set_data(t_u_hist, U_hist[:, 0])
            self.line_U_cl_hist.set_data(t_u_hist,  U_hist[:, 1])
        else:
            self.line_U_so3_hist.set_data([], [])
            self.line_U_cl_hist.set_data([], [])
        U_plan = np.asarray(U_plan, dtype=float)
        if U_plan.size > 0:
            t_step_so3, y_step_so3 = self._stairs(t0_abs, t_u_pred_rel, U_plan[:, 0], self.Ts)
            t_step_cl,  y_step_cl  = self._stairs(t0_abs, t_u_pred_rel, U_plan[:, 1], self.Ts)
            self.line_U_so3_pred.set_data(t_step_so3, y_step_so3)
            self.line_U_cl_pred.set_data(t_step_cl,  y_step_cl)
        else:
            self.line_U_so3_pred.set_data([], [])
            self.line_U_cl_pred.set_data([], [])

        # states
        if self.ax_states is not None and (X_hist is not None) and (X_pred is not None):
            X_hist = np.asarray(X_hist, dtype=float)
            X_pred = np.asarray(X_pred, dtype=float)

            for i in range(8):
                self.state_lines_hist[i].set_data(t_hist, X_hist[:, i])
                self.state_lines_pred[i].set_data(t_pred_abs, X_pred[:, i])

            # PFAS₁–₇ keep ΣPFAS limits; F⁻ auto-scale with margin
            self._sync_pfas_y_with_total()
            f_hist = X_hist[:, 7] if X_hist.shape[1] >= 8 else np.array([])
            f_pred = X_pred[:, 7] if X_pred.shape[1] >= 8 else np.array([])
            if f_hist.size or f_pred.size:
                f_all = np.concatenate([f_hist, f_pred])
                f_min, f_max = float(np.min(f_all)), float(np.max(f_all))
                if f_max == f_min:
                    pad = max(abs(f_max) * 0.1, 1e-15)
                    self.ax_states[7].set_ylim(f_max - pad, f_max + pad)
                else:
                    f_margin = 0.1 * (f_max - f_min)
                    self.ax_states[7].set_ylim(f_min - f_margin, f_max + f_margin)

        # fix x-range and draw
        self.axZ.set_xlim(0.0, self.t_max)
        self.axU.set_xlim(0.0, self.t_max)
        if self.ax_states is not None:
            for ax in self.ax_states:
                ax.set_xlim(0.0, self.t_max)


        plt.pause(1)
"""