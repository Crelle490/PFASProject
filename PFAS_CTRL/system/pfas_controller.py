from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any
import serial
import time


from PFAS_CTRL.drivers.pump_wx10 import WX10Pump
from PFAS_CTRL.drivers.gpio_control import GPIOCtrl
from PFAS_CTRL.drivers.flow_slf3c import SLF3C
from PFAS_CTRL.drivers.flouride_sensor import SerialConfig as FluorSerialConfig, FluorideAnalyzer
from PFAS_CTRL.drivers.ph_sensor import SerialConfig as PHSerialConfig, PHAnalyzer
from PFAS_CTRL.drivers.orion_sensor import SerialConfig as OrionSerialConfig, OrionVersaStarPro


# ---------------- Low-pass filter helper ----------------

class FirstOrderLPF:
    """Simple 1st-order low-pass filter with time-constant tau."""
    def __init__(self, tau_s: float, x0: float = 0.0):
        self.tau_s = float(max(1e-6, tau_s))
        self.y = float(x0)
        self._t_prev = None

    def reset(self, x0: float = 0.0):
        self.y = float(x0)
        self._t_prev = None

    def update(self, x: float, t: float | None = None) -> float:
        if t is None:
            t = time.monotonic()
        x = float(x)
        if self._t_prev is None:
            self._t_prev = t
            self.y = x
            return self.y
        dt = max(0.0, float(t - self._t_prev))
        self._t_prev = t
        alpha = dt / (self.tau_s + dt)
        self.y = (1.0 - alpha) * self.y + alpha * x
        return self.y


# ---- simple configs ----
@dataclass
class PumpBusConfig:
    port: str = "/dev/ttyUSB1"
    baudrate: int = 9600
    timeout: float = 0.5
    addrs: Dict[str, int] | None = None
    pumps: Dict[str, Dict[str, int]] | None = None
    pump_num_rollers: Dict[str, int] | None = None

    def __post_init__(self):
        if self.addrs is None:
            # Description -> address
            self.addrs = {
                "mix_to_reaction":   1,  # "Mix to reaction chamber"
                "pfas":              7,  # "PFAS"
                "c1":                3,  # "C1"
                "buffer":            4,  # "Buffer"
                "water":             5,  # "Water"
                "holding_to_valves": 6,  # "Holding to valves"
                "c2":                2,  # "C2"
            }
        if self.pump_num_rollers is None:
            # Description -> rollers
            self.pump_num_rollers = {
                "mix_to_reaction":   8,
                "pfas":              4,
                "c1":                8,
                "buffer":            4,
                "water":             4,
                "holding_to_valves": 4,
                "c2":                8,
            }


@dataclass
class PHBusConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    parity: str = "N"
    stopbits: int = 1
    timeout: float = 1.0
    device_id: int = 2


@dataclass
class FluorideBusConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    parity: str = "N"
    stopbits: int = 1
    timeout: float = 1.0
    device_id: int = 1


@dataclass
class OrionBusConfig:
    """
    Orion Versa Star Pro over USB serial.
    Use your working port (/dev/ttyACM0 or /dev/serial/by-id/...).
    """
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    parity: str = "N"
    stopbits: int = 1
    timeout: float = 1.0
    channel: int = 1
    ion_marker: str = "F-"   # value after this token
    default_single_shot_s: int = 10


class PFASController:
    """
    Minimal aggregator:
      - self.pump         : WX10Pump on RS-485 (shared bus)
      - self.pump_addrs   : logical name -> address
      - self.gpio         : GPIOCtrl (valves/fans)
      - self.flow         : SLF3C flow helper
      - self.ph           : PHAnalyzer (call .open()/.close() yourself)
      - self.fluoride     : FluorideAnalyzer (call .open()/.close() yourself)
      - self.orion        : OrionVersaStarPro (call .open()/.close() yourself)
      - self.logger       : optional TimelineLogger-like object
    """

    def __init__(
        self,
        pump_cfg: PumpBusConfig = PumpBusConfig(),
        ph_cfg: Optional[PHBusConfig] = None,
        fluoride_cfg: Optional[FluorideBusConfig] = None,
        orion_cfg: Optional[OrionBusConfig] = None,
        *,
        logger: Any | None = None,
        gpio_active_low: bool = False,
        gpio_chip: str = "/dev/gpiochip4",
    ):
        # store logger
        self.logger = logger

        # Pump bus
        self.pump_cfg = pump_cfg
        self.pump_ser = serial.Serial(
            pump_cfg.port, pump_cfg.baudrate,
            bytesize=8, parity=serial.PARITY_EVEN, stopbits=1,
            timeout=pump_cfg.timeout,
        )
        self.pump = WX10Pump(self.pump_ser)

        self.pump_addrs = dict(pump_cfg.addrs)
        self.pump_num_rollers = dict(pump_cfg.pump_num_rollers)

        # volumes
        self.reactor_volume_ml = 6.5   # measured reactor volume (L)
        self.reactor_tube_volume = 3.8    # measured tube/dead volume (D)

        # GPIO + Flow
        self.gpio = GPIOCtrl(
            active_low=gpio_active_low,
            chip_path=gpio_chip,
            logger=self.logger,
        )
        self.gpio.open()
        self.flow = SLF3C()

        # pH (not opened here)
        self.ph = None
        if ph_cfg:
            self.ph_cfg = ph_cfg
            self.ph = PHAnalyzer(
                PHSerialConfig(
                    port=ph_cfg.port,
                    baudrate=ph_cfg.baudrate,
                    parity=ph_cfg.parity,
                    stopbits=ph_cfg.stopbits,
                    timeout=ph_cfg.timeout,
                ),
                device_id=ph_cfg.device_id,
            )

        # Fluoride Modbus (not opened here)
        self.fluoride = None
        if fluoride_cfg:
            self.fluoride_cfg = fluoride_cfg
            self.fluoride = FluorideAnalyzer(
                FluorSerialConfig(
                    port=fluoride_cfg.port,
                    baudrate=fluoride_cfg.baudrate,
                    parity=fluoride_cfg.parity,
                    stopbits=fluoride_cfg.stopbits,
                    timeout=fluoride_cfg.timeout,
                ),
                device_id=fluoride_cfg.device_id,
            )

        # Orion Versa Star Pro (not opened here)
        self.orion = None
        if orion_cfg:
            self.orion_cfg = orion_cfg
            self.orion = OrionVersaStarPro(
                OrionSerialConfig(
                    port=orion_cfg.port,
                    baudrate=orion_cfg.baudrate,
                    parity=orion_cfg.parity,
                    stopbits=orion_cfg.stopbits,
                    timeout=orion_cfg.timeout,
                ),
                channel=orion_cfg.channel,
                default_single_shot_s=orion_cfg.default_single_shot_s,
                ion_marker=orion_cfg.ion_marker,
            )

    # --- logging helpers -------------------------------------------------
    # --- logging helpers 

    def _log_pump_run(self, pump_name: str, addr: int, speed_pct: float, volume_ml: float | None):
        if self.logger is None:
            return
        channel = f"pump_{addr}"
        self.logger.log(channel, speed_pct, volume_ml=volume_ml)

    def _log_pump_stop(self, addr: int):
        if self.logger is None:
            return
        channel = f"pump_{addr}"
        self.logger.log(channel, 0.0)

    def close(self):
        """Optional helper to close serial + GPIO."""
        try:
            if self.pump_ser and self.pump_ser.is_open:
                self.pump_ser.close()
        except Exception:
            pass
        try:
            self.gpio.close()
        except Exception:
            pass

    # --- Flow sensor shim ------------------------------------------------

    def _read_flow_ml_min(self) -> float:
        """
        Read flow in mL/min from SLF3C. This tries common method names.
        Edit this if your SLF3C API differs.
        """
        f = self.flow

        # Common patterns
        for name in ("read_flow_ml_min", "read_flow_mL_min", "flow_ml_min", "get_flow_ml_min", "read"):
            if hasattr(f, name):
                v = getattr(f, name)()
                return float(v)

        # Sometimes .read() returns dict
        if hasattr(f, "read"):
            d = f.read()
            if isinstance(d, dict):
                for k in ("flow_ml_min", "flow_mL_min", "flow"):
                    if k in d:
                        return float(d[k])

        raise AttributeError("SLF3C flow driver has no recognized read method. Update _read_flow_ml_min().")

    def _run_with_flow_stop(
        self,
        *,
        stop_addr: int,
        stop_fn,
        deadline_s: float,
        flow_zero_threshold_ml_min: float,
        flow_zero_hold_s: float,
        lpf_tau_s: float,
        sample_hz: float,
        log_prefix: str = "",
    ) -> dict:
        """
        Monitor flow sensor and stop pump early if filtered flow remains below threshold
        for flow_zero_hold_s.
        """
        lpf = FirstOrderLPF(tau_s=lpf_tau_s, x0=0.0)
        period = 1.0 / max(0.5, float(sample_hz))
        zero_start = None

        t0 = time.monotonic()
        deadline = t0 + float(deadline_s)
        last_flow = None
        last_flow_f = None

        while True:
            now = time.monotonic()
            if now >= deadline:
                break

            # sample flow
            try:
                flow = self._read_flow_ml_min()
            except Exception:
                flow = float("nan")

            flow_f = lpf.update(flow, t=now)
            last_flow = flow
            last_flow_f = flow_f

            # zero detection (filtered)
            if flow_zero_threshold_ml_min is not None and flow_zero_hold_s is not None:
                if (flow_f == flow_f) and (flow_f <= float(flow_zero_threshold_ml_min)):  # not NaN + below thr
                    if zero_start is None:
                        zero_start = now
                    elif (now - zero_start) >= float(flow_zero_hold_s):
                        # stop early
                        try:
                            stop_fn(address=stop_addr)
                        except TypeError:
                            stop_fn(stop_addr)
                        return {
                            "stopped_early": True,
                            "reason": "flow_zero",
                            "flow_last_ml_min": last_flow,
                            "flow_f_last_ml_min": last_flow_f,
                            "elapsed_s": now - t0,
                        }
                else:
                    zero_start = None

            time.sleep(min(0.1, period))

        # normal stop at deadline
        return {
            "stopped_early": False,
            "reason": "deadline",
            "flow_last_ml_min": last_flow,
            "flow_f_last_ml_min": last_flow_f,
            "elapsed_s": time.monotonic() - t0,
        }

    # --- helpers ---------------------------------------------------------

    def flow_rate_for_pump(self, pump_name: str, speed: float) -> float:
        if pump_name not in self.pump_addrs:
            raise KeyError(f"Unknown pump '{pump_name}'. Known: {list(self.pump_addrs.keys())}")

        addr = self.pump_addrs[pump_name]
        if addr == 1:
            return 0.11889 * speed - 0.03
        elif addr == 2:
            return 0.10591 * speed - 0.0156
        elif addr == 3:
            return 0.11531 * speed - 0.08833
        elif addr == 4:
            return 0.19518 * speed - 0.00764
        elif addr == 5:
            return 0.19518 * speed - 0.00764
        elif addr == 6:
            return 0.17918 * speed - 0.07183
        elif addr == 7:
            return 0.19385 * speed - 0.03210
        else:
            raise ValueError(f"No calibration defined for pump '{pump_name}' (addr {addr})")

    def _dose(self, pump_name: str, volume_ml: float, speed: float = 50.0, cw: bool = True) -> float:
        if pump_name not in self.pump_addrs:
            raise ValueError(f"Unknown pump {pump_name!r}")
        flow_ml_min = self.flow_rate_for_pump(pump_name, speed)
        if flow_ml_min <= 0:
            raise ValueError(f"Estimated flow for {pump_name} at {speed}% is <= 0")
        run_s = (volume_ml / flow_ml_min) * 60.0
        addr = self.pump_addrs[pump_name]

        self._log_pump_run(pump_name, addr, speed_pct=float(speed), volume_ml=float(volume_ml))
        self.pump.set_speed(speed, run=True, cw=cw, address=addr)
        time.sleep(run_s)
        self.pump.stop(address=addr)
        self._log_pump_stop(addr)

        return run_s

    def speed_for_flow(self, pump_name: str, flow_ml_min: float) -> float:
        if pump_name not in self.pump_addrs:
            raise KeyError(f"Unknown pump '{pump_name}'. Known: {list(self.pump_addrs.keys())}")

        addr = self.pump_addrs[pump_name]
        if addr == 1:
            speed = (float(flow_ml_min) + 0.03) / 0.11889
        elif addr == 2:
            speed = (float(flow_ml_min) + 0.0156) / 0.10591
        elif addr == 3:
            speed = (float(flow_ml_min) + 0.08833) / 0.11531
        elif addr == 4:
            speed = (float(flow_ml_min) + 0.00764) / 0.19518
        elif addr == 5:
            speed = (float(flow_ml_min) + 0.00764) / 0.19518
        elif addr == 6:
            speed = (float(flow_ml_min) + 0.07183) / 0.17918
        elif addr == 7:
            speed = (float(flow_ml_min) + 0.03210) / 0.19385
        else:
            raise ValueError(f"No calibration defined for pump '{pump_name}' (addr {addr})")

        return max(0.0, min(100.0, speed))



    # 1) Create a mix between PFAS, C1, and C2
    def create_mixture(
        self,
        total_ml: float,
        *,
        pfas: float,
        c1: float,
        c2: float,
        speed: float = 50.0,            # dosing speed for PFAS/C1/C2
        cw: bool = True,
        sequential: bool = True,
        settle_s: float = 0.5,
        # NEW:
        post_push_s: float = 10.0,      # seconds to pump waiting->stirrer after mix (0 disables)
        post_push_speed_pct: float | None = None,  # None => reuse 'speed'
    ) -> dict:
        """
        Dose PFAS/C1/C2 into the stirring chamber to reach 'total_ml'.
        Then (optionally) route to STIRRER and pump from the waiting chamber
        for 'post_push_s' seconds to prime/start mixing flow.
        """
        import time

        def norm(x: float) -> float:  # accept 0..1 or 0..100
            return x / 100.0 if x > 1.0 else x

        f_pfas, f_c1, f_c2 = map(norm, (pfas, c1, c2))
        if min(f_pfas, f_c1, f_c2) < 0:
            raise ValueError("Fractions must be ≥ 0")
        s = f_pfas + f_c1 + f_c2
        if s > 1.0001:
            raise ValueError(f"Fractions sum to {s:.3f} (> 1.0)")

        vols = {
            "pfas": total_ml * f_pfas,
            "c1":   total_ml * f_c1,
            "c2":   total_ml * f_c2,
        }

        result = {"volumes_ml": vols.copy(), "runs_s": {}}

        # --- dose PFAS/C1/C2 into stirrer (existing behavior) ---
        if sequential:
            for name, vol in vols.items():
                if vol <= 0:
                    continue
                result["runs_s"][name] = self._dose(name, vol, speed=speed, cw=cw)
                time.sleep(settle_s)
        else:
            plan = {}
            for name, vol in vols.items():
                if vol <= 0:
                    continue
                flow_ml_min = self.flow_rate_for_pump(name, speed)
                if flow_ml_min <= 0:
                    raise ValueError(f"Estimated flow for {name} at {speed}% is <= 0")
                run_s = (vol / flow_ml_min) * 60.0
                plan[name] = {"addr": self.pump_addrs[name], "run_s": run_s}
                result["runs_s"][name] = run_s

            for name, p in plan.items():
                self.pump.set_speed(speed, run=True, cw=cw, address=p["addr"])

            t0 = time.time()
            remaining = set(plan.keys())
            while remaining:
                elapsed = time.time() - t0
                to_stop = [n for n in list(remaining) if elapsed >= plan[n]["run_s"]]
                for n in to_stop:
                    self.pump.stop(address=plan[n]["addr"])
                    remaining.remove(n)
                time.sleep(0.05)

        # --- NEW: post-mix push from waiting chamber -> stirrer for 30 s ---
        post = {
            "enabled": bool(post_push_s and post_push_s > 0),
            "requested_seconds": float(post_push_s),
            "used_speed_percent": None,
            "estimated_flow_ml_min": None,
            "elapsed_s": 0.0,
            "estimated_transferred_ml": 0.0,
        }

        if post["enabled"]:
            # Route to stirrer
            if not hasattr(self, "gpio"):
                raise RuntimeError("GPIO controller not initialized")
            self.gpio.off("valve1")   # redudant
            self.gpio.off("valve2")  # to mixing

            pump_name = "holding_to_valves"  # waiting -> valves/stirrer path
            if pump_name not in self.pump_addrs:
                raise ValueError(f"Pump mapping missing: {pump_name!r}")
            addr = self.pump_addrs[pump_name]

            spd = float(post_push_speed_pct) if post_push_speed_pct is not None else float(speed)
            q_est = self.flow_rate_for_pump(pump_name, spd)  # mL/min
            if q_est <= 1e-9:
                raise ValueError("Estimated flow is ~0; increase speed or check model.")

            # Run to deadline
            t0 = time.monotonic()
            deadline = t0 + float(post_push_s)
            self.pump.set_speed(spd, run=True, cw=cw, address=addr)
            try:
                while True:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        break
                    time.sleep(min(0.1, rem))
            finally:
                self.pump.stop(address=addr)

            elapsed = time.monotonic() - t0
            transferred_ml_est = q_est * (elapsed / 60.0)

            post.update(
                used_speed_percent=spd,
                estimated_flow_ml_min=q_est,
                elapsed_s=elapsed,
                estimated_transferred_ml=transferred_ml_est,
            )

        result["post_push_waiting_to_stirrer"] = post
        return result

    
        # 2) Empty the stirring chamber reactor
    def supply_reactor(
        self,
        reaction_time_s: float,
        dosage_ml: float,
        *,
        cw: bool = True,
        speed_clip_warn: bool = True,
    ) -> dict:
        import math, time

        # geometry
        volume_ml = float(self.reactor_volume_ml)

        # target flow:
        # F = A*L / t  (mL/s)
        F_ml_s = volume_ml / float(reaction_time_s)
        F_ml_min = F_ml_s * 60.0

        # total run time
        delta_t_s = float(dosage_ml) / F_ml_s            # (mL) / (mL/s) = s
        t_total_s = float(reaction_time_s) + delta_t_s

        # choose speed for that flow (mL/min)
        pump_name = "mix_to_reaction"
        addr = self.pump_addrs[pump_name]
        speed_pct = self.speed_for_flow(pump_name, F_ml_min)
        if speed_clip_warn and not (0.0 < speed_pct < 100.0):
            print(f"[supply_reactor] speed clipped to {speed_pct:.1f}% for {F_ml_min:.3f} mL/min")
        
        # run
        self.pump.set_speed(speed_pct, run=True, cw=cw, address=addr)
        time.sleep(t_total_s)
        self.pump.stop(address=addr)

        return {
            "reactor_volume_ml": volume_ml,
            "reaction_time_s": float(reaction_time_s),
            "dosage_ml": float(dosage_ml),
            "target_flow_ml_s": F_ml_s,
            "target_flow_ml_min": F_ml_min,
            "pump_speed_percent": speed_pct,
            "t_total_s": t_total_s,
            "delta_t_s": delta_t_s,
        }


    # 3) Dispatch sensor fluid + mix buffer
    def dispatch_sensor(
        self,
        volume_ml: float = 10.0,
        *,
        speed_pct: float = 60.0,          # main (waiting->sensor) speed
        cw: bool = True,
        buffer_pct: float = 0.0,          # fraction of volume_ml to add as buffer
    ) -> dict:
        """
        Route to SENSOR and pump `volume_ml` from the waiting chamber while
        simultaneously dosing buffer at `buffer_pct * volume_ml`. Both pumps
        start and stop together (open-loop using the pump flow model).
        """

        # Route to sensor
        if not hasattr(self, "gpio"):
            raise RuntimeError("GPIO controller not initialized")
        self.gpio.off("valve1")   # to sensor
        self.gpio.on("valve2")    # not to stirrer
        self.gpio.off("valve3")

        # Resolve pumps/addresses
        main_name = "holding_to_valves"   # waiting -> valves
        if main_name not in self.pump_addrs:
            raise ValueError(f"Pump mapping missing: {main_name!r}")
        main_addr = self.pump_addrs[main_name]

        buf_name = "buffer"
        buf_addr = self.pump_addrs.get(buf_name) if buffer_pct and buffer_pct > 0 else None

        # Main flow & run time
        q_main = self.flow_rate_for_pump(main_name, speed_pct)  # mL/min
        if q_main <= 1e-9:
            raise ValueError("Estimated main flow is ~0; increase speed_pct or check model.")
        T = float(volume_ml) / q_main * 60.0  # seconds to deliver volume_ml at q_main

        # Buffer flow to match timing (same T)
        buf_speed = None
        buf_q_est = None
        buf_target_ml = max(0.0, float(volume_ml) * float(buffer_pct))
        clipped = False

        if buf_addr is not None:
            # Need q_buf so that V_buf = q_buf * T / 60  => q_buf = V_buf * 60 / T
            # Since T = 60*V_main/q_main, this simplifies to q_buf = q_main * buffer_pct
            q_buf_needed = q_main * float(buffer_pct)

            # Compute buffer speed to achieve q_buf_needed
            try:
                buf_speed = self.speed_for_flow(buf_name, q_buf_needed)
            except AttributeError:
                # If you don't have speed_for_flow, invert your linear models here.
                # Example (replace with your actual mapping):
                # rollers = self.pump_cfg.pumps[buf_name]["rollers"]
                # if rollers == 8: buf_speed = (q_buf_needed + 0.065) / 0.109
                # else:            buf_speed = (q_buf_needed + 0.067) / 0.196
                raise RuntimeError("speed_for_flow() is required for buffer ratio matching")

            # Clip to [0,100] and note clipping
            if buf_speed < 0.0:
                buf_speed = 0.0
                clipped = True
            if buf_speed > 100.0:
                buf_speed = 100.0
                clipped = True

            # Actual buffer flow with clipped speed (for reporting)
            buf_q_est = self.flow_rate_for_pump(buf_name, buf_speed)

        # Start both pumps, stop exactly at deadline
        t0 = time.monotonic()
        deadline = t0 + T

        # Start main
        self.pump.set_speed(speed_pct, run=True, cw=cw, address=main_addr)
        # Start buffer (if any)
        if buf_addr is not None and buf_speed is not None and buf_speed > 0.0:
            self.pump.set_speed(buf_speed, run=True, cw=cw, address=buf_addr)

        try:
            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                time.sleep(min(0.1, rem))
        finally:
            # Stop both
            self.pump.stop(address=main_addr)
            if buf_addr is not None:
                self.pump.stop(address=buf_addr)

        elapsed = time.monotonic() - t0

        # Report estimated delivered volumes (model-based)
        delivered_main_ml = q_main * (elapsed / 60.0)
        delivered_buf_ml = (buf_q_est * (elapsed / 60.0)) if buf_q_est is not None else 0.0

        return {
            "route": "sensor",
            "commanded_main_ml": float(volume_ml),
            "buffer_pct": float(buffer_pct),
            "estimated_main_flow_ml_min": q_main,
            "estimated_buffer_flow_ml_min": buf_q_est,
            "speed_main_percent": float(speed_pct),
            "speed_buffer_percent": (None if buf_speed is None else float(buf_speed)),
            "clipped_buffer_speed": bool(clipped),
            "run_time_s": T,
            "elapsed_s": elapsed,
            "delivered_main_ml_est": delivered_main_ml,
            "delivered_buffer_ml_est": delivered_buf_ml,
            "pump_addr_main": main_addr,
            "pump_addr_buffer": buf_addr,
            "total_to_sensor_ml_est": delivered_main_ml + delivered_buf_ml,
        }
    

    # 4) Dispatch rest of waiting to stirrer
    def dispatch_stirrer_rest(
    self,
    total_ml: float,
    already_sent_ml: float = 10.0,   # what you sent to sensor
    *,
    speed_pct: float = 60.0,
    cw: bool = True,
    ) -> dict:
        """
        Route to STIRRER and pump the remaining volume in the waiting chamber:
        remaining = max(0, total_ml - already_sent_ml)
        Open-loop timing from the pump model; stops exactly on deadline.
        """
        import time

        if total_ml < 0 or already_sent_ml < 0:
            raise ValueError("total_ml and already_sent_ml must be >= 0")

        remaining_ml = max(0.0, float(total_ml) - float(already_sent_ml))
        if not hasattr(self, "gpio"):
            raise RuntimeError("GPIO controller not initialized")

        # route to stirrer (opposite of your sensor route where valve1 was OFF)
        self.gpio.off("valve1") # redundant
        self.gpio.off("valve2") # to mixer

        pump_name = "holding_to_valves"
        if pump_name not in self.pump_addrs:
            raise ValueError(f"Pump mapping missing: {pump_name!r}")
        addr = self.pump_addrs[pump_name]

        q_est = self.flow_rate_for_pump(pump_name, speed_pct)  # mL/min
        if q_est <= 1e-9:
            raise ValueError("Estimated flow is ~0; increase speed_pct or check model.")

        run_s = (remaining_ml / q_est) * 60.0 if remaining_ml > 0 else 0.0

        elapsed = 0.0
        if remaining_ml > 0:
            self.pump.set_speed(speed_pct, run=True, cw=cw, address=addr)
            t0 = time.monotonic()
            deadline = t0 + run_s
            try:
                while True:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        break
                    time.sleep(min(0.1, rem))
            finally:
                self.pump.stop(address=addr)
            elapsed = time.monotonic() - t0

        return {
            "route": "stirrer",
            "total_waiting_ml": float(total_ml),
            "already_sent_ml": float(already_sent_ml),
            "remaining_pumped_ml": remaining_ml,
            "speed_percent": float(speed_pct),
            "estimated_flow_ml_min": q_est,
            "run_time_s": run_s,
            "elapsed_s": elapsed,
            "pump_addr": addr,
        }


    # 5) Add additional catalyst to stirrer
    def mapMPC2pump(
        self,
        u_prev,                # [SO3, Cl] in mol/L
        u_k,                   # [SO3, Cl] in mol/L (MPC target)
        *,
        Vs_ml: float,          # current batch volume in mixer [mL]
        Cc_so3: float,         # stock conc SO3 [mol/L]
        Cc_cl: float,          # stock conc Cl  [mol/L]
        pump_so3: str = "c1",
        pump_cl: str  = "c2",
        speed_pct: float = 99.0,
        cw: bool = True,
        eps: float = 1e-12,
        cap_frac: float = 0.95,   # optional: cap Δu < cap_frac*Cc for safety
    ) -> dict:
        """
        Compute ΔV for SO3 and Cl from Δu and add them by running both pumps at full speed.
        Each pump stops when its own volume target is met. Returns a log dict and updates volume.
        """
        import time
        import numpy as np

        u_prev = np.asarray(u_prev, float).ravel()
        u_k    = np.asarray(u_k,    float).ravel()
        if u_prev.size != 2 or u_k.size != 2:
            raise ValueError("u_prev and u_k must be length-2 arrays [SO3, Cl] in mol/L.")

        # current volume (L)
        V_L = float(Vs_ml) / 1000.0

        # Δu (mol/L) and positive parts
        dC_so3 = float(u_k[0] - u_prev[0])
        dC_cl  = float(u_k[1] - u_prev[1])
        dC_so3 = max(0.0, dC_so3)
        dC_cl  = max(0.0, dC_cl)

        # optional safety cap to avoid ΔC -> Cc singularity
        if dC_so3 >= cap_frac * Cc_so3:
            dC_so3 = cap_frac * Cc_so3
        if dC_cl  >= cap_frac * Cc_cl:
            dC_cl  = cap_frac * Cc_cl

        # per-channel dose volumes (L): Vc = (ΔC * V)/(Cc - ΔC)
        def _dose_vol(deltaC, Cc, V_L_):
            if deltaC <= 0.0:
                return 0.0
            denom = max(Cc - deltaC, eps)
            return (deltaC * V_L_) / denom

        Vc_so3_L = _dose_vol(dC_so3, Cc_so3, V_L)
        Vc_cl_L  = _dose_vol(dC_cl,  Cc_cl,  V_L)

        # convert to mL
        Vc_so3_ml = 1000.0 * Vc_so3_L
        Vc_cl_ml  = 1000.0 * Vc_cl_L

        # flows at full speed
        q_so3 = self.flow_rate_for_pump(pump_so3, speed_pct)  # mL/min
        q_cl  = self.flow_rate_for_pump(pump_cl,  speed_pct)  # mL/min
        if Vc_so3_ml > 0 and q_so3 <= 0:
            raise ValueError(f"Estimated flow for {pump_so3} at {speed_pct}% is <= 0")
        if Vc_cl_ml > 0 and q_cl <= 0:
            raise ValueError(f"Estimated flow for {pump_cl} at {speed_pct}% is <= 0")

        # run times for each pump (s) at full speed
        T_so3 = (Vc_so3_ml / q_so3) * 60.0 if Vc_so3_ml > 0 else 0.0
        T_cl  = (Vc_cl_ml  / q_cl)  * 60.0 if Vc_cl_ml  > 0 else 0.0

        # start both; stop each on its own deadline
        addr_so3 = self.pump_addrs[pump_so3]
        addr_cl  = self.pump_addrs[pump_cl]

        now = time.monotonic()
        deadlines = {}
        if Vc_so3_ml > 0:
            self.pump.set_speed(speed_pct, run=True, cw=cw, address=addr_so3)
            deadlines["so3"] = now + T_so3
        if Vc_cl_ml > 0:
            self.pump.set_speed(speed_pct, run=True, cw=cw, address=addr_cl)
            deadlines["cl"] = now + T_cl

        try:
            while deadlines:
                t = time.monotonic()
                to_stop = [k for k, d in deadlines.items() if t >= d]
                for k in to_stop:
                    if k == "so3":
                        self.pump.stop(address=addr_so3)
                    else:
                        self.pump.stop(address=addr_cl)
                    deadlines.pop(k, None)
                if deadlines:
                    time.sleep(0.05)
        finally:
            # safety stops
            try: self.pump.stop(address=addr_so3)
            except Exception: pass
            try: self.pump.stop(address=addr_cl)
            except Exception: pass

        # update batch volume
        V_new_ml = Vs_ml + Vc_so3_ml + Vc_cl_ml

        # --- LOG catalyst doses (if logger available) ---
        if self.logger is not None:
            if Vc_so3_ml > 0:
                self._log_pump_run("c1", addr_so3, speed_pct=float(speed_pct),
                                   volume_ml=float(Vc_so3_ml))
                self._log_pump_stop(addr_so3)
            if Vc_cl_ml > 0:
                self._log_pump_run("c2", addr_cl, speed_pct=float(speed_pct),
                                   volume_ml=float(Vc_cl_ml))
                self._log_pump_stop(addr_cl)


        return {
            "delta_u_M": {"so3": dC_so3, "cl": dC_cl},
            "dose_ml": {"so3": Vc_so3_ml, "cl": Vc_cl_ml, "total": Vc_so3_ml + Vc_cl_ml},
            "speeds_percent": {"so3": float(speed_pct) if Vc_so3_ml > 0 else 0.0,
                               "cl":  float(speed_pct) if Vc_cl_ml  > 0 else 0.0},
            "flows_ml_min": {"so3": q_so3 if Vc_so3_ml > 0 else 0.0,
                             "cl":  q_cl  if Vc_cl_ml  > 0 else 0.0},
            "run_time_s": {"so3": T_so3, "cl": T_cl},
            "Vs_ml_before": float(Vs_ml),
            "Vs_ml_after":  float(V_new_ml),
            "pumps": {"so3": pump_so3, "cl": pump_cl},
            "Vs_ml_after": float(V_new_ml),

        }


    # 6) exit fluid
    def exit_fluid(
        self,
        *,
        volume_ml: float | None = None,   # set this OR duration_s
        duration_s: float | None = None,
        speed_pct: float = 60.0,
        cw: bool = True,
        pump_name: str = "holding_to_valves",     # pump for final exit
        mixer_pump: str = "mix_to_reaction",      # NEW: pump that drains from mixer to holding
    ) -> dict:
        """
        Drain the fluid from the stirrer (mixer) chamber to the holding chamber,
        then evacuate it to waste.

        You can specify either a total volume (mL) or a run duration (s).
        If volume_ml is set, it is used for both mixer and exit purge.
        """

        import time

        # --- Safety checks ---
        if not hasattr(self, "gpio"):
            raise RuntimeError("GPIO controller not initialized")
        if mixer_pump not in self.pump_addrs:
            raise ValueError(f"Pump mapping missing: {mixer_pump!r}")
        if pump_name not in self.pump_addrs:
            raise ValueError(f"Pump mapping missing: {pump_name!r}")

        # --- Determine target ---
        if duration_s is None and volume_ml is None:
            raise ValueError("Provide either volume_ml or duration_s")

        # --- Stage 1: Mixer → Holding ---
        mixer_addr = self.pump_addrs[mixer_pump]
        q_mixer = self.flow_rate_for_pump(mixer_pump, speed_pct)
        if q_mixer <= 1e-9:
            raise ValueError("Estimated mixer flow is ~0; increase speed_pct or check model.")

        total_volume = float(volume_ml)+float(self.reactor_volume_ml)+float(self.reactor_tube_volume)
        mixer_run_s = (float(total_volume) / q_mixer * 60.0) if duration_s is None else float(duration_s)
        print(f"Draining mixer to holding for {mixer_run_s:.2f} s...")

        t0_mixer = time.monotonic()
        deadline_mixer = t0_mixer + mixer_run_s
        self.pump.set_speed(speed_pct, run=True, cw=cw, address=mixer_addr)
        try:
            while True:
                rem = deadline_mixer - time.monotonic()
                if rem <= 0:
                    break
                time.sleep(min(0.1, rem))
        finally:
            self.pump.stop(address=mixer_addr)

        elapsed_mixer = time.monotonic() - t0_mixer
        transferred_ml_est = q_mixer * (elapsed_mixer / 60.0)

        # --- Stage 2: Holding → Waste ---
        self.gpio.on("valve1")  # to exit
        self.gpio.on("valve2")  # bypass mixer

        exit_addr = self.pump_addrs[pump_name]
        q_exit = self.flow_rate_for_pump(pump_name, speed_pct)
        if q_exit <= 1e-9:
            raise ValueError("Estimated exit flow is ~0; increase speed_pct or check model.")

        exit_run_s = (float(volume_ml) / q_exit * 60.0) if duration_s is None else float(duration_s)
        print(f"Evacuating from holding to waste for {exit_run_s:.2f} s...")

        t0_exit = time.monotonic()
        deadline_exit = t0_exit + exit_run_s
        self.pump.set_speed(speed_pct, run=True, cw=cw, address=exit_addr)
        try:
            while True:
                rem = deadline_exit - time.monotonic()
                if rem <= 0:
                    break
                time.sleep(min(0.1, rem))
        finally:
            self.pump.stop(address=exit_addr)

        elapsed_exit = time.monotonic() - t0_exit
        evacuated_ml_est = q_exit * (elapsed_exit / 60.0)

        return {
            "stage1_mixer_to_holding": {
                "pump": mixer_pump,
                "elapsed_s": elapsed_mixer,
                "transferred_ml_est": transferred_ml_est,
                "speed_percent": float(speed_pct),
                "estimated_flow_ml_min": q_mixer,
            },
            "stage2_holding_to_waste": {
                "pump": pump_name,
                "elapsed_s": elapsed_exit,
                "evacuated_ml_est": evacuated_ml_est,
                "speed_percent": float(speed_pct),
                "estimated_flow_ml_min": q_exit,
            },
            "total_evacuated_est": transferred_ml_est + evacuated_ml_est,
        }



    # 7) Clean with water

    def flush_sensor_water(
    self,
    volume_ml: float = 20.0,
    *,
    speed_pct: float = 60.0,
    cw: bool = True,
    pump_name: str = "water",   # change if your water line has a different logical name
    ) -> dict:
        """
        Flush clean water through the SENSOR path.
        Opens sensor route and pumps `volume_ml` from the water line.
        Timing is open-loop using the pump flow model.
        """
        import time

        # Route to sensor (adjust if your plumbing is inverted)
        if not hasattr(self, "gpio"):
            raise RuntimeError("GPIO controller not initialized")
        self.gpio.off("valve1")  # not to stirrer
        self.gpio.on("valve2")   # to sensor
        self.gpio.on("valve3")

        # Wait for valves to settle
        time.sleep(8)
        self.gpio.off("valve3")

        # Resolve pump + address
        if pump_name not in self.pump_addrs:
            raise ValueError(f"Pump mapping missing: {pump_name!r}")
        addr = self.pump_addrs[pump_name]

        # Flow estimate & run time
        q_est = self.flow_rate_for_pump(pump_name, speed_pct)  # mL/min
        if q_est <= 1e-9:
            raise ValueError("Estimated flow is ~0; increase speed_pct or check model.")
        run_s = float(volume_ml) / q_est * 60.0

        # Run to a monotonic deadline
        t0 = time.monotonic()
        deadline = t0 + run_s
        self.pump.set_speed(speed_pct, run=True, cw=cw, address=addr)
        try:
            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                time.sleep(min(0.1, rem))
        finally:
            self.pump.stop(address=addr)

        elapsed = time.monotonic() - t0
        flushed_ml_est = q_est * (elapsed / 60.0)

        # Wait for valves to settle
        self.gpio.off("valve3")
        time.sleep(3)
        self.gpio.on("valve3")


        return {
            "route": "sensor_water_flush",
            "pump": pump_name,
            "pump_addr": addr,
            "commanded_volume_ml": float(volume_ml),
            "speed_percent": float(speed_pct),
            "estimated_flow_ml_min": q_est,
            "run_time_s": run_s,
            "elapsed_s": elapsed,
            "flushed_ml_est": flushed_ml_est,
        }

 # 8) Initial fill/prime of PFAS, C1, C2, BUFFER, WATER lines
    def initaize(
        self,
        *,
        # choose one of these ways to specify volumes:
        pfas_ml: float | None = None,
        c1_ml: float | None = None,
        c2_ml: float | None = None,
        buffer_ml: float | None = None,
        water_ml: float | None = None,
        volumes_ml: dict[str, float] | None = {
            "pfas": 6.20, "c1": 6.6, "c2": 6.6, "buffer": 8.3, "water": 8.68
        },  # Results; 2: 7.20, 3: 7.48, 7: 7.53, 5: 8.3, 4: 8.68

        default_volume_ml: float = 4.9,     # used if a pump's volume isn't provided
        speed_pct: float = 99.0,
        cw: bool = True,
        simultaneous: bool = True,
        settle_s: float = 0.2,
    ) -> dict:
        """
        Prime/fill PFAS, C1, C2, BUFFER, and WATER lines with possibly different volumes.

        Volume specification (priority):
        1) volumes_ml={"pfas": ..., "c1": ..., "c2": ..., "buffer": ..., "water": ...}
        2) per-pump kwargs (pfas_ml, c1_ml, c2_ml, buffer_ml, water_ml)
        3) fallback to default_volume_ml for any missing entry

        Args:
            default_volume_ml: used for any pump not explicitly given a volume.
            speed_pct: pump speed (applied to all selected pumps).
            simultaneous: if True, start all specified pumps and stop each at its own deadline.
                        if False, run sequentially in order: PFAS -> C1 -> C2 -> BUFFER -> WATER.
        Returns:
            dict with per-pump timing, estimated delivered volumes, and speeds.
        """
        # Resolve desired volumes by name
        names = ["pfas", "c1", "c2", "buffer", "water"]
        desired = {n: default_volume_ml for n in names}

        if volumes_ml:
            for n in names:
                if n in volumes_ml:
                    desired[n] = float(volumes_ml[n])

        if pfas_ml   is not None: desired["pfas"]   = float(pfas_ml)
        if c1_ml     is not None: desired["c1"]     = float(c1_ml)
        if c2_ml     is not None: desired["c2"]     = float(c2_ml)
        if buffer_ml is not None: desired["buffer"] = float(buffer_ml)
        if water_ml  is not None: desired["water"]  = float(water_ml)

        # Map names -> addresses from your config; validate presence
        try:
            name_to_addr = {name: self.pump_addrs[name] for name in names}
        except KeyError as e:
            raise KeyError(f"Missing pump mapping for {e.args[0]!r} in pump_addrs") from e

        # Build plan for only those with volume > 0
        plan = {}  # name -> dict(addr, q_ml_min, run_s, target_ml)
        for name in names:
            target_ml = float(desired[name])
            if target_ml <= 0.0:
                continue  # skip zero/negative targets
            q_ml_min = self.flow_rate_for_pump(name, speed_pct)
            if q_ml_min <= 1e-9:
                raise ValueError(f"Estimated flow for pump '{name}' at {speed_pct}% is ~0.")
            run_s = target_ml / q_ml_min * 60.0
            plan[name] = {
                "addr": name_to_addr[name],
                "q_ml_min": q_ml_min,
                "run_s": run_s,
                "target_ml": target_ml,
            }

        results = {
            "speed_percent": float(speed_pct),
            "cw": bool(cw),
            "simultaneous": bool(simultaneous),
            "pumps": {},   # name -> details
        }

        if not plan:
            results["note"] = "No pumps had a positive target volume; nothing ran."
            results["total_elapsed_s"] = 0.0
            return results

        if simultaneous:
            # Start all, then stop each at its own deadline relative to t0
            t0 = time.monotonic()
            for name, info in plan.items():
                self.pump.set_speed(speed_pct, run=True, cw=cw, address=info["addr"])

            try:
                remaining = set(plan.keys())
                while remaining:
                    now = time.monotonic()
                    for name in list(remaining):
                        elapsed = now - t0
                        if elapsed >= plan[name]["run_s"]:
                            self.pump.stop(address=plan[name]["addr"])
                            remaining.remove(name)
                    time.sleep(0.05)
            finally:
                # Safety stop
                for name, info in plan.items():
                    try:
                        self.pump.stop(address=info["addr"])
                    except Exception:
                        pass

            t_elapsed = time.monotonic() - t0
            for name, info in plan.items():
                delivered_ml_est = info["q_ml_min"] * (info["run_s"] / 60.0)
                results["pumps"][name] = {
                    "address": info["addr"],
                    "target_ml": info["target_ml"],
                    "estimated_flow_ml_min": info["q_ml_min"],
                    "planned_run_s": info["run_s"],
                    "delivered_ml_est": delivered_ml_est,
                }
            results["total_elapsed_s"] = t_elapsed
            return results

        else:
            # Sequential: PFAS -> C1 -> C2 -> BUFFER -> WATER
            total_elapsed = 0.0
            sequence = [n for n in ["pfas", "c1", "c2", "buffer", "water"] if n in plan]
            for name in sequence:
                info = plan[name]
                self.pump.set_speed(speed_pct, run=True, cw=cw, address=info["addr"])
                t0 = time.monotonic()
                deadline = t0 + info["run_s"]
                try:
                    while True:
                        rem = deadline - time.monotonic()
                        if rem <= 0:
                            break
                        time.sleep(min(0.1, rem))
                finally:
                    self.pump.stop(address=info["addr"])

                elapsed = time.monotonic() - t0
                total_elapsed += elapsed
                delivered_ml_est = info["q_ml_min"] * (elapsed / 60.0)
                results["pumps"][name] = {
                    "address": info["addr"],
                    "target_ml": info["target_ml"],
                    "estimated_flow_ml_min": info["q_ml_min"],
                    "planned_run_s": info["run_s"],
                    "elapsed_s": elapsed,
                    "delivered_ml_est": delivered_ml_est,
                }

                # small pause between pumps
                if name != sequence[-1]:
                    time.sleep(settle_s)
                    total_elapsed += float(settle_s)

            results["total_elapsed_s"] = total_elapsed
            return results

    # 9) initialize reactors - fill tubes before starting reactions and empty afterwards
    def initialize_reactors(self):
        
        # parameters
        volume = 3.8 # mL volume to fill tubes
        pump_name = "mix_to_reaction"
        speed_pct = 99  

        # compute run time
        q_est = self.flow_rate_for_pump(pump_name, speed_pct)
        t_run = (volume / q_est) * 60.0

        # run pump
        self.pump.set_speed(speed_pct, run=True, cw=True, address=self.pump_addrs[pump_name])
        time.sleep(t_run)
        self.pump.stop(address=self.pump_addrs[pump_name])

        return {
            "reactor_initialization": {
                "volume_ml": volume,
                "pump_name": pump_name,
                "speed_percent": speed_pct,
                "estimated_flow_ml_min": q_est,
                "run_time_s": t_run,
            }
        }

    # 10) empty tubes
    def empty_tubes(self, volume_ml: float = 25.0, speed_pct: float = 99.0, preflush_s: float = 30.0):
        """
        1) Run all supply pumps + mix_to_reaction for preflush_s seconds.
        2) Stop ALL those pumps.
        3) Drain mixer->holding and evacuate to waste for ~volume_ml (via exit_fluid()).
        Always hard-stop pumps on exit.
        """
        import time

        pumps_supply = ["pfas", "c1", "c2", "buffer", "water"]
        preflush_dirs = {p: False for p in pumps_supply}  # supply pumps run CCW here (adjust if needed)
        mix_pump = "mix_to_reaction"                      # mixer drain direction typically CW

        # Resolve addresses up-front (fail fast)
        addrs = {}
        for p in pumps_supply + [mix_pump]:
            if p not in self.pump_addrs:
                raise ValueError(f"Pump mapping missing: {p!r}")
            addrs[p] = self.pump_addrs[p]

        print(f"Starting tube emptying: preflush {preflush_s:.1f}s, then ~{volume_ml} mL purge at {speed_pct}%...")

        # Helper to hard-stop pumps
        def _stop(pumps):
            for p in pumps:
                try:
                    # Prefer an explicit stop if available
                    if hasattr(self.pump, "stop"):
                        self.pump.stop(address=addrs[p])
                    else:
                        self.pump.set_speed(0, run=False, address=addrs[p])
                except Exception:
                    pass

        try:
            # --- Step 1: Pre-flush (all supply + mixer) ---
            for p in pumps_supply:
                self.pump.set_speed(speed_pct, run=True, cw=preflush_dirs[p], address=addrs[p])
            self.pump.set_speed(speed_pct, run=True, cw=True, address=addrs[mix_pump])

            t0 = time.monotonic()
            while time.monotonic() - t0 < preflush_s:
                time.sleep(0.1)

            # --- Step 2: Stop ALL preflush pumps before calling exit_fluid ---
            _stop(pumps_supply + [mix_pump])

            # --- Step 3: Drain mixer and evacuate to waste (your exit_fluid handles valves + timing) ---
            self.exit_fluid(
                volume_ml=volume_ml,
                speed_pct=speed_pct,
                cw=True,
                pump_name="holding_to_valves",   # keep if this matches your plumbing
            )

        finally:
            # Safety net: ensure everything is stopped even if exceptions occur
            _stop(pumps_supply + [mix_pump])

        print("✅ Tubes emptied successfully.")


    # 11) initialize sensor
    def initialize_sensor(
        self,
        volume_ml: float = 7.3,
        speed_pct: float = 99,
        cw: bool = True,
        pump_name: str = "holding_to_valves",
    ) -> dict:
        """
        Prime the SENSOR line by pumping `volume_ml` from the waiting chamber.
        Timing is open-loop using the pump flow model.
        """

        # Route to sensor
        if not hasattr(self, "gpio"):
            raise RuntimeError("GPIO controller not initialized")
        self.gpio.off("valve1")  # to sensor
        self.gpio.on("valve2")   # not to stirrer

        # Resolve pump + address
        if pump_name not in self.pump_addrs:
            raise ValueError(f"Pump mapping missing: {pump_name!r}")
        addr = self.pump_addrs[pump_name]

        # Flow estimate & run time
        q_est = self.flow_rate_for_pump(pump_name, speed_pct)  # mL/min
        if q_est <= 1e-9:
            raise ValueError("Estimated flow is ~0; increase speed_pct or check model.")
        run_s = float(volume_ml) / q_est * 60.0

        # Run to a monotonic deadline
        t0 = time.monotonic()
        deadline = t0 + run_s
        self.pump.set_speed(speed_pct, run=True, cw=cw, address=addr)
        try:
            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                time.sleep(min(0.1, rem))
        finally:
            self.pump.stop(address=addr)

        elapsed = time.monotonic() - t0
        primed_ml_est = q_est * (elapsed / 60.0)

        return {
            "route": "sensor_initialization",
            "pump": pump_name,
            "pump_addr": addr,
            "commanded_volume_ml": float(volume_ml),
            "speed_percent": float(speed_pct),
            "estimated_flow_ml_min": q_est,
            "run_time_s": run_s,
            "elapsed_s": elapsed,
            "primed_ml_est": primed_ml_est,
        }
