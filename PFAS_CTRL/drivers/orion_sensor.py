from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time


try:
    import serial  # pyserial
except ImportError as e:
    raise RuntimeError("pyserial is required: pip install pyserial") from e


@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"   # Prefer: /dev/serial/by-id/...
    baudrate: int = 115200       # USB is commonly 115200; RS232 often 9600
    parity: str = "N"            # "N", "E", "O"
    stopbits: int = 1            # 1 or 2
    bytesize: int = 8            # 7 or 8
    timeout: float = 1.0         # seconds


class OrionVersaStarPro:
    """
    Serial (USB/RS232) remote-control reader for Orion Versa Star Pro.

    Uses ASCII commands terminated by CR and waits for '>' prompt.

    Implements:
      - Single-Shot: SETREADTYPE ch, 3, Z; GETMEAS
      - Auto-Read:   SETREADTYPE ch, 1;    GETMEAS (waits for stability)

    Parsing:
      - Designed for your observed CSV output line, e.g.:
        ..., ISE, F-, 2.7, ppm, 96.0, mV, 25.0, C (MAN), -60.4, mV/dec, 1
      - Extracts fluoride value right after "F-".
    """

    PROMPT = b">"
    CR = b"\r"

    def __init__(
        self,
        cfg: SerialConfig,
        *,
        channel: int = 1,
        retries: int = 2,
        retry_delay: float = 0.1,
        inter_cmd_delay: float = 0.05,
        default_single_shot_s: int = 10,
        ion_marker: str = "F-",   # anchor for value parsing
    ):
        self.cfg = cfg
        self.channel = int(channel)
        self.retries = int(retries)
        self.retry_delay = float(retry_delay)
        self.inter_cmd_delay = float(inter_cmd_delay)
        self.default_single_shot_s = int(default_single_shot_s)
        self.ion_marker = str(ion_marker)

        self._ser: Optional[serial.Serial] = None

    # ---- lifecycle ----
    def open(self) -> None:
        if self._ser:
            return

        self._ser = serial.Serial(
            port=self.cfg.port,
            baudrate=self.cfg.baudrate,
            bytesize=serial.EIGHTBITS if self.cfg.bytesize == 8 else serial.SEVENBITS,
            parity=self._map_parity(self.cfg.parity),
            stopbits=self._map_stopbits(self.cfg.stopbits),
            timeout=self.cfg.timeout,
            write_timeout=self.cfg.timeout,
        )

        # Basic handshake
        self._send("SYSTEM", max_wait=5)
        self._send("GETUSERINFO", max_wait=5)
        self._send(f"SETCHANNEL {self.channel}", max_wait=5)

    def close(self) -> None:
        if self._ser:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def __enter__(self) -> "OrionVersaStarPro":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---- public API ----
    def read_measurement(
        self,
        *,
        single_shot_s: Optional[int] = None,
        max_wait: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Single-Shot measurement: return after fixed Z seconds (deterministic).
        """
        self._ensure_open()

        z = int(single_shot_s if single_shot_s is not None else self.default_single_shot_s)
        self._send(f"SETREADTYPE {self.channel}, 3, {z}", max_wait=5)

        if max_wait is None:
            # Must be > z; add slack
            max_wait = max(30.0, float(z) + 20.0)

        txt = self._send("GETMEAS", max_wait=max_wait)

        parsed = self._parse_measurement_text(txt, ion_marker=self.ion_marker)
        parsed["raw_text"] = txt
        parsed["channel"] = self.channel
        parsed["read_type"] = "single-shot"
        return parsed

    def read_autoread(self, *, max_wait: float = 180.0) -> Dict[str, Any]:
        """
        Auto-Read measurement: waits until meter decides value is stable.
        Can legitimately take a long time; use a big max_wait.

        If it times out, we abort the running command and raise.
        """
        self._ensure_open()

        self._send(f"SETREADTYPE {self.channel}, 1", max_wait=5)

        try:
            txt = self._send("GETMEAS", max_wait=max_wait)
        except TimeoutError:
            self.abort()
            raise

        parsed = self._parse_measurement_text(txt, ion_marker=self.ion_marker)
        parsed["raw_text"] = txt
        parsed["channel"] = self.channel
        parsed["read_type"] = "auto-read"
        return parsed

    def read_value(self, *, single_shot_s: Optional[int] = None) -> float:
        """
        Return the numeric value right after the ion marker (default "F-").
        """
        d = self.read_measurement(single_shot_s=single_shot_s)
        v = d.get("value", None)
        if v is None:
            raise ValueError(
                f"Could not parse '{self.ion_marker}' value from meter response: "
                f"{d.get('raw_text','')[:200]!r}"
            )
        return float(v)

    def read(
        self,
        seconds: float | None = None,
        hz: float = 2.0,
        *,
        single_shot_s: Optional[int] = None,
    ):
        """
        Like your Modbus driver:
          - read() -> float (one F- value)
          - read(seconds=..., hz=...) -> list[float] sampled with fixed period
        """
        if seconds is None:
            return self.read_value(single_shot_s=single_shot_s)

        samples: list[float] = []
        period = 1.0 / max(0.1, float(hz))
        end_t = time.time() + float(seconds)
        next_t = time.time()

        while True:
            try:
                samples.append(self.read_value(single_shot_s=single_shot_s))
            except Exception:
                samples.append(float("nan"))

            next_t += period
            now = time.time()
            if now >= end_t:
                break
            sleep_s = max(0.0, next_t - now)
            if sleep_s:
                time.sleep(sleep_s)

        return samples

    def get_mode(self) -> str:
        return self._send(f"GETMODE {self.channel}", max_wait=5)

    def get_channel_config(self) -> str:
        return self._send("GETCHCONFIG", max_wait=5)

    def abort(self) -> None:
        """
        Abort a running command (best-effort).
        Many firmwares accept 'ESC ESC' + CR.
        """
        try:
            self._send("ESC ESC", max_wait=5)
        except Exception:
            pass

    # ---- internals ----
    def _ensure_open(self) -> None:
        if not self._ser:
            raise RuntimeError("Serial port is not open. Use .open() or context manager.")

    def _send(self, cmd: str, *, max_wait: float) -> str:
        """
        Send a command terminated by CR and wait for '>' prompt.
        Includes retry on I/O issues.
        """
        self._ensure_open()
        assert self._ser is not None

        last_err: Optional[Exception] = None
        for _attempt in range(self.retries + 1):
            try:
                self._ser.reset_input_buffer()
                self._ser.write(cmd.encode("ascii") + self.CR)
                self._ser.flush()
                time.sleep(self.inter_cmd_delay)

                out = self._read_until_prompt(max_wait=max_wait)
                return out.decode(errors="ignore")
            except (OSError, serial.SerialException, TimeoutError) as e:
                last_err = e
                time.sleep(self.retry_delay)

        raise IOError(f"Meter command failed after retries: {cmd!r}: {last_err}")

    def _read_until_prompt(self, *, max_wait: float) -> bytes:
        assert self._ser is not None
        buf = bytearray()
        t0 = time.time()
        while True:
            buf += self._ser.read(256) or b""
            if self.PROMPT in buf:
                return bytes(buf)
            if time.time() - t0 > float(max_wait):
                raise TimeoutError("Timeout waiting for prompt from meter")

    @staticmethod
    def _map_parity(p: str) -> str:
        p = (p or "N").upper()
        if p in ("N", "NONE"):
            return serial.PARITY_NONE
        if p in ("E", "EVEN"):
            return serial.PARITY_EVEN
        if p in ("O", "ODD"):
            return serial.PARITY_ODD
        raise ValueError(f"Unsupported parity: {p!r} (use 'N','E','O')")

    @staticmethod
    def _map_stopbits(sb: int) -> float:
        if int(sb) == 1:
            return serial.STOPBITS_ONE
        if int(sb) == 2:
            return serial.STOPBITS_TWO
        raise ValueError(f"Unsupported stopbits: {sb!r} (use 1 or 2)")

    @staticmethod
    def _parse_measurement_text(txt: str, *, ion_marker: str = "F-") -> Dict[str, Any]:
        """
        Robust CSV parser tailored to your observed GETMEAS output.

        Extracts:
          - value + unit right after ion_marker (e.g. F-)
          - electrode_mV (the number immediately before the literal 'mV')
          - temperature_c (the number immediately before a token starting with 'C' e.g. 'C (MAN)')
          - slope_mV_dec (the number immediately before 'mV/dec')
          - mode (e.g. ISE)
          - ion (e.g. F-)
        """
        # Remove prompt, keep content
        cleaned = txt.replace(">", "").strip()

        # Some responses include echoed command + CRLF; keep only the CSV line if present
        # We pick the last line containing commas.
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        csv_line = None
        for ln in reversed(lines):
            if "," in ln:
                csv_line = ln
                break
        if csv_line is None:
            csv_line = cleaned

        fields = [f.strip() for f in csv_line.split(",")]

        out: Dict[str, Any] = {
            "value": None,
            "unit": None,
            "electrode_mV": None,
            "temperature_c": None,
            "slope_mV_dec": None,
            "mode": None,
            "ion": None,
        }

        # Mode is usually present as 'ISE' in your sample
        if "ISE" in fields:
            out["mode"] = "ISE"
        # Ion marker
        out["ion"] = ion_marker if ion_marker in fields else None

        # Parse anchored values
        for i, f in enumerate(fields):
            if f == ion_marker:
                # Next: value, then unit
                try:
                    out["value"] = float(fields[i + 1])
                    out["unit"] = fields[i + 2]
                except (IndexError, ValueError):
                    pass

            elif f == "mV":
                # Electrode mV is just before literal "mV"
                try:
                    out["electrode_mV"] = float(fields[i - 1])
                except (IndexError, ValueError):
                    pass

            elif f.lower() == "mv/dec":
                try:
                    out["slope_mV_dec"] = float(fields[i - 1])
                except (IndexError, ValueError):
                    pass

            elif f.startswith("C"):
                # Temperature is right before "C" or "C (MAN)"
                try:
                    out["temperature_c"] = float(fields[i - 1])
                except (IndexError, ValueError):
                    pass

        return out


# ---- example usage ----
if __name__ == "__main__":
    cfg = SerialConfig(
        port="/dev/ttyACM0",   # set to your working port or /dev/serial/by-id/...
        baudrate=115200,
        parity="N",            # your handshake looked like 8N1; change if needed
        stopbits=1,
        timeout=1.0,
    )

    with OrionVersaStarPro(cfg, channel=1, default_single_shot_s=10, ion_marker="F-") as m:
        print("Mode:", m.get_mode())

        meas = m.read_measurement(single_shot_s=10)
        print("Single-Shot measurement:", meas)
        print("F- value:", m.read_value(single_shot_s=10))

        samples = m.read(seconds=5, hz=1.0, single_shot_s=10)
        print("samples:", samples)

        auto_meas = m.read_autoread(max_wait=120.0)
        print("Auto-Read measurement:", auto_meas)
