from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Callable
import struct
import time

from pymodbus.client import ModbusSerialClient
from pymodbus.pdu import ExceptionResponse
from pymodbus.exceptions import ModbusIOException

# -------- register map --------
IR_TEMP = 0x0001  # two words @ 0x0001..0x0002  -> °C (float32)
IR_CONC = 0x0003  # two words @ 0x0003..0x0004  -> mg/L (float32)
IR_MV   = 0x0007  # two words @ 0x0007..0x0008  -> mV (float32)
IR_I1   = 0x0009  # two words @ 0x0009..0x000A  -> mA (float32)
IR_I2   = 0x000B  # two words @ 0x000B..0x000C  -> mA (float32)

REG_MAP: Dict[str, int] = {
    "temperature_c": IR_TEMP,
    "concentration_mgL": IR_CONC,
    "electrode_mV": IR_MV,
    "current1_mA": IR_I1,
    "current2_mA": IR_I2,
}

@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    parity: str = "N"       # "N", "E", "O"
    stopbits: int = 1
    bytesize: int = 8
    timeout: float = 1.0    # seconds

class FluorideAnalyzer:
    """
    Minimal Modbus-RTU reader.
    - float32 values as two 16-bit registers (default word order: HI, LO)
    - FC04 by default; set use_holding=True if your device uses FC03
    """

    def __init__(self,
                 cfg: SerialConfig,
                 device_id: int = 1,
                 big_endian_words: bool = True,
                 use_holding: bool = False,
                 retries: int = 2,
                 retry_delay: float = 0.1):
        self.cfg = cfg
        self.device_id = int(device_id)
        self.big_endian_words = bool(big_endian_words)
        self.use_holding = bool(use_holding)
        self.retries = int(retries)
        self.retry_delay = float(retry_delay)
        self._client: Optional[ModbusSerialClient] = None

    # ---- lifecycle ----
    def open(self) -> None:
        self._client = ModbusSerialClient(
            port=self.cfg.port,
            baudrate=self.cfg.baudrate,
            parity=self.cfg.parity,
            stopbits=self.cfg.stopbits,
            bytesize=self.cfg.bytesize,
            timeout=self.cfg.timeout,
        )
        if not self._client.connect():
            raise IOError(f"Failed to open {self.cfg.port} @ {self.cfg.baudrate} bps")

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # context manager sugar
    def __enter__(self) -> "FluorideAnalyzer":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---- public reads ----
    def read_temperature_c(self) -> float:
        return self._read_float(IR_TEMP)

    def read_concentration_mgL(self) -> float:
        return self._read_float(IR_CONC)

    def read_electrode_mV(self) -> float:
        return self._read_float(IR_MV)

    def read_current1_mA(self) -> float:
        return self._read_float(IR_I1)

    def read_current2_mA(self) -> float:
        return self._read_float(IR_I2)
    
    def read(self, seconds: float | None = None, hz: float = 2.0):
        """
        Default: one concentration sample in mg/L.
        read() -> float

        Period mode: sample for `seconds` at `hz` (samples/second), returning a list of floats.
        read(seconds=3, hz=5) -> ~15 samples
        """
        if seconds is None:
            return self.read_concentration_mgL()

        samples: list[float] = []
        period = 1.0 / max(0.1, float(hz))  # avoid 0; lower bound ~0.1 Hz
        end_t = time.time() + float(seconds)
        next_t = time.time()

        while True:
            samples.append(self.read_concentration_mgL())
            next_t += period
            now = time.time()
            if now >= end_t:
                break
            sleep_s = max(0.0, next_t - now)
            if sleep_s:
                time.sleep(sleep_s)

        return samples



    def read_all(self) -> Dict[str, float]:
        """
        Try a single block read (0x0001..0x000C, 12 regs) for speed.
        Fallback to per-signal reads if needed.
        """
        base = min(REG_MAP.values())               # 0x0001
        last = max(addr + 1 for addr in REG_MAP.values())  # inclusive end word
        count = last - base                        # 0x000C - 0x0001 + 1 => 12 regs
        try:
            regs = self._read_regs_block(base, count)
            out: Dict[str, float] = {}
            for name, addr in REG_MAP.items():
                off = (addr - base)
                hi, lo = regs[off], regs[off + 1]
                out[name] = self._decode_float(hi, lo)
            return out
        except Exception:
            # Fallback to individual reads
            return {
                "temperature_c": self.read_temperature_c(),
                "concentration_mgL": self.read_concentration_mgL(),
                "electrode_mV": self.read_electrode_mV(),
                "current1_mA": self.read_current1_mA(),
                "current2_mA": self.read_current2_mA(),
            }

    # ---- internals ----
    def _decode_float(self, hi: int, lo: int) -> float:
        if not self.big_endian_words:
            hi, lo = lo, hi
        raw = (hi << 16) | (lo & 0xFFFF)
        return struct.unpack("!f", raw.to_bytes(4, "big"))[0]

    def _read_float(self, addr: int) -> float:
        rr = self._read_regs(addr, count=2)
        hi, lo = rr.registers
        return self._decode_float(hi, lo)

    def _read_regs_block(self, address: int, count: int):
        rr = self._read_regs(address, count=count)
        regs = getattr(rr, "registers", None)
        if not regs or len(regs) != count:
            raise IOError(f"Block read failed @0x{address:04X} x{count}: {rr}")
        return regs

    def _read_regs(self, address: int, count: int):
        """
        Compatibility shim across PyModbus variants:
        tries parameter names in order: unit -> slave -> device_id.
        Includes simple retry on I/O errors.
        """
        assert self._client is not None
        fn: Callable[..., object] = (
            self._client.read_holding_registers if self.use_holding
            else self._client.read_input_registers
        )

        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                try:
                    return fn(address=address, count=count, unit=self.device_id)
                except TypeError:
                    try:
                        return fn(address=address, count=count, slave=self.device_id)
                    except TypeError:
                        return fn(address=address, count=count, device_id=self.device_id)
            except (ModbusIOException, OSError) as e:
                last_err = e
                time.sleep(self.retry_delay)

        raise IOError(f"Modbus read failed @0x{address:04X} x{count}: {last_err}")
