# sensor_modbus.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import struct

from pymodbus.client import ModbusSerialClient
from pymodbus.pdu import ExceptionResponse


# -------- device map you discovered --------
IR_TEMP = 0x0001  # word pair: 0x0001, 0x0002  -> °C
IR_PH = 0x0003  # word pair: 0x0003, 0x0004  -> mg/L


@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"              # e.g. "/dev/ttyUSB0" or "COM3"
    baudrate: int = 9600
    parity: str = "N"          # "N", "E", "O"
    stopbits: int = 2
    bytesize: int = 8
    timeout: float = 1.0       # seconds


class PHAnalyzer:
    """
    Minimal Modbus-RTU reader for your analyzer using pymodbus v4 API.
    Reads 32-bit floats from Input Registers at the addresses you found.

    Assumptions:
      • 32-bit IEEE-754 floats stored as [HighWord][LowWord] (big-endian by word).
      • Values exposed as Input Registers (FC 0x04).
    """

    def __init__(self, cfg: SerialConfig, device_id: int = 1, big_endian_words: bool = True):
        self.cfg = cfg
        self.device_id = int(device_id)
        self.big_endian_words = bool(big_endian_words)
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

    # ---- public reads ----
    def read_temperature_c(self) -> float:
        return self._read_float_ir(IR_TEMP)

    def read_pH(self) -> float:
        return self._read_float_ir(IR_PH)

    def read_all(self) -> Dict[str, float]:
        return {
            "temperature_c": self.read_temperature_c(),
            "pH": self.read_pH(),
        }

    # ---- internals ----
    def _read_float_ir(self, addr: int) -> float:
        """Read a 32-bit float from two Input Registers (FC04) at 'addr'."""
        if not self._client:
            raise RuntimeError("Client not open. Call open() first.")
        rr = self._client.read_input_registers(addr, count=2, device_id=self.device_id)
        if rr is None or isinstance(rr, ExceptionResponse) or getattr(rr, "isError", lambda: False)():
            raise IOError(f"IR read failed at 0x{addr:04X}: {rr}")
        hi, lo = rr.registers
        if not self.big_endian_words:
            hi, lo = lo, hi
        raw = (hi << 16) | (lo & 0xFFFF)
        return struct.unpack("!f", raw.to_bytes(4, "big"))[0]
