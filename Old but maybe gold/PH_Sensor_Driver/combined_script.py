from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Mapping, Tuple
import struct
import time

from pymodbus.client import ModbusSerialClient
from pymodbus.pdu import ExceptionResponse


# --------------------------
# Serial + Bus management
# --------------------------
@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"  # e.g. "/dev/ttyUSB0" or "COM3"
    baudrate: int = 9600
    parity: str = "N"           # "N", "E", "O"
    stopbits: int = 1
    bytesize: int = 8
    timeout: float = 1.0        # seconds
    silent_interval_s: float = 0.004  # quiet time between requests on a shared bus


class ModbusBus:
    """
    A single shared Modbus-RTU client for the whole RS-485 line.
    Not thread-safe. Perform requests sequentially.
    """
    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self._client: Optional[ModbusSerialClient] = None
        self._last_io_time: float = 0.0

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

    # Internal pacing so different devices don't talk over each other.
    def _respect_silent_interval(self):
        dt = time.monotonic() - self._last_io_time
        if dt < self.cfg.silent_interval_s:
            time.sleep(self.cfg.silent_interval_s - dt)

    def read_float32_from_ir(
        self,
        addr: int,
        device_id: int,
        big_endian_words: bool = True,
        retries: int = 2,
        retry_delay_s: float = 0.05,
    ) -> float:
        """
        Read a 32-bit float from two Input Registers (FC04).
        Assumes IEEE-754 float with 16-bit words; 'big_endian_words' controls word order.
        """
        if not self._client:
            raise RuntimeError("Bus not open. Call open() first.")

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                self._respect_silent_interval()
                rr = self._client.read_input_registers(addr, count=2, device_id=device_id)
                self._last_io_time = time.monotonic()

                if rr is None or isinstance(rr, ExceptionResponse) or getattr(rr, "isError", lambda: False)():
                    raise IOError(f"IR read failed at 0x{addr:04X}: {rr}")

                hi, lo = rr.registers
                if not big_endian_words:
                    hi, lo = lo, hi
                raw = (hi << 16) | (lo & 0xFFFF)
                return struct.unpack("!f", raw.to_bytes(4, "big"))[0]

            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(retry_delay_s)
                else:
                    raise

        # Should never get here
        if last_err:
            raise last_err
        raise IOError("Unknown IR read error")


# --------------------------
# Device wrappers
# --------------------------
class FloatMapDevice:
    """
    Generic float-register device.
    Provide:
      - device_id: Modbus address on the RS-485 line
      - regmap: mapping name -> (start_addr_of_float)
      - big_endian_words: True if device stores float as [HighWord][LowWord]
    """
    def __init__(
        self,
        bus: ModbusBus,
        device_id: int,
        regmap: Mapping[str, int],
        big_endian_words: bool = True,
    ):
        self.bus = bus
        self.device_id = int(device_id)
        self.regmap = dict(regmap)
        self.big_endian_words = bool(big_endian_words)

    def read_value(self, name: str) -> float:
        addr = self.regmap[name]
        return self.bus.read_float32_from_ir(addr, device_id=self.device_id, big_endian_words=self.big_endian_words)

    def read_all(self) -> Dict[str, float]:
        return {name: self.read_value(name) for name in self.regmap.keys()}


# --------------------------
# Concrete sensors
# --------------------------
# Adjust these addresses to match each device's manual/map.
# (Below mirrors your example: temperature at 0x0001, measurement at 0x0003.)

PH_REGMAP = {
    "temperature_c": 0x0001,  # word pair 0x0001/0x0002
    "pH":            0x0003,  # word pair 0x0003/0x0004
}

F_REGMAP = {
    "temperature_c": 0x0001,
    "fluoride_mgL":  0x0003,
}


class PHAnalyzer(FloatMapDevice):
    def __init__(self, bus: ModbusBus, device_id: int = 1, big_endian_words: bool = True):
        super().__init__(bus, device_id, PH_REGMAP, big_endian_words)

    # Convenience accessors (optional)
    def read_temperature_c(self) -> float:
        return self.read_value("temperature_c")

    def read_pH(self) -> float:
        return self.read_value("pH")


class FluorideAnalyzer(FloatMapDevice):
    def __init__(self, bus: ModbusBus, device_id: int = 2, big_endian_words: bool = True):
        super().__init__(bus, device_id, F_REGMAP, big_endian_words)

    def read_temperature_c(self) -> float:
        return self.read_value("temperature_c")

    def read_fluoride_mgL(self) -> float:
        return self.read_value("fluoride_mgL")


MEASUERMENT_INTERVAL = 10.0  # seconds between readings
# --------------------------
# Example usage
# --------------------------
def main():
    cfg = SerialConfig(
        port="/dev/ttyUSB0",
        baudrate=9600,
        parity="N",
        stopbits=1,
        bytesize=8,
        timeout=1.0,
        silent_interval_s=0.004,  # ~3.5 char times @ 9600 bps is ~3.6ms
    )
    bus = ModbusBus(cfg)

    
    try:
        bus.open()

        ph = PHAnalyzer(bus, device_id=2, big_endian_words=True)         # <- give your pH sensor's ID
        f  = FluorideAnalyzer(bus, device_id=1, big_endian_words=True)   # <- give your F- sensor's ID

        # poll both devices sequentially
        while True:
            readings = {
                "ph": ph.read_all(),
                "fluoride": f.read_all(),
            }
            print(readings)
            # a small delay keeps the line calm; tune as needed
            time.sleep(MEASUERMENT_INTERVAL)

    finally:
        bus.close()


if __name__ == "__main__":
    main()
