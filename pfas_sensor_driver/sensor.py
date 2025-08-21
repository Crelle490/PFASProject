from __future__ import annotations
from typing import Optional, Tuple
import time

# pip install pymodbus==3.*  pyserial
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException


class FluorideIonSensorModbus:
    """
    RS-485 Modbus-RTU driver for the fluoride ion analyzer.

    What this class assumes from the manual:
      • Transport: RS-485 (Modbus-RTU)
      • Baud rates supported: 4800 / 9600 / 19200
      • Parity: None / Odd / Even
      • Stop bits: 1 or 2
      • Slave address: user-configurable "网络节点" (node)
    The manual shows how to set those on the instrument. It does NOT include a register
    map, so you must supply the addresses & scaling for the values you want to read.

    Typical usage:
        sensor = FluorideIonSensorModbus(
            port="/dev/ttyUSB0",
            slave=1,
            baudrate=9600,
            parity="N",
            stopbits=1,
            timeout=1.0,
        )
        sensor.open()
        conc, temp, stat = sensor.read_measurement(
            conc_reg=0x0000, conc_scale=1000.0,     # <-- fill in from vendor register map
            temp_reg=0x0001, temp_scale=100.0,     # <-- fill in from vendor register map
            stat_reg=0x0002                        # <-- optional
        )
        sensor.close()
    """

    SUPPORTED_BAUDS = (4800, 9600, 19200)  # from the device menu
    SUPPORTED_PARITIES = {"N": "N", "E": "E", "O": "O"}  # None/Even/Odd
    SUPPORTED_STOPBITS = (1, 2)

    def __init__(
        self,
        port: str,
        slave: int,
        baudrate: int = 9600,
        parity: str = "N",         # "N", "E", or "O"
        stopbits: int = 1,         # 1 or 2
        bytesize: int = 8,         # instrument menus imply 8 data bits as usual
        timeout: float = 1.0,
    ):
        if baudrate not in self.SUPPORTED_BAUDS:
            raise ValueError(f"baudrate must be one of {self.SUPPORTED_BAUDS}")
        if parity not in self.SUPPORTED_PARITIES:
            raise ValueError("parity must be 'N', 'E', or 'O'")
        if stopbits not in self.SUPPORTED_STOPBITS:
            raise ValueError("stopbits must be 1 or 2")

        self.port = port
        self.slave = int(slave)
        self.baudrate = int(baudrate)
        self.parity = parity
        self.stopbits = int(stopbits)
        self.bytesize = int(bytesize)
        self.timeout = float(timeout)

        self._client: Optional[ModbusSerialClient] = None

    # --- connection lifecycle ---

    def open(self) -> None:
        """Open the Modbus-RTU link."""
        self._client = ModbusSerialClient(
            method="rtu",
            port=self.port,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=self.timeout,
        )
        if not self._client.connect():
            raise IOError(f"Failed to open {self.port} at {self.baudrate} bps.")

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # --- reads ---

    def read_measurement(
        self,
        conc_reg: int,
        conc_scale: float = 1.0,
        temp_reg: Optional[int] = None,
        temp_scale: float = 1.0,
        stat_reg: Optional[int] = None,
        word_count: int = 1,
    ) -> Tuple[float, Optional[float], Optional[int]]:
        """
        Read fluoride concentration, optional temperature, optional status.

        You must pass the correct register addresses and any scaling (e.g., if the device
        stores mg/L * 1000, set conc_scale=1000.0).

        Args
        ----
        conc_reg : holding register for fluoride concentration
        conc_scale : divide raw register by this factor to get engineering units
        temp_reg : holding register for temperature (optional)
        temp_scale : divide raw temperature register by this factor
        stat_reg : holding register for a status/alarm code (optional)
        word_count : number of 16-bit words per value (default 1)

        Returns
        -------
        (conc, temp, status)
        """
        if not self._client:
            raise RuntimeError("Not connected. Call open() first.")

        try:
            conc = self._read_number(conc_reg, word_count) / float(conc_scale)
            temp = None
            status = None

            if temp_reg is not None:
                temp = self._read_number(temp_reg, word_count) / float(temp_scale)
            if stat_reg is not None:
                status = int(self._read_number(stat_reg, 1))

            return float(conc), (float(temp) if temp is not None else None), status

        except ModbusException as e:
            raise IOError(f"Modbus error while reading: {e}") from e

    # --- utilities ---

    def _read_number(self, start_reg: int, words: int = 1) -> int:
        """
        Read 'words' 16-bit holding registers and combine into an integer.
        Big-endian 16-bit words -> 32/48/etc as needed.
        """
        rr = self._client.read_holding_registers(start_reg, count=words, slave=self.slave)
        if rr.isError():
            raise ModbusException(rr)
        regs = rr.registers
        # Combine 16-bit words into a single integer (big-endian word order).
        val = 0
        for r in regs:
            val = (val << 16) | (r & 0xFFFF)
        return val

    # Optional: quick sanity read of a known-good register (once you have it)
    def ping(self, test_reg: int = 0x0000) -> bool:
        """Try one read to see if the device responds (True=ok)."""
        try:
            _ = self._client.read_holding_registers(test_reg, count=1, slave=self.slave)
            return True
        except Exception:
            return False

    @staticmethod
    def convert_4_20mA_to_units(current_mA: float, lo_units: float, hi_units: float) -> float:
        """
        Helper only if you *also* use the analog output.
        Converts a current in mA to engineering units based on configured range.
        """
        span = hi_units - lo_units
        return lo_units + (max(0.0, min(20.0, current_mA)) - 4.0) * (span / 16.0)
