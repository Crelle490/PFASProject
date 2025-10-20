# PFAS_CTRL/drivers/flow_slf3c.py
from smbus2 import SMBus, i2c_msg
import time

class SLF3C:
    """
    Minimal, stateless helper:
      SLF3C().read("flow1") -> returns flow in ml/min
    Hardcoded: i2c-1, mux 0x70, sensor 0x08, flow1->ch1, flow2->ch2.
    """

    I2C_BUS      = 1
    MUX_ADDR     = 0x70
    SENSOR_ADDR  = 0x08
    CHANNEL_MAP  = {"flow1": 0, "flow2": 1}

    CMD_START_WATER = 0x3608
    CMD_STOP        = 0x3FF9

    def read(self, sensor: str = "flow1", seconds: float | None = None, hz: float = 10.0):
        """
        Return flow in ml/min.
        - Single sample: read("flow1") -> float
        - Period:        read("flow1", seconds=5, hz=10) -> list[float]
        """
        if sensor not in self.CHANNEL_MAP:
            raise ValueError('sensor must be "flow1" or "flow2"')
        ch = self.CHANNEL_MAP[sensor]

        def read_once(bus):
            rx = i2c_msg.read(self.SENSOR_ADDR, 9)
            bus.i2c_rdwr(rx)
            data = list(rx)
            flow_raw = self._parse_word(data[0:3])
            # consume temp/flags to keep stream aligned
            _ = self._parse_word(data[3:6])
            _ = self._parse_word(data[6:9])
            return flow_raw / 500.0  # ml/min

        with SMBus(self.I2C_BUS) as bus:
            # select mux channel & start measurement
            bus.write_byte(self.MUX_ADDR, 1 << ch)
            time.sleep(0.001)
            self._write16(bus, self.SENSOR_ADDR, self.CMD_START_WATER)
            time.sleep(0.02)  # first data ready ~12 ms

            try:
                if seconds is None:
                    return read_once(bus)
                else:
                    samples: list[float] = []
                    period = 1.0 / max(0.1, float(hz))  # steady scheduler (>=0.1 Hz)
                    end_t = time.time() + float(seconds)
                    next_t = time.time()
                    while True:
                        samples.append(read_once(bus))
                        next_t += period
                        now = time.time()
                        if now >= end_t:
                            break
                        sleep_s = max(0.0, next_t - now)
                        if sleep_s:
                            time.sleep(sleep_s)
                    return samples
            finally:
                # stop sensor & deselect mux (best-effort)
                try: self._write16(bus, self.SENSOR_ADDR, self.CMD_STOP)
                except Exception: pass
                try: bus.write_byte(self.MUX_ADDR, 0x00)
                except Exception: pass



    # ----- internals -----
    @staticmethod
    def _crc8_sensirion(two_bytes: list[int]) -> int:
        c = 0xFF
        for b in two_bytes:
            c ^= b
            for _ in range(8):
                c = ((c << 1) ^ 0x31) & 0xFF if (c & 0x80) else ((c << 1) & 0xFF)
        return c

    @classmethod
    def _parse_word(cls, trio: list[int]) -> int:
        w = trio[0:2]; crc = trio[2]
        if cls._crc8_sensirion(w) != crc:
            raise ValueError("CRC mismatch")
        val = (w[0] << 8) | w[1]
        if val & 0x8000: val -= 0x10000
        return val

    @staticmethod
    def _write16(bus: SMBus, addr: int, cmd16: int) -> None:
        bus.write_i2c_block_data(addr, (cmd16 >> 8) & 0xFF, [cmd16 & 0xFF])
