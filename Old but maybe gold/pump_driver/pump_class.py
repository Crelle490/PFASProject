from __future__ import annotations
import serial
import time
#import gpiod
from gpiod.line import Direction, Value
from typing import Optional, Dict, Any, Iterable

try:
    import serial  # pyserial
except ImportError as e:
    raise RuntimeError("pyserial is required: pip install pyserial") from e


class WX10Pump:
    """
    Driver for Longer T100-S500 & WX10 peristaltic pumps (e.g., T100-WX10-14-H)
    over RS-485 using the vendor protocol (Appendix B).

    - Uses 8-E-1, 9600 bps by default.
    - Handles E8/E9 byte-stuffing.
    - Calculates/validates FCS (XOR).
    - Exposes set_speed / stop / get_state / read_id and broadcast helpers.

    You can pass either a pyserial Serial object or a serial port string.
    """
    # --- Protocol constants ---
    HDR = 0xE9
    ESC = 0xE8
    ADDR_MIN, ADDR_MAX = 1, 30
    ADDR_BROADCAST = 31  # write-only, no responses

    def __init__(
        self,
        port: serial.Serial | str,
        address: int = 1,
        *,
        baudrate: int = 9600,
        bytesize: int = 8,
        parity: str = "E",
        stopbits: int = 1,
        timeout: float = 0.5,
        inter_frame_delay: float = 0.03,
    ):
        self.address = int(address)
        self._ser: Optional[serial.Serial] = None
        self._owned_serial = False
        self._inter_delay = float(inter_frame_delay)

        if isinstance(port, serial.Serial):
            # Use provided Serial instance (assumed already configured as 8-E-1).
            self._ser = port
            self._owned_serial = False
        else:
            # Create and open our own Serial.
            self._ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=getattr(serial, f"PARITY_{parity.upper()}"),
                stopbits=stopbits,
                timeout=timeout,
            )
            self._owned_serial = True

    # ---------------- Public API ----------------

    def set_address(self, address: int) -> None:
        """Change the *target* address this instance talks to (1..30 or 31=broadcast)."""
        self.address = int(address)
        time.sleep(0.01)

    def close(self) -> None:
        """Close the port if we own it."""
        self.stop(address=31) # Stops all pumps.
        
        if self._ser and self._owned_serial:
            try:
                self._ser.close()
            finally:
                self._ser = None

    # Context manager support
    def __enter__(self) -> "WX10Pump":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---- Commands ----

    def set_speed(
        self,
        rpm: float,
        *,
        run: bool = True,
        full_speed: bool = False,
        cw: bool = True,
        address: Optional[int] = None,
        broadcast: bool = False,
    ) -> None:
        """
        Set speed/direction/run flags.
        - rpm: float in [0..100], resolution 0.1 rpm (spec).
        - run: bit0 (1=run, 0=stop)
        - full_speed: bit1 (1=full-speed prime, 0=normal)
        - cw: direction bit0 (1=CW, 0=CCW)
        - broadcast=True sends to addr 31 (no response).
        """
        addr = self._resolve_addr(address, broadcast)
        rpm10 = max(0, min(int(round(rpm * 10)), 1000))  # clamp 0..100 rpm
        pdu = bytearray()
        pdu += b"WJ"  # 0x57 0x4A
        pdu += bytes([(rpm10 >> 8) & 0xFF, rpm10 & 0xFF])  # MSB first
        ctrl = (0x01 if run else 0x00) | (0x02 if full_speed else 0x00)
        pdu.append(ctrl)
        pdu.append(0x01 if cw else 0x00)  # direction bit0
        self._send_frame(addr, bytes(pdu), expect_response=False if addr == self.ADDR_BROADCAST else True)

    def stop(self, *, address: Optional[int] = None) -> None:
        """Stop pump (run=0, rpm=0)."""
        self.set_speed(0.0, run=False, address=address)

    def prime(self, *, cw: bool = True, address: Optional[int] = None, broadcast: bool = False) -> None:
        """
        Convenience: run at 'full speed' prime mode (bit1=1).
        Note: speed value is ignored by the drive in full-speed mode.
        """
        self.set_speed(0.0, run=True, full_speed=True, cw=cw, address=address, broadcast=broadcast)

    def get_state(self, *, address: Optional[int] = None) -> Dict[str, Any]:
        """
        Read current running state (RJ). Only valid for unicast addresses (1..30).
        Returns: dict(speed_rpm, running, full_speed, direction)
        """
        addr = self._resolve_addr(address, broadcast=False)
        if addr == self.ADDR_BROADCAST:
            raise ValueError("RJ read is unicast-only (addresses 1..30).")
        pdu = b"RJ"
        frame = self._send_frame(addr, pdu, expect_response=True)
        # Parse logical (already unescaped) bytes: [HDR][addr][len][pdu...][fcs]
        # pdu should be: 'RJ' + speed(2) + direction(1) + control(1) per spec.
        body = frame[1:]  # addr..fcs
        _addr, ln = body[0], body[1]
        p = body[2 : 2 + ln]
        if len(p) < 2 + 2 + 1 + 1 or p[0:2] != b"RJ":
            raise ValueError(f"Unexpected RJ payload: {p.hex()}")

        speed_01 = (p[2] << 8) | p[3]
        direction_b = p[4]
        control_b = p[5]

        speed_rpm = speed_01 / 10.0
        running = bool(control_b & 0x01)
        full_speed = bool(control_b & 0x02)
        direction = "CW" if (direction_b & 0x01) else "CCW"

        return {
            "speed_rpm": speed_rpm,
            "running": running,
            "full_speed": full_speed,
            "direction": direction,
            "raw": bytes(p),
        }

    def read_id(self, *, address: Optional[int] = None) -> int:
        """
        Read device address using RID (unicast only).
        Returns the address reported by the unit (1..30).
        """
        addr = self._resolve_addr(address, broadcast=False)
        if addr == self.ADDR_BROADCAST:
            raise ValueError("RID read is unicast-only (addresses 1..30).")
        pdu = b"RID"
        frame = self._send_frame(addr, pdu, expect_response=True)
        body = frame[1:]
        _addr, ln = body[0], body[1]
        p = body[2 : 2 + ln]
        if len(p) < 3 or p[:3] != b"RID":
            raise ValueError(f"Unexpected RID payload: {p.hex()}")
        # Many firmwares echo the actual address after 'RID' (check yours).
        # If not present, return the unicast addr we queried.
        reported = p[3] if len(p) >= 4 else _addr
        return int(reported & 0xFF)
    
    # ---- GPIO control - Valve and Fans ----
    def control_io(self,
                which="valve1",          # str/int, list/tuple of these, or dict {which: action}
                action="on",             # "on" | "off" | "blink" (used if 'which' is not a dict)
                period_s=5.0,
                cycles=None,             # None=infinite for blink
                active_low=False,
                chip_path="/dev/gpiochip4"):
        """
        Control one or more outputs simultaneously.
        which:
            - "valve1" | "valve2" | "fans" | <int BCM pin>
            - list/tuple of the above (same 'action' for all)
            - dict mapping each target -> action ("on"/"off"/"blink")
        Examples:
        control_io(["valve1","valve2"], "on")
        control_io({"valve1":"on", "fans":"off"})
        control_io({"valve1":"blink", "valve2":"on"}, period_s=2, cycles=5)
        """
        import time, gpiod
        from gpiod.line import Direction, Value

        alias = {"valve1": 23, "valve2": 24, "fans": 25}

        def to_pin(x):
            if isinstance(x, int):
                return x
            if x in alias:
                return alias[x]
            raise ValueError(f'Unknown target {x!r}; use "valve1","valve2","fans" or BCM pin int')

        # Normalize inputs -> dict {pin: action}
        if isinstance(which, dict):
            plan = {to_pin(k): v.lower() for k, v in which.items()}
        else:
            if isinstance(which, (list, tuple, set)):
                pins = [to_pin(w) for w in which]
            else:
                pins = [to_pin(which)]
            plan = {p: action.lower() for p in pins}

        valid = {"on", "off", "blink"}
        for a in plan.values():
            if a not in valid:
                raise ValueError(f'action must be one of {valid}; got {a!r}')

        pins_all = set(plan.keys())
        on  = Value.INACTIVE if active_low else Value.ACTIVE
        off = Value.ACTIVE   if active_low else Value.INACTIVE

        with gpiod.Chip(chip_path) as chip:
            # Request ONLY the pins we intend to touch (so others keep their state)
            cfg = gpiod.LineSettings(direction=Direction.OUTPUT, output_value=off)
            req = chip.request_lines(consumer="WX10Pump", config={p: cfg for p in pins_all})

            try:
                # Set static outputs immediately
                for p, act in plan.items():
                    if act == "on":
                        req.set_value(p, on)
                    elif act == "off":
                        req.set_value(p, off)

                # If no blinking requested, we’re done
                if all(act != "blink" for act in plan.values()):
                    return

                # Blinking pins
                blink_pins = [p for p, act in plan.items() if act == "blink"]
                count = 0
                while True:
                    for p in blink_pins:
                        req.set_value(p, on)
                    time.sleep(period_s)
                    for p in blink_pins:
                        req.set_value(p, off)
                    time.sleep(period_s)

                    if cycles is not None:
                        count += 1
                        if count >= cycles:
                            break
            finally:
                req.release()
                time.sleep(0.1)  # small settle

    #---- I2C Flow Sensor Reading ----
    def print_flow(self, which="flow1", seconds=5.0):
        """
        Read SLF3C-1300F via I2C mux and print flow at 10 Hz for `seconds`.
        Hardcoded: i2c_bus=1, mux_addr=0x70, sensor_addr=0x08, flow1->ch1, flow2->ch2.
        """
        import time
        from smbus2 import SMBus, i2c_msg

        # Hardcoded constants
        I2C_BUS      = 1
        MUX_ADDR     = 0x70
        SENSOR_ADDR  = 0x08
        CHANNEL_MAP  = {"flow1": 0, "flow2": 1}
        SAMPLE_PERIOD = 0.1  # 10 Hz

        # Sensor commands (water calibration)
        CMD_START_WATER = 0x3608
        CMD_STOP        = 0x3FF9

        if which not in CHANNEL_MAP:
            raise ValueError('which must be "flow1" or "flow2"')

        channel = CHANNEL_MAP[which]

        def mux_select(bus, ch):
            bus.write_byte(MUX_ADDR, 1 << ch)
            time.sleep(0.001)

        def write_cmd16(bus, addr, cmd16):
            bus.write_i2c_block_data(addr, (cmd16 >> 8) & 0xFF, [cmd16 & 0xFF])

        def crc8_sensirion(two_bytes):
            crc = 0xFF
            for b in two_bytes:
                crc ^= b
                for _ in range(8):
                    crc = ((crc << 1) ^ 0x31) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
            return crc

        def read_flow_temp(bus):
            rx = i2c_msg.read(SENSOR_ADDR, 9)  # flow(2)+CRC, temp(2)+CRC, flags(2)+CRC
            bus.i2c_rdwr(rx)
            data = list(rx)

            vals = []
            for i in range(0, 9, 3):
                w = data[i:i+2]
                c = data[i+2]
                if crc8_sensirion(w) != c:
                    raise ValueError("CRC mismatch")
                v = (w[0] << 8) | w[1]
                if v & 0x8000:
                    v -= 0x10000
                vals.append(v)

            flow_raw, temp_raw, flags = vals
            flow_ml_min = flow_raw / 500.0
            temp_c      = temp_raw  / 200.0
            return flow_ml_min, temp_c, flags

        end_t = time.time() + float(seconds)

        with SMBus(I2C_BUS) as bus:
            # select channel and start continuous measurement
            mux_select(bus, channel)
            write_cmd16(bus, SENSOR_ADDR, CMD_START_WATER)
            time.sleep(0.02)  # first data ready

            try:
                while time.time() < end_t:
                    try:
                        flow, temp, flags = read_flow_temp(bus)
                        print(f"[{which}] {flow:.3f} ml/min, {temp:.2f} °C, flags=0x{flags:04X}")
                    except Exception as e:
                        print(f"[{which}] read error: {e}")
                    time.sleep(SAMPLE_PERIOD)
            finally:
                # stop and deselect
                try: write_cmd16(bus, SENSOR_ADDR, CMD_STOP)
                except Exception: pass
                try: bus.write_byte(MUX_ADDR, 0x00)
                except Exception: pass



    # ---------------- Internals ----------------

    def _resolve_addr(self, address: Optional[int], broadcast: bool) -> int:
        if broadcast:
            return self.ADDR_BROADCAST
        if address is None:
            return self.address
        return int(address)

    @staticmethod
    def _fcs(addr: int, pdu: bytes) -> int:
        ln = len(pdu) & 0xFF
        f = addr ^ ln
        for b in pdu:
            f ^= b
        return f & 0xFF

    @staticmethod
    def _escape(data: bytes) -> bytes:
        out = bytearray()
        for b in data:
            if b == WX10Pump.ESC:          # 0xE8 -> E8 00
                out += b"\xE8\x00"
            elif b == WX10Pump.HDR:        # 0xE9 -> E8 01
                out += b"\xE8\x01"
            else:
                out.append(b)
        return bytes(out)

    @staticmethod
    def _unescape(data: bytes) -> bytes:
        out = bytearray()
        i = 0
        L = len(data)
        while i < L:
            b = data[i]
            if b == WX10Pump.ESC and i + 1 < L:
                nxt = data[i + 1]
                if nxt == 0x00:
                    out.append(WX10Pump.ESC)
                    i += 2
                    continue
                if nxt == 0x01:
                    out.append(WX10Pump.HDR)
                    i += 2
                    continue
            out.append(b)
            i += 1
        return bytes(out)

    def _build_frame(self, addr: int, pdu: bytes) -> bytes:
        """Build a TX frame with escaping applied after header."""
        ln = len(pdu) & 0xFF
        fcs = self._fcs(addr, pdu)
        body = bytes([addr & 0xFF, ln]) + pdu + bytes([fcs])
        return bytes([self.HDR]) + self._escape(body)

    def _send_frame(self, addr: int, pdu: bytes, *, expect_response: bool) -> Optional[bytes]:
        """
        Send one command. If expect_response=True, return the **logical** frame bytes:
        [HDR][addr][len][pdu...][fcs] (i.e., unescaped already for convenience).
        """
        if not self._ser:
            raise RuntimeError("Serial port is closed")

        # TX
        frame = self._build_frame(addr, pdu)
        self._ser.reset_input_buffer()
        self._ser.write(frame)
        self._ser.flush()
        time.sleep(self._inter_delay)

        if not expect_response or addr == self.ADDR_BROADCAST:
            return None

        # RX: find HDR, then read raw bytes while incrementally unescaping
        start_t = time.time()

        # 1) seek header 0xE9
        while True:
            b = self._ser.read(1)
            if b == b"":
                if time.time() - start_t > self._ser.timeout:
                    raise TimeoutError("Timeout waiting for header")
                continue
            if b[0] == self.HDR:
                break

        # 2) read until we have logical addr+len available
        raw = bytearray()
        while True:
            raw += self._ser.read(1) or b""
            un = self._unescape(bytes(raw))
            if len(un) >= 2:
                rx_addr, ln = un[0], un[1]
                break

        # 3) read until we have logical body of size (addr,len,pdu(ln),fcs) => ln+3 bytes
        need_logical = ln + 3
        while len(un) < need_logical:
            raw += self._ser.read(1) or b""
            un = self._unescape(bytes(raw))

        # We now have full logical body (addr..fcs). Rebuild logical frame with header for parsing symmetry.
        logical = bytes([self.HDR]) + un

        # 4) FCS check
        body = logical[1:]  # addr..fcs
        r_addr, r_len = body[0], body[1]
        r_pdu = body[2 : 2 + r_len]
        r_fcs = body[2 + r_len]
        calc = self._fcs(r_addr, r_pdu)
        if r_fcs != calc:
            raise ValueError(f"FCS mismatch (got 0x{r_fcs:02X}, want 0x{calc:02X})")

        return logical

