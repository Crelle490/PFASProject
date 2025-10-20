from smbus2 import SMBus, i2c_msg
import time

I2C_BUS = 1
MUX_ADDR = 0x70         # your TCA/PCA9548A
SENSOR_ADDR = 0x08      # SLF3C-1300F

CMD_START_WATER = 0x3608 # start continuous (H2O)
CMD_STOP        = 0x3FF9 # stop measurement

def mux_select(bus, channel):
    assert 0 <= channel <= 7
    bus.write_byte(MUX_ADDR, 2 << channel)
    time.sleep(0.001)

def crc8_sensirion(two_bytes):
    crc = 0xFF
    for b in two_bytes:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc

def write_cmd16(bus, addr, cmd16):
    # 16-bit command, big endian
    bus.write_i2c_block_data(addr, (cmd16 >> 8) & 0xFF, [cmd16 & 0xFF])

def read_flow_temp(bus):
    # Read 9 bytes: flow(2)+CRC, temp(2)+CRC, flags(2)+CRC
    rx = i2c_msg.read(SENSOR_ADDR, 9)    # plain I2C read header (no command)
    bus.i2c_rdwr(rx)
    data = list(rx)

    words = []
    for i in range(0, 9, 3):
        w = data[i:i+2]
        c = data[i+2]
        if crc8_sensirion(w) != c:
            raise ValueError("CRC mismatch")
        val = (w[0] << 8) | w[1]
        if val & 0x8000:  # signed 16-bit
            val -= 0x10000
        words.append(val)

    flow_raw, temp_raw, flags = words
    flow_ml_min = flow_raw / 500.0   # scale factors
    temp_c      = temp_raw  / 200.0
    return flow_ml_min, temp_c, flags

with SMBus(I2C_BUS) as bus:
    mux_select(bus, channel=0)                 # pick your mux channel here
    write_cmd16(bus, SENSOR_ADDR, CMD_START_WATER)   # start continuous (H2O) :contentReference[oaicite:3]{index=3}
    time.sleep(0.02)                            # first data ready after ~12 ms :contentReference[oaicite:4]{index=4}

    for _ in range(20):
        f, t, flags = read_flow_temp(bus)
        print(f"{f:.3f} ml/min, {t:.2f} °C, flags=0x{flags:04X}")
        time.sleep(0.05)                        # 20 Hz read is fine (datasheet suggests 50–2000 Hz max) :contentReference[oaicite:5]{index=5}

    write_cmd16(bus, SENSOR_ADDR, CMD_STOP)    # stop when done :contentReference[oaicite:6]{index=6}
    bus.write_byte(MUX_ADDR, 0x00)             # (optional) deselect all mux channels
