from PH_sensor import SerialConfig, PHAnalyzer

cfg = SerialConfig(port="/dev/ttyUSB0", baudrate=9600, parity="N", stopbits=1, timeout=1.0)
sensor = PHAnalyzer(cfg, device_id=2)  # unit = “网络节点” in the instrument menu

sensor.open()
try:
    data = sensor.read_all()
    print(data)
finally:
    sensor.close()
