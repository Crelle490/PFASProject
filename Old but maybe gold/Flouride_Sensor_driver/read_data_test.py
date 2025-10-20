from flouride_sensor import SerialConfig, FluorideAnalyzer

cfg = SerialConfig(port="/dev/ttyUSB1", baudrate=9600, parity="N", stopbits=1, timeout=1.0)
sensor = FluorideAnalyzer(cfg, device_id=1)  # unit = “网络节点” in the instrument menu

sensor.open()
try:
    data = sensor.read_all()
    print(data)
finally:
    sensor.close()
