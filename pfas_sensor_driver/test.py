from .sensor import FluorideIonSensorModbus

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