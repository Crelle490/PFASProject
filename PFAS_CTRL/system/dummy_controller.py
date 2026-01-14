import argparse
import atexit
import signal
import threading
import time
from pathlib import Path

import serial

# Allow running as a script when PFAS_CTRL isn't on PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from PFAS_CTRL.drivers.pump_wx10 import WX10Pump


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple motor control for pump address 1.",
    )
    parser.add_argument("--port", default="/dev/ttyUSB1", help="RS-485 serial port")
    parser.add_argument("--baudrate", type=int, default=9600, help="Serial baudrate")
    parser.add_argument("--timeout", type=float, default=0.5, help="Serial timeout (s)")
    parser.add_argument("--address", type=int, default=1, help="Pump address (default: 1)")
    parser.add_argument("--speed", type=float, default=50.0, help="Speed percent (0-100)")
    parser.add_argument("--duration", type=float, default=None, help="Run time in seconds")
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep running until the program exits (Ctrl+C to stop)",
    )
    parser.add_argument(
        "--ccw",
        action="store_true",
        help="Run counter-clockwise (default: clockwise)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.duration is None and not args.hold:
        parser.error("Provide --duration or --hold")
    if args.duration is not None and args.duration < 0:
        parser.error("--duration must be >= 0")

    stop_event = threading.Event()

    ser = serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=8,
        parity=serial.PARITY_EVEN,
        stopbits=1,
        timeout=args.timeout,
    )
    pump = WX10Pump(ser)

    def _stop():
        if stop_event.is_set():
            return
        stop_event.set()
        try:
            pump.stop(address=args.address)
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass

    atexit.register(_stop)

    def _handle_signal(_sig, _frame):
        _stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pump.set_speed(
        args.speed,
        run=True,
        cw=not args.ccw,
        address=args.address,
    )
    flow_ml_min = 0.11889 * float(args.speed) - 0.03
    print(f"Flow estimate (addr {args.address}): {flow_ml_min:.5f} mL/min")

    if args.hold:
        while not stop_event.is_set():
            time.sleep(0.2)
        return 0

    deadline = time.monotonic() + float(args.duration or 0.0)
    while not stop_event.is_set():
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)

    _stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
