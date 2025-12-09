# PFAS_CTRL/drivers/gpio_io.py
from __future__ import annotations
from typing import Iterable, Mapping, Union
import time
import gpiod
from gpiod.line import Direction, Value

NameOrPin = Union[str, int]

class GPIOCtrl:
    """
    Minimal controller for 3 outputs on gpiochip4:
      - valve1 -> BCM 23
      - valve2 -> BCM 24
      - fans   -> BCM 25

    Keeps lines requested for the lifetime of this object so states persist
    across calls. Default logic is active-high (active_low=False).
    """

    DEFAULT_MAP = {"valve1": 23, "valve2": 24, "fans": 25}

    def __init__(
        self,
        *,
        chip_path: str = "/dev/gpiochip4",
        pin_map: Mapping[str, int] = None,
        active_low: bool = False,
        init_off: bool = True,
        logger=None,
    ):
        self.chip_path = chip_path
        self.pin_map = dict(pin_map or self.DEFAULT_MAP)
        self.active_low = bool(active_low)

        self._chip: gpiod.Chip | None = None
        self._req: gpiod.LineRequest | None = None

        off = Value.ACTIVE if active_low else Value.INACTIVE
        self._default_value = off if init_off else None  # None => leave to caller after open()
        self.logger = logger

    #  lifecycle 
    def open(self) -> "GPIOCtrl":
        self._chip = gpiod.Chip(self.chip_path)
        cfg = gpiod.LineSettings(direction=Direction.OUTPUT)
        # if init_off is set, define initial output
        if self._default_value is not None:
            cfg = gpiod.LineSettings(direction=Direction.OUTPUT, output_value=self._default_value)
        self._req = self._chip.request_lines(
            consumer="pfas-io",
            config={pin: cfg for pin in self.pin_map.values()}
        )
        return self

    def close(self) -> None:
        if self._req:
            try:
                self._req.release()
            finally:
                self._req = None
        if self._chip:
            try:
                self._chip.close()
            finally:
                self._chip = None

    def __enter__(self) -> "GPIOCtrl":
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    #  helpers 
    def _to_pin(self, which: NameOrPin) -> int:
        if isinstance(which, int):
            return which
        if which in self.pin_map:
            return self.pin_map[which]
        raise ValueError(f'Unknown target {which!r}; use one of {list(self.pin_map)} or a BCM int')

    @property
    def _ON(self) -> Value:
        return Value.INACTIVE if self.active_low else Value.ACTIVE

    @property
    def _OFF(self) -> Value:
        return Value.ACTIVE if self.active_low else Value.INACTIVE

    #  operations 
    def set(self, which: NameOrPin | Iterable[NameOrPin], state: bool) -> None:
        """Set one or many outputs True/False (on/off)."""
        assert self._req, "open() first"
        pins = [self._to_pin(which)] if not isinstance(which, (list, tuple, set)) else [self._to_pin(w) for w in which]
        val = self._ON if state else self._OFF
        for p in pins:
            self._req.set_value(p, val)
            if self.logger is not None:
                name = self._from_pin(p)  # or use 'valve1'/'valve2'
                if name in ("valve1", "valve2"):
                    ch = "valve_1" if name == "valve1" else "valve_2"
                    self.logger.log(ch, 1 if state else 0)

    def on(self, which: NameOrPin | Iterable[NameOrPin]) -> None:
        self.set(which, True)

    def off(self, which: NameOrPin | Iterable[NameOrPin]) -> None:
        self.set(which, False)

    def blink(self, which: NameOrPin, period_s: float = 1.0, cycles: int = 5) -> None:
        """Blink a single output (non-blocking alternatives can be added later)."""
        assert self._req, "open() first"
        pin = self._to_pin(which)
        for _ in range(int(cycles)):
            self._req.set_value(pin, self._ON)
            time.sleep(period_s)
            self._req.set_value(pin, self._OFF)
            time.sleep(period_s)

    def set_many(self, plan: Mapping[NameOrPin, bool]) -> None:
        """Set multiple named pins atomically-ish (loop). Example: {'valve1':True, 'fans':False}."""
        for k, v in plan.items():
            self.set(k, bool(v))
