import os
from pathlib import Path
import typer, yaml
from pfas_ctrl.drivers.pump_wx10 import WX10Pump

app = typer.Typer(add_completion=False)

def resolve_config_path(user_path: str | None = None) -> Path:
    if user_path:
        return Path(user_path).expanduser().resolve()
    env = os.getenv("CTRL_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    # default: ../config/ctrl_config.yaml relative to this file
    return (Path(__file__).resolve().parent.parent / "config" / "ctrl_config.yaml")

def load_cfg(path: str | None = None) -> dict:
    cfg_path = resolve_config_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())

def open_pump(cfg) -> WX10Pump:
    p = cfg["pump"]
    return WX10Pump(
        port=p["port"],
        address=int(p.get("address", 1)),
        baudrate=int(p.get("baudrate", 9600)),
        parity=p.get("parity", "EVEN"),
        stopbits=int(p.get("stopbits", 1)),
        timeout=float(p.get("timeout", 0.8)),
    )

@app.command()
def speed(rpm: float, cw: bool = True, full_speed: bool = False, config: str | None = None):
    """Set speed in RPM (0..100)."""
    cfg = load_cfg(config)
    pump = open_pump(cfg)
    try:
        pump.set_speed(rpm, run=True, cw=cw, full_speed=full_speed)
        typer.echo(f"Set {rpm:.1f} RPM (cw={cw}, full={full_speed})")
    finally:
        pump.close()

@app.command()
def stop(config: str | None = None):
    """Stop the pump."""
    cfg = load_cfg(config)
    pump = open_pump(cfg)
    try:
        pump.stop()
        typer.echo("Stopped.")
    finally:
        pump.close()

@app.command()
def prime(cw: bool = True, config: str | None = None):
    """Prime mode (full speed)."""
    cfg = load_cfg(config)
    pump = open_pump(cfg)
    try:
        pump.prime(cw=cw)
        typer.echo(f"Prime (cw={cw})")
    finally:
        pump.close()

@app.command()
def state(config: str | None = None):
    """Read current pump state."""
    cfg = load_cfg(config)
    pump = open_pump(cfg)
    try:
        s = pump.get_state()
        typer.echo(s)
    finally:
        pump.close()
