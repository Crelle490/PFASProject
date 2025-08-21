# live_plotter_headless.py
from __future__ import annotations
import csv, time, struct
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from Flouride_Sensor_driver.flouride_sensor import SerialConfig, FluorideAnalyzer

# ---- Config ----
PORT = "/dev/ttyUSB0"
UNIT = 1
BAUD = 9600
PARITY = "N"
STOPBITS = 1
TIMEOUT_S = 1.5
BIG_ENDIAN_WORDS = True
SAMPLE_PERIOD = 60            # 1 minute
CSV_PATH = Path("live_log.csv")
BASE_DIR = Path(__file__).resolve().parent.parent
PNG_PATH   = BASE_DIR / "live_plot.png"
INDEX_HTML = BASE_DIR / "index.html"
INDEX_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Fluoride Analyzer — Live Plot</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font-family:system-ui,sans-serif;margin:1rem}header{display:flex;align-items:baseline;gap:1rem}img{max-width:100%;height:auto;border:1px solid #ddd}.muted{color:#666;font-size:.9rem}</style></head><body><header><h1>Fluoride Analyzer — Live Plot</h1><div class="muted">Auto-refreshes every 60s</div></header><p class="muted">If the image looks stale, caching is bypassed via a timestamp.</p><img id="plot" alt="Live Plot" src="live_plot.png"><p class="muted" id="status">Loading…</p><script>const img=document.getElementById('plot');const status=document.getElementById('status');function refreshPlot(){const ts=Date.now();img.src=`live_plot.png?t=${ts}`;status.textContent=`Last refresh: ${new Date().toLocaleString()}`;}refreshPlot();setInterval(refreshPlot,60000);</script></body></html>"""


ROLLING_POINTS = 1440         # keep last 24h at 1-min sampling
# ----------------

def align_to_next(period_s: int) -> float:
    now = time.time()
    return period_s - (now % period_s)

def ensure_csv_header(path: Path):
    if not path.exists():
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_iso", "concentration_mgL", "temperature_c", "electrode_mV", "current1_mA", "current2_mA"])

def append_csv(path: Path, row):
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(row)

def render_plot(t_vals, conc_vals, temp_vals):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.set_title("Fluoride Analyzer — Live (headless)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Concentration (mg/L)")
    ax2.set_ylabel("Temperature (°C)")

    ax1.plot(t_vals, conc_vals, label="F⁻ (mg/L)")
    ax2.plot(t_vals, temp_vals, linestyle="--", label="Temp (°C)")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)

def ensure_index_html():
    if not INDEX_HTML.exists():
        INDEX_HTML.write_text(INDEX_TEMPLATE, encoding="utf-8")

def main():
    cfg = SerialConfig(port=PORT, baudrate=BAUD, parity=PARITY, stopbits=STOPBITS, timeout=TIMEOUT_S)
    sensor = FluorideAnalyzer(cfg, device_id=UNIT, big_endian_words=BIG_ENDIAN_WORDS)

    for attempt in range(3):
        try:
            sensor.open()
            break
        except Exception as e:
            if attempt == 2: raise
            time.sleep(1.0)

    ensure_index_html()

    ensure_csv_header(CSV_PATH)

    t_vals, conc_vals, temp_vals = [], [], []
    print("Headless live plotter running — writing live_plot.png every minute. Ctrl+C to stop.")
    time.sleep(align_to_next(SAMPLE_PERIOD))

    try:
        while True:
            ts = datetime.now()
            try:
                d = sensor.read_all()
                conc = float(d["concentration_mgL"])
                temp = float(d["temperature_c"])
                mv   = float(d["electrode_mV"])
                i1   = float(d["current1_mA"])
                i2   = float(d["current2_mA"])

                t_vals.append(ts); conc_vals.append(conc); temp_vals.append(temp)

                # keep a rolling window (optional)
                if len(t_vals) > ROLLING_POINTS:
                    t_vals = t_vals[-ROLLING_POINTS:]
                    conc_vals = conc_vals[-ROLLING_POINTS:]
                    temp_vals = temp_vals[-ROLLING_POINTS:]

                # save CSV row
                append_csv(CSV_PATH, [ts.isoformat(), f"{conc:.6g}", f"{temp:.6g}", f"{mv:.6g}", f"{i1:.6g}", f"{i2:.6g}"])

                # render PNG
                render_plot(t_vals, conc_vals, temp_vals)

                print(f"[{ts:%H:%M:%S}] F⁻={conc:.4f} mg/L  T={temp:.2f} °C  (updated {PNG_PATH})")

            except Exception as e:
                print(f"[{ts:%H:%M:%S}] Read error: {e}")

            time.sleep(align_to_next(SAMPLE_PERIOD))
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        sensor.close()

if __name__ == "__main__":
    main()
