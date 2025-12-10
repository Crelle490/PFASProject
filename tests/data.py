#!/usr/bin/env python3
# PFASProject/tests/data.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.errors import EmptyDataError, ParserError

EXPECTED_COLS = ["sample_idx", "timestamp_iso", "elapsed_s", "fluoride_mgL", "fluoride_M"]

def find_project_root(start: Path) -> Path:
    parts = start.resolve().parts
    if "PFASProject" in parts:
        return Path(*parts[: parts.index("PFASProject") + 1])
    return start.resolve()

def try_read_csv(p: Path) -> pd.DataFrame | None:
    """Try to read a CSV robustly. Returns DataFrame or None."""
    try:
        # attempt with automatic delimiter inference
        df = pd.read_csv(p, engine="python")
    except EmptyDataError:
        print(f"[SKIP] Empty file: {p.name}")
        return None
    except ParserError as e:
        print(f"[WARN] ParserError on {p.name}: {e}. Retrying with sep=',' and then ';'")
        for sep in [",", ";", r"\s+"]:
            try:
                df = pd.read_csv(p, engine="python", sep=sep)
                break
            except Exception:
                df = None
        if df is None:
            return None
    except Exception as e:
        print(f"[WARN] Failed reading {p.name}: {e}")
        return None

    # If no columns (weird), bail
    if df is None or df.shape[1] == 0:
        print(f"[SKIP] No of columns in: {p.name}")
        return None

    # If header missing, try to assign expected names when counts match
    if not set(EXPECTED_COLS).issubset(df.columns):
        if df.shape[1] == len(EXPECTED_COLS):
            print(f"[INFO] Assigning expected column names to: {p.name}")
            df.columns = EXPECTED_COLS
        else:
            print(f"[WARN] Unexpected columns in {p.name}: {list(df.columns)}")
            # try to rename by best guess
            # do not hard fail—just continue and we’ll check presence below

    return df

def load_latest_valid_df(data_folder: Path) -> tuple[pd.DataFrame, Path] | tuple[None, None]:
    files = sorted(data_folder.glob("fluoride_log_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(f"[ERROR] No fluoride_log_*.csv files in {data_folder}")
        return None, None

    for f in files:
        size = f.stat().st_size
        print(f"[INFO] Considering {f.name} (size {size} bytes)")
        if size < 32:
            print(f"[SKIP] Too small to contain data: {f.name}")
            continue
        df = try_read_csv(f)
        if df is None:
            continue

        # ensure required columns exist (at least mg/L)
        if "fluoride_mgL" not in df.columns:
            print(f"[SKIP] Missing 'fluoride_mgL' in {f.name}")
            continue

        # coerce numerics
        for col in ["elapsed_s", "fluoride_mgL", "fluoride_M"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # drop rows with no mg/L
        df = df.dropna(subset=["fluoride_mgL"])
        if df.empty:
            print(f"[SKIP] No valid fluoride rows in {f.name}")
            continue

        return df, f

    print("[ERROR] Could not find a non-empty, parseable CSV with valid fluoride data.")
    return None, None

def main():
    # auto-detect PFASProject/data regardless of script location
    this_file = Path(__file__).resolve()
    project_root = find_project_root(this_file)
    data_folder = project_root / "data"

    if not data_folder.exists():
        print(f"[ERROR] Folder not found: {data_folder}")
        return

    df, csv_path = load_latest_valid_df(data_folder)
    if df is None:
        return

    t = df["elapsed_s"].to_numpy() if "elapsed_s" in df.columns else None
    f_mgL = df["fluoride_mgL"].to_numpy()
    f_M = df["fluoride_M"].to_numpy() if "fluoride_M" in df.columns else None

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = t if t is not None else range(len(f_mgL))
    ax1.plot(x, f_mgL, lw=2, label="Fluoride [mg/L]")
    ax1.set_xlabel("Elapsed time [s]" if t is not None else "Sample index")
    ax1.set_ylabel("Fluoride [mg/L]")
    ax1.grid(True)

    if f_M is not None:
        ax2 = ax1.twinx()
        ax2.plot(x, f_M, lw=1.8, ls="--", label="Fluoride [M]")
        ax2.set_ylabel("Fluoride [M]")

    start = df["timestamp_iso"].iloc[0] if "timestamp_iso" in df.columns else ""
    end = df["timestamp_iso"].iloc[-1] if "timestamp_iso" in df.columns else ""
    plt.title(f"{csv_path.name}\n{start} → {end}")

    fig.tight_layout()
    plt.show()
    print(f"[OK] Plotted {csv_path.name}")

if __name__ == "__main__":
    main()
