import argparse
from pathlib import Path

import pandas as pd


WAVEMAP_CSV_COLUMNS = [
    "unknown_0",
    "unknown_1",
    "unknown_2",
    "intersection_avg",
    "intersection_min",
    "intersection_max",
    "intersection_last",
    "intersection_count",
    "points_per_second",
    "avg_points_per_second",
    "total_points_processed",
    "total_callbacks",
    "memory_rss_mb",
    "memory_vms_mb",
    "cpu_percent",
    "cpu_min_percent",
    "cpu_max_percent",
    "cpu_avg_percent",
    "uptime_seconds",
]


def parse_filename(path: Path):
    stem = path.stem
    framework, world = stem.split("_", 1)
    return framework.lower(), world


def read_wavemap_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=WAVEMAP_CSV_COLUMNS,
    )

    numeric_columns = [
        "intersection_avg",
        "intersection_min",
        "intersection_max",
        "intersection_count",
        "cpu_percent",
        "cpu_min_percent",
        "cpu_max_percent",
        "cpu_avg_percent",
        "uptime_seconds",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stejný filtr nevalidního začátku jako u grafů
    valid = (
        (df["intersection_count"] > 0)
        & (df["intersection_avg"] > 0)
        & (df["intersection_max"] > 0)
        & (df["intersection_min"] >= 0)
        & (df["intersection_min"] < 1e100)
    )

    if valid.any():
        first_valid_index = valid.idxmax()
        df = df.loc[first_valid_index:].copy()
    else:
        df = df.iloc[0:0].copy()

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    wavemap_files = sorted(args.input_dir.glob("wavemap_*.csv"))

    if not wavemap_files:
        raise RuntimeError(f"No wavemap CSV files found in {args.input_dir}")

    overall_max = None

    for file in wavemap_files:
        framework, world = parse_filename(file)

        df = read_wavemap_csv(file)

        if df.empty:
            print(f"{world}: no valid data")
            continue

        max_cpu = df["cpu_max_percent"].max()
        avg_cpu = df["cpu_avg_percent"].mean()

        if overall_max is None or max_cpu > overall_max:
            overall_max = max_cpu

        print(f"{world}")
        print(f"  max CPU usage: {max_cpu:.2f} %")
        print(f"  avg CPU usage: {avg_cpu:.2f} %")
        print()

    if overall_max is not None:
        print("Overall")
        print(f"  max WaveMap CPU usage: {overall_max:.2f} %")


if __name__ == "__main__":
    main()