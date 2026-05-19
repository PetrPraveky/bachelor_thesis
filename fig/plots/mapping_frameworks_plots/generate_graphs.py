import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CSV header
# ------------------------------------------------------------

CSV_COLUMNS = [
    "unknown_0",
    "unknown_1",
    "unknown_2",
    "metric_type",
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
    "esdf_avg_update",
    "esdf_min_update",
    "esdf_max_update",
    "esdf_last_update",
    "esdf_update_freq",
    "esdf_avg_update_freq",
    "esdf_total_voxels",
    "esdf_total_obstacles_added",
    "esdf_total_obstacles_removed",
    "sdf_total_updates",
    "esdf_avg_voxel_per_update",
    "esdf_obstacles_per_update",
    "uptime_seconds",
]

# ------------------------------------------------------------
# Metric configuration
# ------------------------------------------------------------

METRICS = {
    "insertion_time": {
        "title": "Insertion Time",
        "ylabel": "Insertion Time (ms)",
        "avg": "intersection_avg",
        "min": "intersection_min",
        "max": "intersection_max",
        "scale": 1.0,
        "shade": True,
    },
    "memory_usage": {
        "title": "Memory Usage",
        "ylabel": "Memory Usage (GB)",
        "avg": "memory_rss_mb",
        "min": None,
        "max": None,
        "scale": 1.0 / 1024.0,  # MB -> GB
        "shade": False,
    },
    "cpu_usage": {
        "title": "CPU Usage",
        "ylabel": "CPU Usage (%)",
        "avg": "cpu_avg_percent",
        "min": "cpu_min_percent",
        "max": "cpu_max_percent",
        "scale": 1.0,
        "shade": True,
    },
}


FRAMEWORK_LABELS = {
    "octomap": "OctoMap",

}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def parse_filename(path: Path):
    """
    Example:
        octomap_desert_oasis_palms_2026-05-05_22-57-19_0.csv

    framework = octomap
    world     = desert_oasis_palms_2026-05-05_22-57-19_0
    """
    stem = path.stem
    framework, world = stem.split("_", 1)
    return framework, world


def sanitize_filename(name: str) -> str:
    return "".join(
        c if c.isalnum() or c in "_-." else "_"
        for c in name
    )


def read_metrics_csv(path: Path, warmup_seconds: float = 0.0) -> pd.DataFrame:
    # CSV nemá header, proto header=None.
    # Názvy sloupců si přiřadíme ručně podle indexů.
    df = pd.read_csv(
        path,
        header=None,
        names=CSV_COLUMNS,
    )

    needed_columns = [
        "uptime_seconds",
        "intersection_avg",
        "intersection_min",
        "intersection_max",
        "intersection_count",
        "memory_rss_mb",
        "cpu_avg_percent",
        "cpu_min_percent",
        "cpu_max_percent",
    ]

    for col in needed_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["uptime_seconds"])
    df = df[df["uptime_seconds"] >= warmup_seconds]

    # Odstranění nevalidního začátku.
    # Podle toho, co popisuješ:
    # intersection_avg = 0
    # intersection_max = 0
    # intersection_count = 0
    # intersection_min = obrovská e308 hodnota
    valid_intersection = (
        (df["intersection_count"] > 0)
        & (df["intersection_avg"] > 0)
        & (df["intersection_max"] > 0)
        & (df["intersection_min"] >= 0)
        & (df["intersection_min"] < 1e100)
    )

    if valid_intersection.any():
        first_valid_index = valid_intersection.idxmax()
        df = df.loc[first_valid_index:].copy()
    else:
        print(f"Warning: {path.name} has no valid intersection rows.")
        df = df.iloc[0:0].copy()

    if not df.empty:
        df["time_s"] = df["uptime_seconds"] - df["uptime_seconds"].iloc[0]

    return df


def plot_single_metric(
    world_name: str,
    metric_key: str,
    framework_data: dict[str, pd.DataFrame],
    output_dir: Path,
):
    metric = METRICS[metric_key]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for framework, df in sorted(framework_data.items()):
        if df.empty:
            continue

        avg_col = metric["avg"]
        min_col = metric["min"]
        max_col = metric["max"]
        scale = metric["scale"]

        if avg_col not in df.columns:
            print(f"Skipping {framework} / {world_name}: missing {avg_col}")
            continue

        label = FRAMEWORK_LABELS.get(framework.lower(), framework)

        x = df["time_s"]
        y = df[avg_col] * scale

        line, = ax.plot(
            x,
            y,
            linewidth=1.4,
            label=label,
        )

        if metric["shade"] and min_col and max_col:
            if min_col in df.columns and max_col in df.columns:
                ymin = df[min_col] * scale
                ymax = df[max_col] * scale

                ax.fill_between(
                    x,
                    ymin,
                    ymax,
                    color=line.get_color(),
                    alpha=0.14,
                    linewidth=0,
                )
    # 0 Offset
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0, y=0)

    ax.set_title(metric["title"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(metric["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    safe_world = sanitize_filename(world_name)
    output_path = output_dir / f"{safe_world}_{metric_key}.png"

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plots will be saved.",
    )

    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help="Optional manual time cut from beginning of each CSV.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.input_dir.glob("*.csv"))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {args.input_dir}")

    # grouped[world][framework] = dataframe
    grouped: dict[str, dict[str, pd.DataFrame]] = {}

    for csv_file in csv_files:
        try:
            framework, world = parse_filename(csv_file)
        except ValueError:
            print(f"Skipping {csv_file.name}: filename does not contain '_'")
            continue

        try:
            df = read_metrics_csv(csv_file, warmup_seconds=args.warmup_seconds)
        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")
            continue

        grouped.setdefault(world, {})[framework] = df

    for world_name, framework_data in grouped.items():
        for metric_key in METRICS.keys():
            plot_single_metric(
                world_name=world_name,
                metric_key=metric_key,
                framework_data=framework_data,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()