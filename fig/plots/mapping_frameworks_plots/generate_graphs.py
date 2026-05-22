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
# Wavemap CSV Columns
# ------------------------------------------------------------

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
    "wavemap": "WaveMap",
    "losos": "Volumetric/occupancy-based mapper",
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

def get_csv_columns(path: Path, framework: str) -> list[str]:
    """
    Selects the correct CSV column layout.

    Standard / losos / octomap CSV:
        32 columns, includes metric_type and ESDF/SDF fields.

    wavemap CSV:
        19 columns, does not include metric_type or ESDF/SDF fields.
    """
    framework_lower = framework.lower()

    if framework_lower == "wavemap":
        return WAVEMAP_CSV_COLUMNS

    # Fallback podle počtu sloupců, kdyby se framework jmenoval trochu jinak.
    first_row = pd.read_csv(path, header=None, nrows=1)
    column_count = first_row.shape[1]

    if column_count == len(WAVEMAP_CSV_COLUMNS):
        return WAVEMAP_CSV_COLUMNS

    if column_count == len(CSV_COLUMNS):
        return CSV_COLUMNS

    raise ValueError(
        f"{path.name}: unsupported CSV column count {column_count}. "
        f"Expected {len(CSV_COLUMNS)} or {len(WAVEMAP_CSV_COLUMNS)}."
    )

def read_metrics_csv(path: Path, framework: str, warmup_seconds: float = 0.0) -> pd.DataFrame:
    columns = get_csv_columns(path, framework)

    df = pd.read_csv(
        path,
        header=None,
        names=columns,
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

    available_frameworks = {
        framework.lower()
        for framework, df in framework_data.items()
        if not df.empty
    }

    detail_frameworks = []
    if metric_key == "insertion_time":
        if "losos" in available_frameworks:
            detail_frameworks.append("losos")
        if "wavemap" in available_frameworks:
            detail_frameworks.append("wavemap")

    has_detail_plots = len(detail_frameworks) > 0

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    if has_detail_plots:
        # Hlavní graf nahoře, detailní grafy dole
        fig = plt.figure(figsize=(6.8, 6.0))

        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[3.45, 1.05],
            hspace=0.22,
            wspace=0.22,
        )

        ax = fig.add_subplot(gs[0, :])

        detail_axes = {}

        if len(detail_frameworks) == 1:
            detail_axes[detail_frameworks[0]] = fig.add_subplot(gs[1, :])
        else:
            detail_axes[detail_frameworks[0]] = fig.add_subplot(gs[1, 0])
            detail_axes[detail_frameworks[1]] = fig.add_subplot(gs[1, 1])
    else:
        fig, ax = plt.subplots(figsize=(6.8, 4.9))
        detail_axes = {}

    plotted_series = {}

    # ------------------------------------------------------------
    # Main plot
    # ------------------------------------------------------------
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

        color = line.get_color()

        plotted_series[framework.lower()] = {
            "x": x,
            "y": y,
            "df": df,
            "label": label,
            "color": color,
            "min_col": min_col,
            "max_col": max_col,
            "scale": scale,
        }

        if metric["shade"] and min_col and max_col:
            if min_col in df.columns and max_col in df.columns:
                ymin = df[min_col] * scale
                ymax = df[max_col] * scale

                ax.fill_between(
                    x,
                    ymin,
                    ymax,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )

    # 0,0 opravdu v levém dolním rohu
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0, y=0)

    ax.set_title(metric["title"], fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(metric["ylabel"], fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.25)

    ax.legend(
        loc="upper left",
        fontsize=10,
        frameon=True,
    )

    # ------------------------------------------------------------
    # Detail plots under insertion time graph
    # ------------------------------------------------------------
    def draw_detail_axis(detail_ax, target_framework: str, title: str, y_max: float):
        if target_framework not in plotted_series:
            return

        series = plotted_series[target_framework]

        detail_ax.plot(
            series["x"],
            series["y"],
            linewidth=1.3,
            color=series["color"],
        )

        if metric["shade"] and series["min_col"] and series["max_col"]:
            df = series["df"]
            min_col = series["min_col"]
            max_col = series["max_col"]
            scale = series["scale"]

            if min_col in df.columns and max_col in df.columns:
                detail_ax.fill_between(
                    series["x"],
                    df[min_col] * scale,
                    df[max_col] * scale,
                    color=series["color"],
                    alpha=0.14,
                    linewidth=0,
                )

        detail_ax.set_xlim(left=0)
        detail_ax.set_ylim(0, y_max)
        detail_ax.margins(x=0)

        detail_ax.set_title(title, fontsize=9, pad=3)
        detail_ax.set_xlabel("Time (s)", fontsize=9)
        detail_ax.tick_params(axis="both", labelsize=8)
        detail_ax.grid(True, alpha=0.25)

    if metric_key == "insertion_time":
        # Losos detail: klasicky podle maxima lososa + offset
        if "losos" in plotted_series and "losos" in detail_axes:
            losos_y = plotted_series["losos"]["y"].dropna()
            losos_y = losos_y[losos_y >= 0]

            if not losos_y.empty:
                losos_max = losos_y.max()
                losos_offset = max(5.0, losos_max * 0.15)
                losos_y_max = losos_max + losos_offset

                draw_detail_axis(
                    detail_axes["losos"],
                    target_framework="losos",
                    title="V/O mapper detail",
                    y_max=losos_y_max,
                )

        # wavemap detail: pouze wavemap, y-limit podle hodnot po prvních N sekundách
        if "wavemap" in plotted_series and "wavemap" in detail_axes:
            wavemap_x = plotted_series["wavemap"]["x"]
            wavemap_y = plotted_series["wavemap"]["y"]

            # Ignorujeme počáteční spike při určování měřítka detail grafu.
            # Spike pořád zůstane vidět v hlavním grafu.
            wavemap_ignore_first_seconds = 20.0

            stable_mask = wavemap_x >= wavemap_ignore_first_seconds
            stable_y = wavemap_y[stable_mask].dropna()
            stable_y = stable_y[stable_y >= 0]

            if not stable_y.empty:
                wavemap_max = stable_y.max()
                wavemap_offset = max(5.0, wavemap_max * 0.10)
                wavemap_y_max = wavemap_max + wavemap_offset
            else:
                # fallback, kdyby po 20 s nebyla žádná data
                wavemap_y_clean = wavemap_y.dropna()
                wavemap_y_clean = wavemap_y_clean[wavemap_y_clean >= 0]

                if wavemap_y_clean.empty:
                    wavemap_y_max = 1.0
                else:
                    wavemap_y_max = wavemap_y_clean.max() * 1.1

            draw_detail_axis(
                detail_axes["wavemap"],
                target_framework="wavemap",
                title="WaveMap detail",
                y_max=wavemap_y_max,
            )

    if has_detail_plots:
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.94,
            bottom=0.08,
        )
    else:
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
            df = read_metrics_csv(
                csv_file,
                framework=framework,
                warmup_seconds=args.warmup_seconds,
            )
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