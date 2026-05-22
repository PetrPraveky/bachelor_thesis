import argparse
from pathlib import Path

import pandas as pd


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


def parse_filename(path: Path):
    """
    Example:
        octomap_desert_oasis_palms_2026-05-05_22-57-19_0.csv

    framework = octomap
    world     = desert_oasis_palms_2026-05-05_22-57-19_0
    """
    stem = path.stem
    framework, world = stem.split("_", 1)
    return framework.lower(), world


def read_intersection_csv(path: Path, warmup_seconds: float = 0.0) -> pd.DataFrame:
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
    ]

    for col in needed_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["uptime_seconds", "intersection_avg"])
    df = df[df["uptime_seconds"] >= warmup_seconds].copy()

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
        return df.iloc[0:0].copy()

    df["time_s"] = df["uptime_seconds"] - df["uptime_seconds"].iloc[0]

    return df


def compare_world(world_name: str, octomap_df: pd.DataFrame, losos_df: pd.DataFrame):
    if octomap_df.empty or losos_df.empty:
        print(f"\n{world_name}")
        print("  Skipping: empty octomap or losos data.")
        return

    octo_mean = octomap_df["intersection_avg"].mean()
    losos_mean = losos_df["intersection_avg"].mean()

    octo_median = octomap_df["intersection_avg"].median()
    losos_median = losos_df["intersection_avg"].median()

    mean_speedup = octo_mean / losos_mean if losos_mean > 0 else float("nan")
    median_speedup = octo_median / losos_median if losos_median > 0 else float("nan")

    print(f"\nWorld: {world_name}")
    print(f"  OctoMap mean insertion: {octo_mean:.4f} ms")
    print(f"  Losos   mean insertion: {losos_mean:.4f} ms")
    print(f"  Mean speedup:           {mean_speedup:.2f}x")

    print(f"  OctoMap median insertion: {octo_median:.4f} ms")
    print(f"  Losos   median insertion: {losos_median:.4f} ms")
    print(f"  Median speedup:           {median_speedup:.2f}x")

    # ------------------------------------------------------------
    # Optional paired comparison by nearest time sample
    # ------------------------------------------------------------
    octo_pair = octomap_df[["time_s", "intersection_avg"]].copy()
    losos_pair = losos_df[["time_s", "intersection_avg"]].copy()

    octo_pair = octo_pair.sort_values("time_s")
    losos_pair = losos_pair.sort_values("time_s")

    paired = pd.merge_asof(
        octo_pair,
        losos_pair,
        on="time_s",
        direction="nearest",
        tolerance=0.5,
        suffixes=("_octomap", "_losos"),
    )

    paired = paired.dropna()

    if not paired.empty:
        losos_better = paired["intersection_avg_losos"] < paired["intersection_avg_octomap"]

        better_count = int(losos_better.sum())
        total_count = len(paired)
        better_percent = 100.0 * better_count / total_count

        paired["speedup"] = (
            paired["intersection_avg_octomap"] / paired["intersection_avg_losos"]
        )

        avg_paired_speedup = paired["speedup"].mean()
        median_paired_speedup = paired["speedup"].median()

        print(f"  Paired samples:          {total_count}")
        print(f"  Losos better samples:    {better_count}/{total_count} ({better_percent:.1f}%)")
        print(f"  Paired mean speedup:     {avg_paired_speedup:.2f}x")
        print(f"  Paired median speedup:   {median_paired_speedup:.2f}x")
    else:
        print("  Paired samples:          none, time samples did not match closely enough")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing CSV files.",
    )

    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help="Optional manual time cut from beginning of each CSV.",
    )

    args = parser.parse_args()

    csv_files = sorted(args.input_dir.glob("*.csv"))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {args.input_dir}")

    grouped = {}

    for csv_file in csv_files:
        try:
            framework, world = parse_filename(csv_file)
        except ValueError:
            print(f"Skipping {csv_file.name}: filename does not contain '_'")
            continue

        if framework not in ["octomap", "losos"]:
            continue

        try:
            df = read_intersection_csv(
                csv_file,
                warmup_seconds=args.warmup_seconds,
            )
        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")
            continue

        grouped.setdefault(world, {})[framework] = df

    for world_name, data in sorted(grouped.items()):
        if "octomap" not in data or "losos" not in data:
            print(f"\nWorld: {world_name}")
            print("  Skipping: missing octomap or losos CSV.")
            continue

        compare_world(
            world_name=world_name,
            octomap_df=data["octomap"],
            losos_df=data["losos"],
        )


if __name__ == "__main__":
    main()