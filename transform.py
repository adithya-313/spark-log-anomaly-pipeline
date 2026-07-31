r"""
transform.py
============
Phase 2 (Transform) of the Spark log-anomaly pipeline.

Reads Phase 1's output (extract_output.parquet) and produces windowed,
feature-engineered, labeled data, ready for Phase 3 (Load) to write out.

Pipeline:
  Clean   - drop rows tagged parse_ok=False (logging the count BEFORE the
            drop), cast timestamp/status_code/response_size to real types,
            and drop exact-duplicate rows (count reported even if zero).
  Window  - bucket clean rows into 5-minute windows whose boundaries EXACTLY
            match ground_truth_windows.csv (same 5-minute floor).
  Feature - per window (global, not per endpoint): total request count,
            error rate (share of 4xx/5xx status codes), average response
            size, and the p95 of response size.
  Label   - join the REAL ground-truth labels (is_anomalous, anomaly_type)
            from ground_truth_windows.csv onto each window. NO heuristic
            detection (e.g. std-dev threshold) is used -- this is synthetic
            data with known labels, so the real labels are used directly.

Output: transform_output.parquet (window_start, is_anomalous, anomaly_type,
        total_requests, error_rate, avg_response_size, p95_response_size).

ENV NOTE (required to run Spark on this laptop):
    $env:JAVA_HOME = "C:\Users\sinne\jdk17\jdk-17.0.20+8"
    $env:HADOOP_HOME = "C:\Users\sinne\hadoop-utils"          # needed for file writes
    $env:PATH = "C:\Users\sinne\hadoop-utils\bin;$env:PATH"   # hadoop.dll for native IO
    $env:SPARK_LOCAL_IP = "127.0.0.1"
    $env:PYSPARK_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"
    $env:PYSPARK_DRIVER_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"

Usage:
    python transform.py
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, max, min, percentile, to_timestamp, window, when

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_OUTPUT = os.path.join(SCRIPT_DIR, "extract_output.parquet")
GROUND_TRUTH = os.path.join(SCRIPT_DIR, "ground_truth_windows.csv")
TRANSFORM_OUTPUT = os.path.join(SCRIPT_DIR, "transform_output.parquet")

# The generator bucketed rows into 5-minute windows starting on the hour;
# F.window with the session timezone pinned to UTC floors wall-clock times to
# the exact same boundaries as ground_truth_windows.csv.
WINDOW_SIZE = "5 minutes"


def build_spark():
    """Start a local SparkSession. UTC pins window boundaries to wall-clock time."""
    spark = (
        SparkSession.builder
        .appName("spark-log-anomaly-transform")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_extract_output(spark, path):
    """Read Phase 1's persisted output (all parsed fields are still strings)."""
    return spark.read.parquet(path)


def drop_unparsed(df):
    """
    Drop rows that failed to parse in Phase 1.
    Returns (clean_df, dropped_count) -- the count is reported BEFORE dropping
    so the before/after row count is visible, not silent.
    """
    dropped_count = df.filter(~col("parse_ok")).count()
    # parse_ok is now always True and value is the raw line; neither is needed.
    clean = df.filter(col("parse_ok")).drop("parse_ok", "value")
    return clean, dropped_count


def cast_types(df):
    """Cast the string columns to real types now that parsing is Transform's job."""
    return (
        df
        .withColumn("timestamp", col("timestamp").cast("timestamp"))
        .withColumn("status_code", col("status_code").cast("int"))
        .withColumn("response_size", col("response_size").cast("int"))
        .select("timestamp", "endpoint", "status_code", "response_size")
    )


def dedupe(df):
    """
    Remove exact-duplicate rows.
    Returns (clean_df, duplicate_count, deduped_count); the counts are reported
    even if zero.

    The dropDuplicates shuffle is computed ONCE and its row count is reused for
    the duplicate count below, instead of also running a separate
    distinct().count() (the same dedupe operation) and a separate recount
    upstream. This removes one full dedupe shuffle over the ~1.48M rows.
    """
    full_count = df.count()
    deduped = df.dropDuplicates()
    deduped_count = deduped.count()
    duplicate_count = full_count - deduped_count
    return deduped, duplicate_count, deduped_count


def compute_window_features(clean_df):
    """
    Bucket rows into 5-minute windows and compute global (not per-endpoint)
    features per window: request count, error rate, avg and p95 response size.
    """
    windowed = (
        clean_df
        .groupBy(window(col("timestamp"), WINDOW_SIZE).alias("window"))
        .agg(
            count("*").alias("total_requests"),
            # error_rate is stored as a fraction 0..1 (share of 4xx/5xx).
            avg((col("status_code") >= 400).cast("int")).alias("error_rate"),
            avg("response_size").alias("avg_response_size"),
            percentile("response_size", 0.95).alias("p95_response_size"),
        )
        .withColumn("window_start", col("window.start"))
        .drop("window")
    )
    return windowed


def read_ground_truth(spark, path):
    """Load ground_truth_windows.csv: window_start (timestamp), is_anomalous, anomaly_type."""
    return (
        spark.read.option("header", True).csv(path)
        .withColumn("window_start", to_timestamp(col("window_start"), "yyyy-MM-dd HH:mm:ss"))
        .withColumn("is_anomalous", col("is_anomalous").cast("int"))
        .select("window_start", "is_anomalous", "anomaly_type")
    )


def attach_labels(features_df, gt_df):
    """
    Attach ground-truth labels to each window.
    The ground truth is the canonical window list, so the join starts from the
    ground truth side: exactly 2016 windows survive even if some window had no
    clean rows (features would be NULL there, but no window is dropped).
    """
    return gt_df.join(features_df, on="window_start", how="left")


def verify(labeled_df, step_counts):
    """Print the required verification numbers and sample windows."""
    total_windows = labeled_df.count()
    anomalous = labeled_df.filter(col("is_anomalous") == 1).count()
    null_features = labeled_df.filter(col("total_requests").isNull()).count()
    null_labels = labeled_df.filter(col("is_anomalous").isNull()).count()

    error_stats = labeled_df.agg(
        min("error_rate").alias("min"),
        max("error_rate").alias("max"),
        avg("error_rate").alias("mean"),
    ).collect()[0]
    percentiles = labeled_df.agg(
        percentile("error_rate", [0.25, 0.5, 0.75, 0.95]).alias("pcts")
    ).collect()[0]["pcts"]

    distinct_error_rates = labeled_df.select("error_rate").distinct().count()
    zero_error_windows = labeled_df.filter(col("error_rate") == 0.0).count()

    # Coarse histogram of error rates across windows (checks it's not degenerate).
    bucket = (
        when(col("error_rate") < 0.05, "0-5%")
        .when(col("error_rate") < 0.15, "5-15%")
        .when(col("error_rate") < 0.40, "15-40%")
        .otherwise("40-100%")
    )
    histogram = (
        labeled_df.groupBy(bucket.alias("error_rate_bucket"))
        .count()
        .orderBy("error_rate_bucket")
        .collect()
    )

    print("=" * 64)
    print("VERIFICATION")
    print("=" * 64)
    print("CLEANING -- before/after row counts per step")
    print(f"  rows read from extract output : {step_counts['extracted_rows']}")
    print(f"  parse_ok = False (dropped)    : {step_counts['parse_ok_false']}")
    print(f"  rows after parse_ok drop      : {step_counts['rows_after_parse_drop']}")
    print(f"  exact duplicate rows found    : {step_counts['duplicate_rows_found']}")
    print(f"  rows after dedupe             : {step_counts['rows_after_dedupe']}")

    print("\nWINDOWS + LABELS")
    print(f"  total windows after join      : {total_windows} (expected 2016)")
    print(f"  anomalous windows             : {anomalous} (expected 81, 4.02%)")
    print(f"  windows with NULL features    : {null_features} (0 = every GT window has data)")
    print(f"  windows with NULL labels      : {null_labels} (0 = all windows matched GT)")

    print("\nERROR-RATE DISTRIBUTION ACROSS WINDOWS (fraction of 4xx/5xx)")
    print(f"  min   : {error_stats['min'] * 100:.2f}%")
    print(f"  max   : {error_stats['max'] * 100:.2f}%")
    print(f"  mean  : {error_stats['mean'] * 100:.2f}%")
    p25, p50, p75, p95 = percentiles
    print(f"  p25   : {p25 * 100:.2f}%")
    print(f"  p50   : {p50 * 100:.2f}%")
    print(f"  p75   : {p75 * 100:.2f}%")
    print(f"  p95   : {p95 * 100:.2f}%")
    print(f"  distinct error-rate values   : {distinct_error_rates} (must be > 1)")
    print(f"  windows with error rate 0    : {zero_error_windows}")
    print("  histogram (windows per error-rate band):")
    for row in histogram:
        print(f"    {row['error_rate_bucket']:<10}: {row['count']}")

    print("\n--- sample labeled windows ---")
    print("  (3 error-type, 2 size-type, 2 normal)")
    err = labeled_df.filter(col("anomaly_type") == "error").orderBy("window_start").limit(3)
    siz = labeled_df.filter(col("anomaly_type") == "size").orderBy("window_start").limit(2)
    norm = labeled_df.filter(col("is_anomalous") == 0).orderBy("window_start").limit(2)
    for part in (err, siz, norm):
        for r in part.collect():
            print(f"  {r['window_start']}  anom={r['is_anomalous']} "
                  f"type={r['anomaly_type']:<5}  req={r['total_requests']:>5} "
                  f"err={r['error_rate'] * 100:5.1f}%  avg={r['avg_response_size']:7.1f} "
                  f"p95={r['p95_response_size']:7.1f}")


def run_transform(spark, extract_output, ground_truth, output_path):
    """
    Run the Transform stage with the given shared SparkSession.
    Returns (labeled_df, stats).
    """
    raw = read_extract_output(spark, extract_output)

    step_counts = {}
    step_counts["extracted_rows"] = raw.count()

    cleaned, bad = drop_unparsed(raw)
    step_counts["parse_ok_false"] = bad
    step_counts["rows_after_parse_drop"] = cleaned.count()

    typed = cast_types(cleaned)
    typed, dups, deduped_count = dedupe(typed)
    step_counts["duplicate_rows_found"] = dups
    step_counts["rows_after_dedupe"] = deduped_count

    window_features = compute_window_features(typed)
    # verify() runs ~15 Spark actions over the labeled frame (counts, filters,
    # agg, collect, sample windows) plus the final count/write. Without caching,
    # every one of those re-executes the window aggregation AND the dedupe
    # shuffle over the ~1.48M rows. The window result is only 2016 rows, so
    # caching it is cheap (fits comfortably in memory, unlike the 1.48M-row
    # deduped frame which measured slower when cached) and makes all downstream
    # actions read from memory.
    window_features = window_features.cache()
    window_features.count()  # materialize the small cache now, before verify()
    gt_df = read_ground_truth(spark, ground_truth)
    labeled = attach_labels(window_features, gt_df)

    verify(labeled, step_counts)

    labeled.write.mode("overwrite").parquet(output_path)
    print(f"\nWrote Transform output to {output_path}")

    stats = {"rows_in": step_counts["extracted_rows"], "rows_out": labeled.count()}
    return labeled, stats


def main():
    spark = build_spark()
    try:
        run_transform(spark, EXTRACT_OUTPUT, GROUND_TRUTH, TRANSFORM_OUTPUT)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
