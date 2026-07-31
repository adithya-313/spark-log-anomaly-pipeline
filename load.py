r"""
load.py
=======
Phase 3 (Load) of the Spark log-anomaly pipeline.

Reads Phase 2's output (transform_output.parquet) and writes the final
partitioned deliverable: windowed_features/ partitioned by date (derived
from window_start).

Partitioning decision: DATE only, not by hour. 2016 windows span 7 days, so
date partitions hold ~288 rows each (sensible); hour partitions would hold
only ~12 rows each -- far too small to justify the file overhead at this
row count. If the dataset grew to millions of windows, hour partitions
would be worth revisiting.

Verification (round-trip): re-reads the newly written partitioned output and
checks (1) row count == 2016, (2) schema unchanged (no type coercion),
(3) exact data equality against the input, (4) per-partition counts sum to
2016, and (5) prints the actual on-disk directory tree.

Git/data management decision: generated data (access.log, *.parquet, sqlite,
ground-truth CSV) is reproducible by running the pipeline scripts, so it is
.gitignored going forward. Previously committed copies remain in git
history -- they are untracked in this commit, not deleted, and history is
NOT rewritten.

ENV NOTE (required to run Spark on this laptop):
    $env:JAVA_HOME = "C:\Users\sinne\jdk17\jdk-17.0.20+8"
    $env:HADOOP_HOME = "C:\Users\sinne\hadoop-utils"          # needed for file writes
    $env:PATH = "C:\Users\sinne\hadoop-utils\bin;$env:PATH"   # hadoop.dll for native IO
    $env:SPARK_LOCAL_IP = "127.0.0.1"
    $env:PYSPARK_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"
    $env:PYSPARK_DRIVER_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"

Usage:
    python load.py
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORM_OUTPUT = os.path.join(SCRIPT_DIR, "transform_output.parquet")
LOAD_OUTPUT = os.path.join(SCRIPT_DIR, "windowed_features")

# The data columns (everything except the derived partition column 'date').
DATA_COLUMNS = [
    "window_start", "is_anomalous", "anomaly_type", "total_requests",
    "error_rate", "avg_response_size", "p95_response_size",
]


def build_spark():
    """Start a local SparkSession. UTC keeps timestamps wall-clock consistent."""
    spark = (
        SparkSession.builder
        .appName("spark-log-anomaly-load")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_transform_output(spark, path):
    """Read Phase 2's output: 2016 labeled window rows."""
    return spark.read.parquet(path)


def derive_partition(df):
    """
    Add the partition column 'date' from window_start (e.g. 2026-07-01).
    Spark uses this column to create date=... directories on write.
    """
    return df.withColumn("date", date_format(col("window_start"), "yyyy-MM-dd"))


def write_partitioned(df, path):
    """Write the final partitioned artifact (overwrites a previous run)."""
    df.write.mode("overwrite").partitionBy("date").parquet(path)


def list_tree(root):
    """Return a printable tree of the output directory (real on-disk layout)."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        depth = dirpath[len(root):].count(os.sep)
        indent = "  " * depth
        lines.append(f"{indent}{os.path.basename(dirpath)}/")
        for name in filenames:
            lines.append(f"{indent}  {name}")
    return "\n".join(lines)


def verify(spark, partitioned_df):
    """Round-trip verification of the partitioned write."""
    reread = spark.read.parquet(LOAD_OUTPUT)

    row_count = reread.count()

    # Schema check: compare ONLY the 7 real data columns on both sides. The
    # derived partition column 'date' is deliberately excluded because Spark
    # writes it as a string but infers it as DATE type when re-reading a
    # partitioned directory -- normal Spark behavior, not data coercion. The
    # data columns themselves must be identical (names + types, no coercion).
    data_cols = set(DATA_COLUMNS)
    expected_fields = {(f.name, str(f.dataType)) for f in partitioned_df.schema.fields
                       if f.name in data_cols}
    actual_fields = {(f.name, str(f.dataType)) for f in reread.schema.fields
                     if f.name in data_cols}
    schema_ok = actual_fields == expected_fields

    # Exact data round-trip: compare every row (dropping the derived 'date'
    # column, which is a duplicate of window_start) in a stable order.
    left = partitioned_df.select(*DATA_COLUMNS).orderBy(*DATA_COLUMNS).collect()
    right = reread.select(*DATA_COLUMNS).orderBy(*DATA_COLUMNS).collect()
    data_ok = left == right

    per_date = reread.groupBy("date").count().orderBy("date").collect()
    per_date_total = sum(r["count"] for r in per_date)

    print("=" * 64)
    print("VERIFICATION (round-trip)")
    print("=" * 64)
    print(f"row count in re-read     : {row_count} (expected 2016)")
    print(f"schema matches (no coercion): {schema_ok}")
    print("  written schema:")
    for f in partitioned_df.schema.fields:
        print(f"    {f.name}: {f.dataType}")
    print("  re-read schema:")
    for f in reread.schema.fields:
        print(f"    {f.name}: {f.dataType}")
    print("  (note: 'date' changes StringType -> DateType on re-read -- expected "
          "Spark partition-column inference; excluded from the check)")
    print(f"exact data round-trip    : {data_ok}")
    print(f"per-partition counts     : {len(per_date)} partitions, total {per_date_total}")
    for r in per_date:
        print(f"    {r['date']}: {r['count']} rows")

    print("\n--- on-disk directory structure ---")
    print(list_tree(LOAD_OUTPUT))


def run_load(spark, transform_output, output_path):
    """
    Run the Load stage with the given shared SparkSession.
    Returns (reread_df, stats).
    """
    original = read_transform_output(spark, transform_output)
    partitioned = derive_partition(original)
    write_partitioned(partitioned, output_path)
    verify(spark, partitioned)
    print(f"\nWrote partitioned Load output to {output_path}")

    reread = spark.read.parquet(output_path)
    stats = {"rows_in": partitioned.count(), "rows_out": reread.count()}
    return reread, stats


def main():
    spark = build_spark()
    try:
        run_load(spark, TRANSFORM_OUTPUT, LOAD_OUTPUT)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
