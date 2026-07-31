r"""
run_pipeline.py
===============
Phase 4 (Integration) of the Spark log-anomaly pipeline.

Orchestrates the three existing stages in one end-to-end run:
    Extract -> Transform -> Load
using ONE shared SparkSession (created here, passed into each stage), and
logs rows-in / rows-out / elapsed wall-clock time per stage.

The stage logic itself stays in extract.py / transform.py / load.py -- this
script only calls their run_* entry points in order. It contains no stage
logic of its own.

Usage:
    python run_pipeline.py                   # full 1.5M-row dataset
    python run_pipeline.py --sample-rows 10000   # first 10,000 lines only

ENV NOTE (required to run Spark on this laptop):
    $env:JAVA_HOME = "C:\Users\sinne\jdk17\jdk-17.0.20+8"
    $env:HADOOP_HOME = "C:\Users\sinne\hadoop-utils"          # needed for file writes
    $env:PATH = "C:\Users\sinne\hadoop-utils\bin;$env:PATH"   # hadoop.dll for native IO
    $env:SPARK_LOCAL_IP = "127.0.0.1"
    $env:PYSPARK_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"
    $env:PYSPARK_DRIVER_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"
"""

import argparse
import io
import json
import time
import urllib.request

import extract
import load
import transform


def build_spark(shuffle_partitions=None):
    """
    Create the ONE SparkSession shared by all three stages.
    UTC is required by Transform's 5-minute window bucketing, so the shared
    session carries that config (Extract and Load do not care).

    shuffle_partitions: optional override of spark.sql.shuffle.partitions
    (the number of reduce-side partitions after any shuffle). Default 200 is
    sized for cluster shuffles; on a single laptop producing only 2016 window
    rows it is far more reduce tasks than needed, so this knob lets us prove
    a smaller value (e.g. 8) speeds things up without changing results.
    """
    from pyspark.sql import SparkSession
    builder = (
        SparkSession.builder
        .appName("spark-log-anomaly-pipeline")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
    )
    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", shuffle_partitions)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def print_stage_metrics(spark):
    """
    Headless substitute for the Spark UI's DAG/SQL stage view: query the
    running app's own REST API (localhost:<uiPort>/api/v1/applications/...) for
    every completed stage and print, per stage: name, number of tasks, and the
    shuffle bytes read/written. This is the same data the UI's Stages tab
    shows, fetched programmatically because there is no browser in this env.
    """
    app_id = spark.sparkContext.applicationId
    ui_url = spark.sparkContext.uiWebUrl  # e.g. http://127.0.0.1:4040
    url = f"{ui_url}/api/v1/applications/{app_id}/stages"
    with urllib.request.urlopen(url, timeout=60) as resp:
        stages = json.load(resp)

    print("\n--- per-stage execution metrics (from Spark UI REST API) ---")
    for s in sorted(stages, key=lambda x: x["stageId"]):
        shuffle_in = s.get("shuffleReadBytes", 0) or 0
        shuffle_out = s.get("shuffleWriteBytes", 0) or 0
        print(f"  stage {s['stageId']:<3} tasks={s['numTasks']:<4} "
              f"shuffle read={shuffle_in:>12,} B  shuffle write={shuffle_out:>12,} B  "
              f"{s['name'][:80]}")


def run_stage(name, fn, **kwargs):
    """Call a stage function once, then report rows in / rows out / elapsed time."""
    start = time.perf_counter()
    _df, stats = fn(**kwargs)
    elapsed = time.perf_counter() - start

    print("=" * 64)
    print(f"STAGE: {name}")
    print(f"  rows in : {stats['rows_in']}")
    print(f"  rows out: {stats['rows_out']}")
    print(f"  elapsed : {elapsed:.2f} s")
    print("=" * 64)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run the full log-anomaly pipeline end to end."
    )
    parser.add_argument(
        "--sample-rows", type=int, default=None,
        help="Only process the first N rows of access.log (default: all rows)",
    )
    parser.add_argument(
        "--shuffle-partitions", type=int, default=None,
        help="Override spark.sql.shuffle.partitions (default: Spark's 200)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="After the run, print per-stage shuffle metrics from the Spark UI "
             "REST API (headless substitute for the UI's stage view)",
    )
    args = parser.parse_args()

    if args.sample_rows:
        print(f"RUN MODE: SAMPLE -- first {args.sample_rows} rows of access.log\n")
    else:
        print("RUN MODE: FULL -- entire access.log\n")
    if args.shuffle_partitions is not None:
        print(f"CONFIG: spark.sql.shuffle.partitions = {args.shuffle_partitions}\n")
    else:
        print("CONFIG: spark.sql.shuffle.partitions = 200 (Spark default)\n")

    spark = build_spark(shuffle_partitions=args.shuffle_partitions)
    try:
        # Stages run strictly in order; each writes its output parquet so the
        # next stage reads it back from disk (same contract as standalone runs).
        run_stage("EXTRACT", extract.run_extract,
                  spark=spark,
                  log_file=extract.LOG_FILE,
                  reference_db=extract.REFERENCE_DB,
                  output_path=extract.EXTRACT_OUTPUT,
                  row_limit=args.sample_rows)

        run_stage("TRANSFORM", transform.run_transform,
                  spark=spark,
                  extract_output=transform.EXTRACT_OUTPUT,
                  ground_truth=transform.GROUND_TRUTH,
                  output_path=transform.TRANSFORM_OUTPUT)

        run_stage("LOAD", load.run_load,
                  spark=spark,
                  transform_output=load.TRANSFORM_OUTPUT,
                  output_path=load.LOAD_OUTPUT)

        if args.profile:
            print_stage_metrics(spark)

        print("\nPipeline complete.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
