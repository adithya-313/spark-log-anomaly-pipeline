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
import time

import extract
import load
import transform


def build_spark():
    """
    Create the ONE SparkSession shared by all three stages.
    UTC is required by Transform's 5-minute window bucketing, so the shared
    session carries that config (Extract and Load do not care).
    """
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("spark-log-anomaly-pipeline")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


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
    args = parser.parse_args()

    if args.sample_rows:
        print(f"RUN MODE: SAMPLE -- first {args.sample_rows} rows of access.log\n")
    else:
        print("RUN MODE: FULL -- entire access.log\n")

    spark = build_spark()
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

        print("\nPipeline complete.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
