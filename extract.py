"""
extract.py
==========
Phase 1 (Extract) of the Spark log-anomaly pipeline.

Reads the two data sources produced by generate_logs.py into Spark:

  1. access.log     -- read with spark.read.text(), treated as semi-structured:
                       one column of raw lines, no schema forced at read time.
  2. reference.db   -- the endpoint_metadata SQLite table (tiny reference data).

Scope guard: Extract only reads and tags. There is no cleaning, filtering,
or feature engineering here -- that belongs to Phase 2. Rows that fail to
parse are TAGGED with parse_ok=False and kept, never dropped, because Phase 2
owns the drop/flag decision.

Usage:
    python extract.py

Prints verification numbers and sample rows to stdout.
"""

import os
import sqlite3

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when
from pyspark.sql.types import LongType, StringType, StructField, StructType

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "access.log")
REFERENCE_DB = os.path.join(SCRIPT_DIR, "reference.db")

# The format the generator wrote (documented at the top of generate_logs.py):
#     <timestamp> <endpoint> <status_code> <response_size>
# e.g. "2026-07-01 14:23:45 /api/search 200 912"
#
# The anchors ^ and $ force a FULL-line match, so any malformed line
# (missing field, garbled timestamp, non-numeric size, extra junk, ...)
# fails and is flagged parse_ok=False instead of being parsed wrongly.
PARSE_REGEX = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (/\S+) (\d{3}) (\d+)$"


def build_spark():
    """Start a local SparkSession on this laptop."""
    spark = (
        SparkSession.builder
        .appName("spark-log-anomaly-extract")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")  # keep the console output readable
    return spark


def read_raw_log(spark, path):
    """
    Read access.log as unstructured text: one column named 'value' holding
    each raw line. No schema is imposed at read time -- parsing happens next.
    """
    return spark.read.text(path)


def parse_log(raw_df):
    """
    Split each raw line into timestamp / endpoint / status_code / response_size
    with a regex. Unparseable lines are tagged parse_ok=False and retained.
    """
    extracted = raw_df
    extracted = extracted.withColumn("timestamp", regexp_extract(col("value"), PARSE_REGEX, 1))
    extracted = extracted.withColumn("endpoint", regexp_extract(col("value"), PARSE_REGEX, 2))
    extracted = extracted.withColumn("status_code", regexp_extract(col("value"), PARSE_REGEX, 3))
    extracted = extracted.withColumn("response_size", regexp_extract(col("value"), PARSE_REGEX, 4))
    # Group 0 is the whole match. With ^...$ it equals the full line only when
    # the line conforms to the format, so it is exactly our parse_ok flag.
    extracted = extracted.withColumn(
        "parse_ok",
        when(regexp_extract(col("value"), PARSE_REGEX, 0) == col("value"), True).otherwise(False),
    )
    return extracted


def read_reference_table(spark, db_path):
    """
    Load endpoint_metadata from SQLite into a Spark DataFrame.

    Chosen approach: stdlib sqlite3 -> spark.createDataFrame, NOT JDBC.
    JDBC would need a third-party sqlite driver jar and a classpath setup;
    this table has 15 rows on a single laptop, so plain sqlite3 (which is in
    Python's standard library) is simpler and has zero extra dependencies.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT endpoint, service_owner, expected_avg_response_size "
        "FROM endpoint_metadata"
    ).fetchall()
    conn.close()

    schema = StructType([
        StructField("endpoint", StringType(), nullable=False),
        StructField("service_owner", StringType(), nullable=False),
        StructField("expected_avg_response_size", LongType(), nullable=False),
    ])
    return spark.createDataFrame(rows, schema)


def broadcast_join(parsed_df, reference_df):
    """
    Join the parsed logs to the endpoint metadata table on 'endpoint'.

    The reference table is tiny (15 rows) and the log data is large (1.5M
    rows), so this uses a BROADCAST join: Spark copies the small table into
    every task and skips an expensive shuffle entirely. This is normally a
    Phase-5 optimization, but it is trivially correct here (the reference
    table fits in memory), so it is applied early -- an obvious win, not
    scope creep.

    A left join guarantees no log row disappears. The reference table's
    'endpoint' column is a PRIMARY KEY, so the join cannot duplicate rows.
    """
    return parsed_df.join(reference_df.hint("broadcast"), on="endpoint", how="left")


def join_used_broadcast(joined_df):
    """
    True when the executed physical plan contains a BroadcastHashJoin.
    Uses the internal _jdf bridge to reach the JVM query execution object;
    the public PySpark API does not expose the raw plan.
    """
    plan = joined_df._jdf.queryExecution().executedPlan().toString()
    return "BroadcastHashJoin" in plan


def verify(raw_df, parsed_df, joined_df):
    """Print the required verification numbers and sample rows."""
    raw_count = raw_df.count()
    parsed_count = parsed_df.count()
    bad_count = parsed_df.filter(~col("parse_ok")).count()
    bad_rate = 100.0 * bad_count / parsed_count if parsed_count else 0.0
    joined_count = joined_df.count()

    print("=" * 64)
    print("VERIFICATION")
    print("=" * 64)
    print(f"raw log lines       : {raw_count}")
    print(f"parsed rows         : {parsed_count}")
    print(f"  parse_ok = False  : {bad_count} ({bad_rate:.3f}%)")
    print(f"  (known ~1.5% malformed = 22,438 / 1,500,000 = 1.496%)")
    print(f"rows after join     : {joined_count}")
    print(f"join preserved all rows (no drop/dup): "
          f"{joined_count == parsed_count == raw_count}")
    print(f"broadcast join used : {join_used_broadcast(joined_df)}")

    print("\n--- 10 sample rows with parse_ok = False ---")
    sample_bad = parsed_df.filter(~col("parse_ok")).select(
        "value", "timestamp", "endpoint", "status_code", "response_size"
    )
    for row in sample_bad.limit(10).collect():
        print(f"  raw   : {row['value']!r}")
        print(f"  parsed: ts={row['timestamp']!r} ep={row['endpoint']!r} "
              f"status={row['status_code']!r} size={row['response_size']!r}")

    print("\n--- 5 sample rows with parse_ok = True (clean) ---")
    # Note: read from joined_df, not parsed_df, because service_owner and
    # expected_avg_response_size only exist after the join to reference.db.
    sample_good = joined_df.filter(col("parse_ok")).select(
        "timestamp", "endpoint", "status_code", "response_size",
        "service_owner", "expected_avg_response_size",
    )
    for row in sample_good.limit(5).collect():
        print(f"  {row['timestamp']} {row['endpoint']} {row['status_code']} "
              f"{row['response_size']}  owner={row['service_owner']} "
              f"expected_size={row['expected_avg_response_size']}")


def main():
    spark = build_spark()
    try:
        raw_df = read_raw_log(spark, LOG_FILE)
        parsed_df = parse_log(raw_df)
        reference_df = read_reference_table(spark, REFERENCE_DB)
        joined_df = broadcast_join(parsed_df, reference_df)

        verify(raw_df, parsed_df, joined_df)
        print("\n--- physical plan (shows the join strategy) ---")
        joined_df.explain()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
