"""
generate_logs.py
================
Synthetic web-server access log generator for the Spark log-anomaly pipeline.

This script creates three files in the same folder as this script:

  1. access.log                -- ~1.5 million lines of synthetic web access logs
  2. ground_truth_windows.csv  -- one label row per 5-minute window (anomaly ground truth)
  3. reference.db              -- SQLite database with endpoint metadata (join target)

LOG FORMAT (space-delimited, one request per line):
    <timestamp> <endpoint> <status_code> <response_size>

  timestamp     : "YYYY-MM-DD HH:MM:SS" (24h)
  endpoint      : one of 15 fake API paths, e.g. /api/search
  status_code   : HTTP status code, e.g. 200
  response_size : integer number of bytes in the response body

Example line:
    2026-07-01 14:23:45 /api/search 200 912

About 1.5% of lines are deliberately malformed (see make_malformed_line)
to simulate real-world log messiness. Malformed rows are spread evenly
across ALL rows and are unrelated to the anomaly labels, so they cannot
be used as a shortcut to find anomalous windows.

Anomalies live at the level of 5-minute windows, not individual rows.
Roughly 4% of all windows are marked anomalous. Each anomalous window is
exactly one of two types:
    - "error" : the share of 4xx/5xx status codes is much higher than normal
    - "size"  : response sizes are much larger than normal (latency/load proxy)
The label for every window is written to ground_truth_windows.csv so the
pipeline can be evaluated later against known ground truth.

Usage:
    python generate_logs.py [--rows 1500000] [--seed 42]
    python generate_logs.py --rows 400000 --seed 12345 --days 2 \\
        --start-datetime "2026-07-08 00:00:00" --prefix new_batch_

    The extra options let this same generator create a NEW, later batch of
    logs (fresh seed, different start date / number of days) into separate
    files (new_batch_access.log, new_batch_ground_truth_windows.csv) WITHOUT
    overwriting the original training data. Phase 7 predicts on such a batch
    to test generalization on data the model has never seen.

Output files are (re)created every run. Exit code 0 on success.
"""

import argparse
import math
import os
import random
import sqlite3
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Tunable settings (constants at the top so they are easy to find and change)
# ---------------------------------------------------------------------------
DEFAULT_ROWS = 1_500_000     # total log lines to write
SEED = 42                    # fixed seed -> reproducible output every run
WINDOW_MINUTES = 5           # anomaly granularity: 5-minute windows
NUM_DAYS = 7                 # logs span a 7-day period
ANOMALY_WINDOW_FRACTION = 0.04    # ~4% of windows are anomalous
MALFORMED_ROW_FRACTION = 0.015    # ~1.5% of rows are deliberately malformed
START_DATETIME = datetime(2026, 7, 1, 0, 0, 0)   # logs start here


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def script_dir():
    """Folder containing this script, so output files sit right next to it."""
    return os.path.dirname(os.path.abspath(__file__))


def build_windows(start, num_days, window_minutes):
    """
    Build the full list of 5-minute window start datetimes.
    Example: 7 days -> 7 * 24 * 12 = 2016 windows.
    """
    step = timedelta(minutes=window_minutes)
    end = start + timedelta(days=num_days)
    windows = []
    current = start
    while current < end:
        windows.append(current)
        current += step
    return windows


def time_of_day_factor(dt):
    """
    Traffic multiplier for the hour of day. Daytime is busy, night is quiet,
    so the log has a believable daily rhythm instead of flat random noise.
    """
    hour = dt.hour
    if 0 <= hour < 5:      # deep night: very quiet
        return 0.15
    if 5 <= hour < 8:      # early morning: waking up
        return 0.5
    if 8 <= hour < 12:     # morning: busy
        return 1.5
    if 12 <= hour < 14:    # lunchtime: slightly slower
        return 1.3
    if 14 <= hour < 18:    # afternoon: busiest
        return 1.6
    if 18 <= hour < 22:    # evening: still active
        return 1.2
    return 0.6             # late night: winding down


def weekday_factor(dt):
    """
    Slight weekend dip: Saturday/Sunday are quieter than weekdays.
    Python weekday(): Monday=0 ... Sunday=6.
    """
    return 0.8 if dt.weekday() >= 5 else 1.0


# ---------------------------------------------------------------------------
# Domain data
# ---------------------------------------------------------------------------

def build_endpoints():
    """
    The 15 fake endpoints. Each entry carries:
      - endpoint          : URL path
      - traffic_weight    : how often it is hit (long-tail: a few get most traffic)
      - service_owner     : made-up owning team (goes into the reference table)
      - expected_avg_size : made-up average response size in bytes (reference table)

    The traffic weights deliberately add up to 1.0 so they can be used as
    probabilities directly.
    """
    return [
        {"endpoint": "/api/search",          "traffic_weight": 0.20, "service_owner": "search-team",   "expected_avg_size": 900},
        {"endpoint": "/api/products",        "traffic_weight": 0.15, "service_owner": "catalog-team",  "expected_avg_size": 2400},
        {"endpoint": "/api/home",            "traffic_weight": 0.12, "service_owner": "web-team",      "expected_avg_size": 450},
        {"endpoint": "/api/login",           "traffic_weight": 0.10, "service_owner": "auth-team",     "expected_avg_size": 320},
        {"endpoint": "/api/profile",         "traffic_weight": 0.08, "service_owner": "user-team",     "expected_avg_size": 380},
        {"endpoint": "/api/cart",            "traffic_weight": 0.06, "service_owner": "cart-team",     "expected_avg_size": 610},
        {"endpoint": "/api/categories",      "traffic_weight": 0.05, "service_owner": "catalog-team",  "expected_avg_size": 700},
        {"endpoint": "/api/recommendations", "traffic_weight": 0.04, "service_owner": "ml-team",       "expected_avg_size": 1500},
        {"endpoint": "/api/orders",          "traffic_weight": 0.04, "service_owner": "order-team",    "expected_avg_size": 520},
        {"endpoint": "/api/notifications",   "traffic_weight": 0.04, "service_owner": "notify-team",   "expected_avg_size": 200},
        {"endpoint": "/api/analytics",       "traffic_weight": 0.03, "service_owner": "data-team",     "expected_avg_size": 1100},
        {"endpoint": "/api/checkout",        "traffic_weight": 0.03, "service_owner": "order-team",    "expected_avg_size": 1600},
        {"endpoint": "/api/reviews",         "traffic_weight": 0.02, "service_owner": "community-team","expected_avg_size": 340},
        {"endpoint": "/api/payments",        "traffic_weight": 0.02, "service_owner": "payments-team", "expected_avg_size": 220},
        {"endpoint": "/api/admin",           "traffic_weight": 0.02, "service_owner": "platform-team", "expected_avg_size": 480},
    ]


# ---------------------------------------------------------------------------
# Anomaly selection (ground truth construction)
# ---------------------------------------------------------------------------

def pick_anomalous_windows(windows, rng):
    """
    Choose ~4% of windows to be anomalous and decide each one's type.
    Types are split roughly 50/50 between "error" and "size" so the label
    file is not dominated by one kind.
    Returns a dict: window index -> anomaly type ("error" or "size").
    """
    total = len(windows)
    n_anomalies = round(total * ANOMALY_WINDOW_FRACTION)
    picks = sorted(rng.sample(range(total), n_anomalies))
    anomalies = {}
    for i, idx in enumerate(picks):
        # Alternate the type across the (already shuffled) picks -> ~50/50 split.
        anomalies[idx] = "error" if i % 2 == 0 else "size"
    return anomalies


# ---------------------------------------------------------------------------
# Per-row generation
# ---------------------------------------------------------------------------

def pick_status_code(rng, error_window):
    """
    Return an HTTP status code.
    In a normal window almost all requests succeed (200).
    In an error-anomalous window a large share of requests are 4xx/5xx.
    """
    if error_window:
        roll = rng.random()
        if roll < 0.45:
            return 200
        if roll < 0.55:
            return 201
        if roll < 0.62:
            return 400
        if roll < 0.77:
            return 404
        if roll < 0.92:
            return 500
        return 503

    roll = rng.random()
    if roll < 0.88:
        return 200
    if roll < 0.92:
        return 201
    if roll < 0.94:
        return 204
    if roll < 0.95:
        return 301
    if roll < 0.97:
        return 302
    if roll < 0.98:
        return 400
    if roll < 0.995:
        return 404
    return 500


def pick_response_size(rng, expected_avg_size, status_code, size_anomaly):
    """
    Draw a response size in bytes from a log-normal distribution centred on
    the endpoint's expected average size (so sizes cluster around a realistic
    value instead of being uniform). Error responses carry a small body.
    In a size-anomalous window every size is inflated several times over.
    """
    if status_code >= 400:
        # Error bodies are small, ~180 bytes on average.
        return int(rng.lognormvariate(math.log(180) - 0.045, 0.3))

    # sigma=0.5, so mean = exp(mu + 0.5^2/2) = exp(mu + 0.125).
    # Setting mu = ln(expected) - 0.125 makes the mean equal the expected size.
    size = int(rng.lognormvariate(math.log(expected_avg_size) - 0.125, 0.5))
    if size_anomaly:
        size = int(size * rng.uniform(5.0, 15.0))   # latency/load proxy spike
    return max(size, 1)


def make_malformed_line(rng, line):
    """
    Turn a valid log line into a malformed one, simulating messy real-world
    logs. Five different kinds of breakage are used. These malformed rows are
    scattered uniformly across all rows, independent of anomaly labels, so
    they do not leak which windows are anomalous.
    """
    kind = rng.randint(0, 4)

    if kind == 0:
        # Drop the status code entirely (all following fields shift).
        parts = line.split(" ")
        del parts[3]
        return " ".join(parts)

    if kind == 1:
        # Garbled timestamp (impossible date/time), endpoint and rest kept.
        return "20XX-XX-XX 99:99:99 " + line.split(" ", 2)[2]

    if kind == 2:
        # Truncated line: cut off mid-way.
        return line[:rng.randint(5, 15)]

    if kind == 3:
        # Non-numeric response size.
        parts = line.split(" ")
        parts[4] = "ABCxyz"
        return " ".join(parts)

    # kind == 4: extra junk tokens appended.
    return line + ' "extra=1" sid=deadbeef'


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_logs(access_log_path, windows, endpoints, anomalies, rng, total_rows):
    """
    Write access.log one line at a time (streaming, so memory stays small).
    Returns (lines_written, malformed_written, endpoint_counter).
    """
    endpoint_names = [e["endpoint"] for e in endpoints]
    traffic_weights = [e["traffic_weight"] for e in endpoints]
    expected_by_endpoint = {e["endpoint"]: e["expected_avg_size"] for e in endpoints}

    # Decide how many lines each window gets so the daily traffic pattern
    # (busy day, quiet night, lighter weekends) actually shows up in the data.
    window_weights = []
    total_weight = 0.0
    for dt in windows:
        weight = time_of_day_factor(dt) * weekday_factor(dt)
        weight *= rng.uniform(0.7, 1.3)   # small random noise per window
        window_weights.append(weight)
        total_weight += weight

    lines_per_window = [round(total_rows * w / total_weight) for w in window_weights]
    # Fix rounding drift so the grand total is exactly total_rows.
    lines_per_window[-1] += total_rows - sum(lines_per_window)

    lines_written = 0
    malformed_written = 0
    endpoint_counter = {name: 0 for name in endpoint_names}

    with open(access_log_path, "w", encoding="utf-8") as f:
        for window_idx, dt in enumerate(windows):
            anomaly_type = anomalies.get(window_idx)
            error_window = anomaly_type == "error"
            size_window = anomaly_type == "size"

            for _ in range(lines_per_window[window_idx]):
                endpoint = rng.choices(endpoint_names, weights=traffic_weights, k=1)[0]
                status = pick_status_code(rng, error_window)
                size = pick_response_size(rng, expected_by_endpoint[endpoint], status, size_window)

                # Jitter the timestamp a few seconds inside the window so not
                # all rows in a window share the exact same second.
                row_time = dt + timedelta(seconds=rng.randint(0, WINDOW_MINUTES * 60 - 1))
                line = f"{row_time.strftime('%Y-%m-%d %H:%M:%S')} {endpoint} {status} {size}"

                endpoint_counter[endpoint] += 1
                if rng.random() < MALFORMED_ROW_FRACTION:
                    line = make_malformed_line(rng, line)
                    malformed_written += 1

                f.write(line + "\n")
                lines_written += 1

    return lines_written, malformed_written, endpoint_counter


def write_ground_truth(csv_path, windows, anomalies):
    """
    Write one CSV row per 5-minute window:
        window_start, is_anomalous (0/1), anomaly_type (none/error/size)
    This file is the defensible ground truth for later evaluation.
    """
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("window_start,is_anomalous,anomaly_type\n")
        for idx, dt in enumerate(windows):
            a = anomalies.get(idx)
            if a is None:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')},0,none\n")
            else:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')},1,{a}\n")


def create_reference_db(db_path, endpoints):
    """
    Create reference.db with a single table endpoint_metadata:
        endpoint TEXT PRIMARY KEY, service_owner TEXT, expected_avg_response_size INTEGER
    Phase 1 (Extract) will join log rows against this table.
    """
    if os.path.exists(db_path):
        os.remove(db_path)   # rebuild fresh every run

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE endpoint_metadata (
            endpoint                    TEXT PRIMARY KEY,
            service_owner               TEXT    NOT NULL,
            expected_avg_response_size  INTEGER NOT NULL
        )
        """
    )
    rows = [
        (e["endpoint"], e["service_owner"], e["expected_avg_size"])
        for e in endpoints
    ]
    cur.executemany("INSERT INTO endpoint_metadata VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic web access logs with injected anomalies."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help="total number of log lines to write")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="random seed for reproducible output")
    parser.add_argument("--days", type=int, default=NUM_DAYS,
                        help="how many days of logs to generate")
    parser.add_argument("--start-datetime", type=str, default=None,
                        help='first timestamp, format "YYYY-MM-DD HH:MM:SS" '
                             '(default: the module constant START_DATETIME)')
    parser.add_argument("--prefix", type=str, default="",
                        help="optional filename prefix, e.g. new_batch_ -- when "
                             "set, logs/ground truth go to <prefix>access.log and "
                             "<prefix>ground_truth_windows.csv instead of the "
                             "default names, so an existing dataset is not "
                             "overwritten")
    args = parser.parse_args()

    if args.start_datetime:
        start = datetime.strptime(args.start_datetime, "%Y-%m-%d %H:%M:%S")
    else:
        start = START_DATETIME

    out_dir = script_dir()
    access_log_path = os.path.join(out_dir, f"{args.prefix}access.log")
    ground_truth_path = os.path.join(out_dir, f"{args.prefix}ground_truth_windows.csv")
    reference_db_path = os.path.join(out_dir, "reference.db")

    rng = random.Random(args.seed)

    windows = build_windows(start, args.days, WINDOW_MINUTES)
    anomalies = pick_anomalous_windows(windows, rng)
    endpoints = build_endpoints()

    # The three outputs are independent of each other; ground truth and the
    # reference table are small, the log is streamed out line by line.
    # The reference table is static endpoint metadata (identical for any
    # batch), so it is always written to the shared reference.db path.
    write_ground_truth(ground_truth_path, windows, anomalies)
    create_reference_db(reference_db_path, endpoints)

    lines_written, malformed_written, endpoint_counter = generate_logs(
        access_log_path, windows, endpoints, anomalies, rng, args.rows
    )

    # Brief human-readable summary printed to stdout.
    print(f"Generated batch from {start} over {args.days} day(s), "
          f"seed={args.seed}")
    print(f"Wrote {lines_written} log lines to {access_log_path}")
    print(f"  malformed lines: {malformed_written} "
          f"({100 * malformed_written / lines_written:.2f}%)")
    print(f"Wrote ground truth for {len(windows)} windows to {ground_truth_path}")
    n_anomalies = len(anomalies)
    print(f"  anomalous windows: {n_anomalies} ({100 * n_anomalies / len(windows):.2f}%)")
    print(f"    type error: {sum(1 for a in anomalies.values() if a == 'error')}")
    print(f"    type size:  {sum(1 for a in anomalies.values() if a == 'size')}")
    print(f"Created SQLite reference table at {reference_db_path}")
    print("Per-endpoint row counts (long-tail check):")
    for name, count in sorted(endpoint_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
