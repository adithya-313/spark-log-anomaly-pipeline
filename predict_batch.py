r"""
predict_batch.py
================
Phase 7 (Prediction) of the Spark log-anomaly pipeline.

Takes a NEW batch of raw logs -- data the model has never seen in ANY form,
not merely held out at training time -- runs it through the same
Extract -> Transform stages used for training, loads the saved Phase-6 model,
and writes per-window predictions (predicted label + probability).

It reuses extract.run_extract and transform.run_transform directly: the
feature engineering is NOT duplicated here. If the features ever change, they
change in exactly one place (transform.py) and this script inherits the change.

This is a batch/offline script, consistent with the rest of the project: no
serving API, no streaming, no containers. Run manually whenever a new log file
arrives.

IMPORTANT -- what this is (and is not):
  The new batch is generated with a FRESH random seed and a LATER date range
  than the training data (see generate_logs.py --prefix new_batch_). It is
  not "the same test split reused" -- it is a simulated batch that arrived
  after the model was trained, so its ground truth was unknown at training
  time. Because the generator writes its own ground-truth CSV, predictions
  can still be evaluated against reality (same report card as Phase 6:
  precision / recall / F1 / confusion matrix).

Outputs:
  new_batch_extract.parquet/    -- Phase-1 output for the new batch
  new_batch_transform.parquet/  -- Phase-2 output (windowed features + labels)
  predictions.parquet/          -- per-window: window_start, is_anomalous,
                                   anomaly_type, prediction, prob_anomalous

ENV NOTE (required to run Spark on this laptop):
    $env:JAVA_HOME = "C:\Users\sinne\jdk17\jdk-17.0.20+8"
    $env:HADOOP_HOME = "C:\Users\sinne\hadoop-utils"          # needed for file writes
    $env:PATH = "C:\Users\sinne\hadoop-utils\bin;$env:PATH"   # hadoop.dll for native IO
    $env:SPARK_LOCAL_IP = "127.0.0.1"
    $env:PYSPARK_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"
    $env:PYSPARK_DRIVER_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"

Usage:
    python predict_batch.py
"""

import os

from pyspark.ml.functions import vector_to_array
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import extract
import transform
from train_model import FEATURE_COLUMNS, LABEL_COLUMN, evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The saved Phase-6 model (PipelineModel: VectorAssembler + LogisticRegression).
MODEL_PATH = os.path.join(SCRIPT_DIR, "trained_model")

# The new, post-training batch produced by generate_logs.py --prefix new_batch_.
NEW_BATCH_LOG = os.path.join(SCRIPT_DIR, "new_batch_access.log")
NEW_BATCH_GROUND_TRUTH = os.path.join(SCRIPT_DIR, "new_batch_ground_truth_windows.csv")
NEW_BATCH_EXTRACT = os.path.join(SCRIPT_DIR, "new_batch_extract.parquet")
NEW_BATCH_TRANSFORM = os.path.join(SCRIPT_DIR, "new_batch_transform.parquet")
PREDICTIONS_OUTPUT = os.path.join(SCRIPT_DIR, "predictions.parquet")


def build_spark():
    """Start a local SparkSession (UTC keeps window boundaries consistent)."""
    spark = (
        SparkSession.builder
        .appName("spark-log-anomaly-predict")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run_extract_and_transform(spark, log_path, ground_truth_path):
    """
    Reuse the existing Extract and Transform stages unchanged on the new batch.
    Returns the labeled windowed-features DataFrame (window_start, is_anomalous,
    anomaly_type, total_requests, error_rate, avg_response_size,
    p95_response_size). The label comes from the new batch's own ground truth.
    """
    print("=" * 64)
    print("STAGE: EXTRACT (new batch)")
    print("=" * 64)
    extract.run_extract(
        spark,
        log_file=log_path,
        reference_db=extract.REFERENCE_DB,
        output_path=NEW_BATCH_EXTRACT,
    )

    print("=" * 64)
    print("STAGE: TRANSFORM (new batch)")
    print("=" * 64)
    labeled, stats = transform.run_transform(
        spark,
        extract_output=NEW_BATCH_EXTRACT,
        ground_truth=ground_truth_path,
        output_path=NEW_BATCH_TRANSFORM,
    )
    print(f"  new-batch windows with features: {stats['rows_out']}")
    return labeled


def load_model(path):
    """Load the saved Phase-6 PipelineModel (native MLlib format)."""
    model = PipelineModel.load(path)
    print(f"  loaded saved model from {path}")
    return model


def predict(model, labeled_df):
    """
    Run the loaded pipeline over the new batch's windows.
    The pipeline already contains the VectorAssembler, so it expects the raw
    feature columns (total_requests, error_rate, avg_response_size,
    p95_response_size) and emits features / rawPrediction / probability /
    prediction. A readable per-window table is built from the two-class
    probability vector (prob_anomalous = P(class 1)).
    """
    pred = (
        model
        .transform(labeled_df)
        .withColumn("prob_anomalous", vector_to_array(col("probability")).getItem(1))
        .select("window_start", LABEL_COLUMN, "anomaly_type", "prediction", "prob_anomalous")
    )

    print("\n--- per-window predictions (new batch, chronological) ---")
    for r in pred.orderBy("window_start").collect():
        print(f"  {r['window_start']}  actual={r[LABEL_COLUMN]} "
              f"type={r['anomaly_type']:<5}  predicted={int(r['prediction'])} "
              f"P(anomalous)={r['prob_anomalous']:.4f}")

    flagged = pred.filter(col("prediction") == 1).count()
    total = pred.count()
    print(f"\n  windows generated        : {total}")
    print(f"  windows flagged anomalous: {flagged} "
          f"({100 * flagged / total:.2f}%)")
    return pred


def write_predictions(pred_df):
    """Persist the predictions per window (overwrites a previous run)."""
    pred_df.write.mode("overwrite").parquet(PREDICTIONS_OUTPUT)
    print(f"\nWrote per-window predictions to {PREDICTIONS_OUTPUT}")


def main():
    spark = build_spark()
    try:
        print("=" * 64)
        print("PREDICT BATCH (Phase 7)")
        print("=" * 64)
        print("""
  This batch was generated AFTER training, with a fresh random seed and a later
  date range than the training data. It is NOT the Phase-6 test split reused --
  it is a simulated new batch that arrived once the model was already built,
  so it tests genuine generalization, not memorization of held-out rows.
""")

        labeled = run_extract_and_transform(
            spark, NEW_BATCH_LOG, NEW_BATCH_GROUND_TRUTH
        )

        model = load_model(MODEL_PATH)
        pred = predict(model, labeled)

        m = evaluate(pred)
        print("\n--- predictions vs. the new batch's own ground truth ---")
        print("  confusion matrix (rows=actual, cols=predicted):")
        print(f"                    predicted normal   predicted anomalous")
        print(f"  actually normal      TN={m['tn']:<5}          FP={m['fp']:<5}")
        print(f"  actually anomalous   FN={m['fn']:<5}          TP={m['tp']:<5}")
        print(f"  total rows: {m['total']}")
        print(f"  accuracy : {m['accuracy'] * 100:.2f}%")
        print(f"  precision: {m['precision']:.4f}")
        print(f"  recall   : {m['recall']:.4f}")
        print(f"  F1       : {m['f1']:.4f}")
        print("""
  Honest note: this is the real generalization test. If these numbers are
  lower than Phase 6's perfect synthetic-separation scores, that is expected
  and valuable -- it means the model is being measured on data it could not
  have memorized. No re-tuning is performed here.
""")

        write_predictions(pred)
        print("\nDone.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
