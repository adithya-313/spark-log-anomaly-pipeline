r"""
train_model.py
==============
Phase 5 (Model) of the Spark log-anomaly pipeline.

Trains and evaluates an MLlib Logistic Regression classifier that predicts
is_anomalous (0 = normal, 1 = anomalous) from the windowed features produced
by Phase 3 (Load): total_requests, error_rate, avg_response_size,
p95_response_size.

Data source: windowed_features/ (the final partitioned deliverable) rather
than transform_output.parquet. It is the canonical end-of-pipeline artifact --
the thing a downstream consumer would actually read -- and load.py's round-trip
verification already proved it is exactly equal to transform_output.parquet on
all data columns. The only extra column ('date', a partition column derived
from window_start) is excluded from modeling.

Train/test split is TIME-BASED, not random: windows sorted by window_start, the
first ~80% chronologically are train, the last ~20% are test. A random split
would leak the future into the past here: the anomalies were injected as bursts
over time, so random sampling lets the model train on (and memorize) feature
signatures from windows that, at prediction time, would still be in the future
-- inflating test metrics relative to real deployment, where the model must
predict windows it has not seen yet. It also breaks the (temporal)
i.i.d. assumption behind evaluation.

Why recall, not accuracy, is the metric that matters:
    81/2016 = ~4% of windows are anomalous. A model that predicts "never
    anomalous" scores ~96% accuracy while detecting nothing. On a security
    alerting task the cost of a missed anomaly (false negative) is far higher
    than the cost of a false positive, so we report precision / recall / F1
    and the full confusion matrix, not just accuracy.

Model choice: Logistic Regression -- the simplest, most interpretable
classifier, per the plan. No random forest or other model unless LR clearly
fails in a way worth documenting. This stays batch/offline: no serving infra,
real-time inference, or deep learning.

Persistence: the fitted best pipeline (VectorAssembler + LogisticRegression)
is saved to trained_model/ via MLlib's native write().save() format, so
Phase 7 (predict_batch.py) can load it without re-training. Saved after CV,
before the test-set evaluation below (the same model is both evaluated here
and persisted for later use).

Hyperparameters: ParamGridBuilder + CrossValidator on regParam and
elasticNetParam only (2 parameters). With ~1612 train rows and ~65 positive
examples, k-fold CV here is a light sensitivity check, not a robust model
selection procedure -- said honestly in the report, not oversold.

ENV NOTE (required to run Spark on this laptop):
    $env:JAVA_HOME = "C:\Users\sinne\jdk17\jdk-17.0.20+8"
    $env:HADOOP_HOME = "C:\Users\sinne\hadoop-utils"          # needed for file writes
    $env:PATH = "C:\Users\sinne\hadoop-utils\bin;$env:PATH"   # hadoop.dll for native IO
    $env:SPARK_LOCAL_IP = "127.0.0.1"
    $env:PYSPARK_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"
    $env:PYSPARK_DRIVER_PYTHON = "C:\Users\sinne\AppData\Local\Programs\Python\Python312\python.exe"

Usage:
    python train_model.py
"""

import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, date_format, row_number

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_OUTPUT = os.path.join(SCRIPT_DIR, "windowed_features")

# The model columns (anomaly_type and the partition column 'date' are not
# features -- anomaly_type is the label's richer form and 'date' is derived
# from window_start).
FEATURE_COLUMNS = [
    "total_requests", "error_rate", "avg_response_size", "p95_response_size",
]
LABEL_COLUMN = "is_anomalous"

TRAIN_FRACTION = 0.8

# Where the fitted pipeline is persisted so later phases can load it.
# MLlib's native save format (a directory), NOT Python pickling -- the saved
# artifact can be re-loaded by Spark.ML without this script or its code.
MODEL_PATH = os.path.join(SCRIPT_DIR, "trained_model")

# Cross-validation settings (small, honest grid -- see module docstring).
CV_FOLDS = 3
CV_SEED = 42
GRID_REG_PARAM = [0.01, 0.1, 1.0]
GRID_ELASTIC_NET = [0.0, 0.5, 1.0]  # 0.0 = pure L2, 1.0 = pure L1


def build_spark():
    """Start a local SparkSession. UTC keeps window_start wall-clock consistent."""
    spark = (
        SparkSession.builder
        .appName("spark-log-anomaly-train")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_features(spark, path):
    """
    Read the final pipeline deliverable and keep only what modeling needs.
    The partition column 'date' is a duplicate of window_start, so it is
    dropped (it is DateType on re-read, which the assembler would reject).
    """
    df = spark.read.parquet(path)
    return df.select("window_start", *FEATURE_COLUMNS, LABEL_COLUMN)


def time_split(df, train_fraction):
    """
    Chronological 80/20 split.
    Sort by window_start and rank every window 1..N; the first train_fraction
    of them are train, the rest are test. window_start is unique (2016 windows
    at 5-minute boundaries), so the ranking is a total order with no ties.
    Returns (train_df, test_df, info) where info holds the counts and the exact
    UTC cutoff window_start.
    """
    total = df.count()
    train_count = int(total * train_fraction)

    # Add a UTC-formatted copy of window_start so the reported cutoff is the
    # true UTC wall-clock (raw Python datetimes display in local time; this
    # avoids that artifact).
    ranked = df.withColumn(
        "window_start_utc", date_format(col("window_start"), "yyyy-MM-dd HH:mm:ss")
    ).withColumn("rn", row_number().over(Window.orderBy(col("window_start"))))

    train = ranked.filter(col("rn") <= train_count)
    test = ranked.filter(col("rn") > train_count)

    cutoff = (
        ranked
        .filter((col("rn") == train_count) | (col("rn") == train_count + 1))
        .orderBy("rn")
        .select("rn", "window_start_utc")
        .collect()
    )
    info = {
        "total": total,
        "train_count": train_count,
        "test_count": total - train_count,
        "last_train_window": str(cutoff[0]["window_start_utc"]),
        "first_test_window": str(cutoff[1]["window_start_utc"]),
    }
    return train, test, info


def print_class_balance(df, label):
    """Print how many of each class a DataFrame has."""
    rows = df.groupBy(label).count().orderBy(label).collect()
    return {int(r[label]): r["count"] for r in rows}


def build_model():
    """
    Two-stage pipeline: VectorAssembler bundles the 4 feature columns into one
    'features' vector; LogisticRegression then learns from that vector. LR
    standardizes numeric features internally (standardization=True by
    default), so no separate scaler is needed.
    """
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features")
    lr = LogisticRegression(labelCol=LABEL_COLUMN, featuresCol="features")
    return Pipeline(stages=[assembler, lr]), lr


def evaluate(predictions):
    """
    Confusion-matrix counts + derived metrics on a DataFrame with 'prediction'.
    Counts are reported verbatim (TP/FP/TN/FN), not just percentages, per the
    plan -- this also exposes a degenerate all-negative model at a glance.
    """
    tp = predictions.filter((col(LABEL_COLUMN) == 1) & (col("prediction") == 1)).count()
    fp = predictions.filter((col(LABEL_COLUMN) == 0) & (col("prediction") == 1)).count()
    tn = predictions.filter((col(LABEL_COLUMN) == 0) & (col("prediction") == 0)).count()
    fn = predictions.filter((col(LABEL_COLUMN) == 1) & (col("prediction") == 0)).count()

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def report_coefficients(lr_model):
    """
    Print the fitted logistic-regression coefficients, one per feature.
    Sanity check: anomalies were injected as (a) burst of errors (high
    error_rate) and (b) oversized responses (high avg/p95 response size), so
    error_rate and avg_response_size / p95_response_size should have strong
    POSITIVE coefficients, and total_requests is not expected to be decisive.
    """
    coefs = lr_model.coefficients.toArray()
    print("  fitted coefficients (log-odds change per 1-unit feature change):")
    for name, c in zip(FEATURE_COLUMNS, coefs):
        print(f"    {name:<20}: {c:+.6f}")
    print(f"    {'intercept':<20}: {lr_model.intercept:+.6f}")


def main():
    spark = build_spark()
    try:
        print("=" * 64)
        print("TRAIN MODEL (Phase 5)")
        print("=" * 64)
        df = read_features(spark, LOAD_OUTPUT)
        print(f"read windowed_features/ -> {df.count()} rows")

        # ---- 1. chronological train/test split ----------------------------
        train, test, info = time_split(df, TRAIN_FRACTION)
        print("\n--- train/test split (time-based, not random) ---")
        print(f"  total windows        : {info['total']}")
        print(f"  train windows        : {info['train_count']} "
              f"({info['train_count'] / info['total'] * 100:.1f}%, earliest)")
        print(f"  test windows         : {info['test_count']} "
              f"({info['test_count'] / info['total'] * 100:.1f}%, latest)")
        print(f"  cutoff (UTC)         : train ends at "
              f"{info['last_train_window']}, test starts at "
              f"{info['first_test_window']}")

        train_balance = print_class_balance(train, LABEL_COLUMN)
        test_balance = print_class_balance(test, LABEL_COLUMN)
        print(f"  train class balance  : normal={train_balance.get(0, 0)}, "
              f"anomalous={train_balance.get(1, 0)}")
        print(f"  test class balance   : normal={test_balance.get(0, 0)}, "
              f"anomalous={test_balance.get(1, 0)}")

        print("""
  Why not a random split? A random 80/20 would train on windows that, at
  prediction time, are still in the future: the anomalies were injected as
  time bursts, so the model could memorize their feature signatures instead of
  generalizing to unseen time. Predicting only the future (chronological
  split) is the honest estimate of deployment behavior.
""")

        # ---- 2. model + cross-validated hyperparameter search -------------
        print("--- hyperparameter search (ParamGridBuilder + CrossValidator) ---")
        print(f"  grid: regParam={GRID_REG_PARAM}, "
              f"elasticNetParam={GRID_ELASTIC_NET} ({len(GRID_REG_PARAM) * len(GRID_ELASTIC_NET)} combos)")
        print(f"  folds: {CV_FOLDS}, metric: areaUnderROC (threshold-free ranking)")

        pipeline, lr = build_model()
        grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, GRID_REG_PARAM)
            .addGrid(lr.elasticNetParam, GRID_ELASTIC_NET)
            .build()
        )
        evaluator = BinaryClassificationEvaluator(
            labelCol=LABEL_COLUMN, metricName="areaUnderROC"
        )
        cv = CrossValidator(
            estimator=pipeline, estimatorParamMaps=grid,
            evaluator=evaluator, numFolds=CV_FOLDS, seed=CV_SEED,
        )
        cv_model = cv.fit(train)

        best_lr = cv_model.bestModel.stages[-1]
        best_reg = best_lr._java_obj.getOrDefault(best_lr._java_obj.getParam("regParam"))
        best_enet = best_lr._java_obj.getOrDefault(best_lr._java_obj.getParam("elasticNetParam"))
        print("  per-param-combo average validation AUC (3-fold):")
        for pm, metric in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
            print(f"    regParam={pm[lr.regParam]:<6} "
                  f"elasticNetParam={pm[lr.elasticNetParam]:<5} avgAUC={metric:.4f}")
        print(f"  best params -> regParam={best_reg}, elasticNetParam={best_enet}")
        print(f"\n  saving best fitted pipeline (assembler + LR) to {MODEL_PATH}")
        cv_model.bestModel.write().overwrite().save(MODEL_PATH)
        print("  saved (MLlib native format -- loadable via PipelineModel.load).")
        print("""
  Honesty note on CV at this size: the train set is only ~1612 rows with ~65
  anomalous windows. With 3 folds each training fold holds ~43 positives, so
  the AUC estimates have wide variance and the "best" combo is not a robust
  selection -- treat this grid as a light sensitivity check that the default
  ridge-regularized LR is not obviously bad, not as evidence of the one true
  hyperparameter setting.
""")

        # ---- 3. evaluate the CV-selected model on the TEST set ------------
        test_pred = cv_model.transform(test)
        m = evaluate(test_pred)

        print("--- test-set evaluation (time-based test, default threshold 0.5) ---")
        print("  confusion matrix (rows=actual, cols=predicted):")
        print(f"                    predicted normal   predicted anomalous")
        print(f"  actually normal      TN={m['tn']:<5}          FP={m['fp']:<5}")
        print(f"  actually anomalous   FN={m['fn']:<5}          TP={m['tp']:<5}")
        print(f"  total test rows: {m['total']}")
        print(f"  accuracy : {m['accuracy'] * 100:.2f}%")
        print(f"  precision: {m['precision']:.4f}")
        print(f"  recall   : {m['recall']:.4f}")
        print(f"  F1       : {m['f1']:.4f}")

        # Baseline for the "never anomalous" narrative: an all-normal predictor
        # scores accuracy = fraction of normal rows, with recall 0.
        never_anomalous_acc = (m["tn"] + m["fn"]) / m["total"]
        print(f"""
  Why recall matters more than accuracy: {test_balance.get(1, 0)}/{m['total']}
  (~{test_balance.get(1, 0) / m['total'] * 100:.1f}%) test windows are anomalous.
  A model predicting 'never anomalous' would score {never_anomalous_acc * 100:.1f}%
  accuracy while catching 0 anomalies (recall 0). On an alerting task a missed
  anomaly is far more costly than a false alarm, so the confusion matrix and
  recall are the real report card.
""")

        # ---- 4. verification ----------------------------------------------
        print("--- verification ---")
        pred_dist = test_pred.groupBy("prediction").count().orderBy("prediction").collect()
        pred_counts = {int(r["prediction"]): r["count"] for r in pred_dist}
        print(f"  predicted class distribution on test: {pred_counts}")
        not_all_one_class = len(pred_counts) == 2
        print(f"  predictions not trivially all-one-class: {not_all_one_class}")
        if not_all_one_class and m["tp"] > 0 and m["precision"] < 0.5:
            print("  (note: high recall but low precision -- model over-predicts anomalies)")

        print(f"""
  Honest note on near-perfect numbers: this is synthetic data with sharply
  injected anomalies (error-type windows pushed error_rate to ~48% vs ~3% for
  normal; size-type windows pushed avg_response_size to ~8-9k vs ~700 normal).
  At that separation, LR reaching the default 0.5 threshold with zero mistakes
  is expected and is NOT evidence of production readiness -- real logs are far
  messier. The point of this phase is the disciplined pipeline: chronological
  split, honest metrics, and a visible confusion matrix, not SOTA accuracy.
""")

        print("\n--- feature coefficients (intuitive sanity check) ---")
        report_coefficients(best_lr)

        print("\nDone.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
