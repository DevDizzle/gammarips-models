#!/usr/bin/env python3
# FILE: inference_pipeline.py
"""
Compiles the ProfitScout Batch Prediction Pipeline.
"""

from kfp import dsl, compiler
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)
import subprocess
import os

# ─────────────────── Config ───────────────────
PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"
PIPELINE_ROOT = f"gs://{PROJECT_ID}-pipeline-artifacts/inference"
PREDICTOR_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/profitscout-predictor:latest"
)
EVALUATOR_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/profit-scout-repo/profitscout-evaluator:latest"
)
# Note: Inference doesn't typically output a model artifact, but CustomJob reqs a base_output_dir
BASE_OUTPUT_DIR = f"{PIPELINE_ROOT}/job-output"

# ───────────────── Components ─────────────────
@dsl.container_component
def prediction_task(
    project: str,
    source_table: str,
    destination_table: str,
    model_base_dir: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=PREDICTOR_IMAGE_URI,
        command=["python3", "main.py"],
        args=[
            "--project-id", project,
            "--source-table", source_table,
            "--destination-table", destination_table,
            "--model-base-dir", model_base_dir,
        ],
    )

@dsl.container_component
def evaluation_task(
    project: str,
    predictions_table: str,
    price_table: str,
    performance_table: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=EVALUATOR_IMAGE_URI,
        command=["python3", "main.py"],
        args=[
            "--project-id", project,
            "--predictions-table", predictions_table,
            "--price-table", price_table,
            "--performance-table", performance_table,
        ],
    )

# Using create_custom_training_job_from_component allows us to run this as a "Custom Job"
prediction_op = create_custom_training_job_from_component(
    component_spec=prediction_task,
    display_name="profitscout-batch-prediction",
    machine_type="n1-standard-8",
    replica_count=1,
    base_output_directory=BASE_OUTPUT_DIR,
)

evaluation_op = create_custom_training_job_from_component(
    component_spec=evaluation_task,
    display_name="profitscout-prediction-evaluation",
    machine_type="n1-standard-4", # Lighter machine needed for SQL checks
    replica_count=1,
    base_output_directory=BASE_OUTPUT_DIR,
)

# ───────────────── Pipeline ─────────────────
@dsl.pipeline(
    name="profitscout-daily-prediction-pipeline",
    description="Generate High Gamma predictions and evaluate past performance.",
    pipeline_root=PIPELINE_ROOT,
)
def inference_pipeline(
    project: str = PROJECT_ID,
    source_table: str = "profit_scout.price_data",
    destination_table: str = "profit_scout.daily_predictions",
    performance_table: str = "profit_scout.prediction_performance",
    model_base_dir: str = "gs://profitscout-lx6bb-pipeline-artifacts/production/model", 
):
    pred_step = prediction_op(
        project=project,
        source_table=source_table,
        destination_table=destination_table,
        model_base_dir=model_base_dir,
    )
    
    # Run evaluation after prediction (conceptually verifies YESTERDAY's predictions using TODAY's data)
    # We pass 'destination_table' as the 'predictions_table' to read from.
    eval_step = evaluation_op(
        project=project,
        predictions_table=destination_table,
        price_table=source_table,
        performance_table=performance_table,
    ).after(pred_step)

# ───────────────── Compile and Upload ─────────────────
if __name__ == "__main__":
    local_path = "pipelines/compiled/inference_pipeline.json"
    gcs_path = f"{PIPELINE_ROOT}/inference_pipeline.json"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path=local_path,
    )
    print(f"✓ Compiled to {local_path}")

    try:
        subprocess.run(["gsutil", "cp", local_path, gcs_path], check=True)
        print(f"✓ Uploaded to {gcs_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ GCS upload failed: {e}")