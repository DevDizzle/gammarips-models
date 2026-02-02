#!/usr/bin/env python3
"""
ProfitScout Evaluation Script
Verifies the performance of historical predictions against actual price movements.
Target: Next Day Price Move > 0.5 * ATR(14)
"""
import argparse
import logging
import pandas as pd
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)

def evaluate_predictions(project_id: str, predictions_table: str, price_table: str, performance_table: str):
    client = bigquery.Client(project=project_id)
    
    # 1. Find predictions that haven't been evaluated yet
    # We look for (ticker, date) pairs in predictions_table that are NOT in performance_table
    # AND where we have a subsequent price in price_table.
    
    query = f"""
    WITH Predictions AS (
        SELECT 
            ticker, 
            date AS pred_date, 
            contract_type, 
            prob, 
            prediction,
            close AS signal_close, 
            atrr_14
        FROM `{project_id}.{predictions_table}`
    ),
    AlreadyEvaluated AS (
        SELECT ticker, pred_date 
        FROM `{project_id}.{performance_table}`
    ),
    Pending AS (
        SELECT p.*
        FROM Predictions p
        LEFT JOIN AlreadyEvaluated e 
        ON p.ticker = e.ticker AND p.pred_date = e.pred_date
        WHERE e.ticker IS NULL
    ),
    NextDayPrices AS (
        SELECT 
            pend.ticker,
            pend.pred_date,
            prices.date AS next_date,
            prices.adj_close AS next_close,
            ROW_NUMBER() OVER(PARTITION BY pend.ticker, pend.pred_date ORDER BY prices.date ASC) as rn
        FROM Pending pend
        JOIN `{project_id}.{price_table}` prices
        ON pend.ticker = prices.ticker AND prices.date > pend.pred_date
    )
    SELECT 
        p.ticker,
        p.pred_date,
        nd.next_date AS outcome_date,
        p.contract_type,
        p.prob,
        p.signal_close,
        nd.next_close,
        p.atrr_14,
        (nd.next_close - p.signal_close) AS price_delta,
        
        -- Logic: Did it hit the High Gamma Target? (> 0.5 * ATR)
        CASE 
            WHEN p.contract_type = 'CALL' THEN 
                (nd.next_close - p.signal_close) > (0.5 * p.atrr_14)
            WHEN p.contract_type = 'PUT' THEN 
                (p.signal_close - nd.next_close) > (0.5 * p.atrr_14)
            ELSE FALSE
        END AS target_met
        
    FROM Pending p
    JOIN NextDayPrices nd
    ON p.ticker = nd.ticker AND p.pred_date = nd.pred_date
    WHERE nd.rn = 1  -- Only take the very next available trading day
    """
    
    logging.info("Querying for pending evaluations...")
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        # If table doesn't exist, we might get an error. 
        # But we assume the tables exist. If performance_table is missing, the query fails.
        # We handle table creation below, so let's try a simpler check first or just assume it fails if empty.
        logging.error("Query failed: %s", e)
        return

    if df.empty:
        logging.info("No pending predictions found to evaluate.")
        return
        
    logging.info("Found %d predictions to evaluate.", len(df))
    
    # 2. Save results to Performance Table
    # Schema matches the query output
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("pred_date", "DATE"),
            bigquery.SchemaField("outcome_date", "DATE"),
            bigquery.SchemaField("contract_type", "STRING"),
            bigquery.SchemaField("prob", "FLOAT64"),
            bigquery.SchemaField("signal_close", "FLOAT64"),
            bigquery.SchemaField("next_close", "FLOAT64"),
            bigquery.SchemaField("atrr_14", "FLOAT64"),
            bigquery.SchemaField("price_delta", "FLOAT64"),
            bigquery.SchemaField("target_met", "BOOLEAN"),
        ]
    )
    
    table_ref = f"{project_id}.{performance_table}"
    
    # Create table if it doesn't exist (handled implicitly by load_table_from_dataframe if valid, 
    # but schema ensures it's created correctly).
    try:
        client.load_table_from_dataframe(df, table_ref, job_config=job_config).result()
        logging.info("Successfully saved evaluation results to %s", performance_table)
    except Exception as e:
        logging.error("Failed to save evaluation results: %s", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--predictions-table", required=True)
    parser.add_argument("--price-table", required=True)
    parser.add_argument("--performance-table", required=True)
    
    args = parser.parse_args()
    
    evaluate_predictions(
        args.project_id, 
        args.predictions_table, 
        args.price_table, 
        args.performance_table
    )

if __name__ == "__main__":
    main()
