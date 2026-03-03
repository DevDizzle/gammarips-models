import os
import sys
import pandas as pd
from google.cloud import bigquery
import numpy as np

# Add src to python path to import existing feature engineering logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering.processing import get_price_history, generate_technical_features

def main():
    print("Fetching bullish predictions and evaluating outcomes...")
    client = bigquery.Client(project="profitscout-lx6bb")
    
    query = """
    WITH NextDayPrices AS (
      SELECT
        ticker,
        date AS current_date,
        LEAD(high) OVER (PARTITION BY ticker ORDER BY date ASC) as next_day_high,
        LEAD(low) OVER (PARTITION BY ticker ORDER BY date ASC) as next_day_low,
        LEAD(date) OVER (PARTITION BY ticker ORDER BY date ASC) as next_day_date
      FROM
        `profitscout-lx6bb.profit_scout.price_data`
    ),
    Evaluation AS (
      SELECT 
        p.ticker,
        p.date as prediction_date,
        p.prediction,
        p.close,
        p.atrr_14,
        ndp.next_day_high,
        CASE 
          WHEN p.prediction = 1 AND ndp.next_day_high >= (p.close + (0.5 * p.atrr_14)) THEN 1
          ELSE 0
        END as target_hit
      FROM 
        `profitscout-lx6bb.profit_scout.daily_predictions` p
      JOIN 
        NextDayPrices ndp 
        ON p.ticker = ndp.ticker AND p.date = ndp.current_date
      WHERE
        p.prediction = 1 AND ndp.next_day_date IS NOT NULL
    )
    SELECT * FROM Evaluation
    """
    
    df_eval = client.query(query).to_dataframe()
    print(f"Found {len(df_eval)} bullish predictions to analyze.")
    
    all_features = []
    
    print("Re-calculating features for each prediction date...")
    for idx, row in df_eval.iterrows():
        ticker = row['ticker']
        pred_date = row['prediction_date']
        target_hit = row['target_hit']
        
        # Get price history up to the prediction date
        df_prices = get_price_history(ticker, str(pred_date))
        if df_prices.empty:
            continue
            
        features = generate_technical_features(df_prices)
        if not features:
            continue
            
        features['ticker'] = ticker
        features['prediction_date'] = pred_date
        features['target_hit'] = target_hit
        all_features.append(features)
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(df_eval)}...")
            
    df_features = pd.DataFrame(all_features)
    
    print("\\n" + "="*50)
    print("FEATURE COMPARISON: WINNERS (1) vs LOSERS (0)")
    print("="*50)
    
    # Calculate means for winners and losers
    winners = df_features[df_features['target_hit'] == 1].drop(columns=['ticker', 'prediction_date', 'target_hit'])
    losers = df_features[df_features['target_hit'] == 0].drop(columns=['ticker', 'prediction_date', 'target_hit'])
    
    # Calculate percentage difference between winners and losers
    mean_winners = winners.mean()
    mean_losers = losers.mean()
    
    diff = ((mean_winners - mean_losers) / mean_losers.abs()) * 100
    
    comparison = pd.DataFrame({
        'Winner_Mean': mean_winners,
        'Loser_Mean': mean_losers,
        'Diff_Pct': diff
    })
    
    # Sort by absolute difference to find the most distinguishing features
    comparison['Abs_Diff_Pct'] = comparison['Diff_Pct'].abs()
    comparison = comparison.sort_values(by='Abs_Diff_Pct', ascending=False).drop(columns=['Abs_Diff_Pct'])
    
    # Format the output nicely
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(comparison.head(15))
    
    print("\\n" + "="*50)
    print("Top 3 Features Where Winners Were HIGHER than Losers:")
    print(comparison[comparison['Diff_Pct'] > 0].head(3))
    
    print("\\nTop 3 Features Where Winners Were LOWER than Losers:")
    print(comparison[comparison['Diff_Pct'] < 0].head(3))

if __name__ == "__main__":
    main()
