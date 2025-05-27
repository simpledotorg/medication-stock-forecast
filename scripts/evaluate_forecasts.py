#!/usr/bin/env python3
import os
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# 1. Determine script base directory
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# 2. Define paths
OUTPUT_DIR  = os.path.abspath(os.path.join(BASE_DIR, '..', 'outputs'))
FORECAST_CSV = os.path.join(OUTPUT_DIR, 'forecast_results_12mo.csv')
ACTUALS_CSV  = os.path.join(OUTPUT_DIR, 'actuals.csv')
METRICS_CSV  = os.path.join(OUTPUT_DIR, 'forecast_evaluation_metrics.csv')

# 3. Load forecast (long form)
df_f = pd.read_csv(FORECAST_CSV, parse_dates=['ds'])
df_f['ds'] = pd.to_datetime(df_f['ds'], utc=True).dt.tz_localize(None)

# 4. Load actuals (wide), melt to long form
df_a_wide = pd.read_csv(ACTUALS_CSV, parse_dates=['month_date'])
df_a_wide['month_date'] = pd.to_datetime(df_a_wide['month_date'], utc=True).dt.tz_localize(None)
# Identify class columns
class_cols = [c for c in df_a_wide.columns if c.startswith('Hypertension:')]
df_a = (
    df_a_wide
      .melt(id_vars=['month_date','facility'], value_vars=class_cols,
            var_name='drug_class', value_name='actual')
      .rename(columns={'month_date':'ds'})
)

# 5. Merge on ds, facility, drug_class
df = pd.merge(df_f, df_a, on=['ds','facility','drug_class'], how='inner')

# 6. Compute metrics per facility+drug_class
metrics = []
for (fac, drug), grp in df.groupby(['facility','drug_class']):
    y_true = grp['actual']
    y_pred = grp['yhat']
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = math.sqrt(mse)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    metrics.append({
        'facility':   fac,
        'drug_class': drug,
        'MAE':        round(mae, 2),
        'RMSE':       round(rmse, 2),
        'MAPE (%)':   round(mape, 2)
    })

# 7. Overall metrics
y_true_all = df['actual']
y_pred_all = df['yhat']
metrics.append({
    'facility':   'All',
    'drug_class': 'All',
    'MAE':        round(mean_absolute_error(y_true_all, y_pred_all), 2),
    'RMSE':       round(math.sqrt(mean_squared_error(y_true_all, y_pred_all)), 2),
    'MAPE (%)':   round(mean_absolute_percentage_error(y_true_all, y_pred_all) * 100, 2)
})

# 8. Save
pd.DataFrame(metrics).to_csv(METRICS_CSV, index=False)
print(f"Saved evaluation metrics to {METRICS_CSV}")
