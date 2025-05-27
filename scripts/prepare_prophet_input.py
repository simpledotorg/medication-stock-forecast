#!/usr/bin/env python3
import os
import pandas as pd

# 1. Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # scripts/
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')              # data/
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')         # outputs/

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_CSV      = os.path.join(DATA_DIR,  'metabase_report.csv')
PROPHET_CSV  = os.path.join(OUTPUT_DIR, 'prophet_input.csv')

# 2. Load your Metabase export
df = pd.read_csv(RAW_CSV)

# 3. Parse 'month_date' automatically
df['month_date'] = pd.to_datetime(df['month_date'], utc=True).dt.tz_convert(None)

# 4. Dynamically identify your class columns
class_cols = [col for col in df.columns if col.startswith('Hypertension:')]

# 5. Unpivot (melt) into long format
df_long = df.melt(
    id_vars=['month_date', 'facility'],
    value_vars=class_cols,
    var_name='drug_class',
    value_name='y'
)

# 6. Rename for Prophet
df_long = df_long.rename(columns={'month_date': 'ds'})

# 7. Write out your Prophet-ready CSV
df_long.to_csv(PROPHET_CSV, index=False)

print(f"Saved Prophet input to {PROPHET_CSV} with columns: {list(df_long.columns)}")
