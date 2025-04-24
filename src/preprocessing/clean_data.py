import pandas as pd
import os

# Define required columns based on your real dataset
REQUIRED_COLUMNS = [
    'date', 'revenue', 'net_profit', 'cogs', 'operating_expense',
    'marketing_cost', 'investment_cost', 'total_customers', 'orders'
]

def load_and_validate_data(df):
    try:
        # Normalize column names
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        
        # Check for required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Your dataset is missing these required columns: {', '.join(missing_cols)}\n"
                f"Uploaded columns: {', '.join(original_columns)}\n"
                f"Required columns: {', '.join(REQUIRED_COLUMNS)}"
            )

        # Date validation
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        invalid_dates = df[df['date'].isna()]
        if not invalid_dates.empty:
            raise ValueError(
                f"Found {len(invalid_dates)} invalid date format(s). "
                "Dates must be in YYYY-MM-DD format."
            )

        # Drop invalid/missing data
        df = df.dropna(subset=['date']).dropna()
        
        if df.empty:
            raise ValueError(
                "No valid data remaining after cleaning. "
                "Please check for missing values in required columns."
            )
            
        return df

    except Exception as e:
        raise ValueError(str(e)) from e

def save_clean_data(df, output_path='data/processed/cleaned_data.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Saving cleaned data...")
    df.to_csv(output_path, index=False)
    print(f"📦 Cleaned data saved to: {output_path}")
