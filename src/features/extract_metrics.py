import pandas as pd

def extract_features(df):
    df = df.copy()

    # Calculate derived metrics only if base columns are present
    if 'revenue' in df.columns and 'net_profit' in df.columns:
        df['Profit_Margin (%)'] = (df['net_profit'] / df['revenue']) * 100

    if 'net_profit' in df.columns and 'investment_cost' in df.columns:
        df['ROI (%)'] = (df['net_profit'] / df['investment_cost']) * 100

    # Revenue & Profit Growth
    if 'revenue' in df.columns:
        df['Revenue_Growth (%)'] = df['revenue'].pct_change() * 100

    if 'net_profit' in df.columns:
        df['Profit_Growth (%)'] = df['net_profit'].pct_change() * 100

    # Break-Even Point (BEP) - needs operating + variable costs
    if 'operating_expense' in df.columns and 'cogs' in df.columns and 'revenue' in df.columns:
        try:
            df['BEP'] = df.apply(
                lambda row: row['operating_expense'] / (row['revenue'] - row['cogs']) 
                if (row['revenue'] - row['cogs']) != 0 else 0, axis=1)
        except Exception as e:
            df['BEP'] = 0  # fallback

    return df
