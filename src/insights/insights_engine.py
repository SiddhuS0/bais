import pandas as pd
import numpy as np

def generate_business_insights(df):
    insights = {}

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    # Top Performing & Loss Months
    monthly_revenue = df.groupby('month')['revenue'].sum()
    top_month = monthly_revenue.idxmax().strftime('%B %Y')
    loss_month = monthly_revenue.idxmin().strftime('%B %Y')
    insights['top_month'] = f"Top-performing month: {top_month} with revenue of {monthly_revenue.max():,.2f}"
    insights['loss_month'] = f"Lowest-performing month: {loss_month} with revenue of {monthly_revenue.min():,.2f}"

    # Cost Optimization Insight
    df['total_cost'] = df['operating_expense'] + df['marketing_cost'] + df['cogs']
    df['cost_to_revenue_ratio'] = df['total_cost'] / df['revenue']
    high_cost_entries = df[df['cost_to_revenue_ratio'] > 0.8]  # high cost warning
    if not high_cost_entries.empty:
        insights['cost_warning'] = f"High cost-to-revenue ratio in {len(high_cost_entries)} records. Review cost control strategies."

    # Customer Acquisition Efficiency
    df['customer_acquisition_cost'] = df['marketing_cost'] / df['new_customers_acquired'].replace(0, np.nan)
    avg_cac = df['customer_acquisition_cost'].mean()
    insights['avg_cac'] = f"Average customer acquisition cost: {avg_cac:,.2f}"

    # Profit Margin
    df['profit_margin'] = ((df['revenue'] - df['cogs']) / df['revenue']) * 100
    avg_margin = df['profit_margin'].mean()
    insights['avg_margin'] = f"Average profit margin: {avg_margin:.2f}%"

    # Best Region
    region_performance = df.groupby('region')['revenue'].sum()
    best_region = region_performance.idxmax()
    insights['best_region'] = f"Top performing region: {best_region} with total revenue of {region_performance.max():,.2f}"

    # Product Performance
    product_performance = df.groupby('product_name')['units_sold'].sum()
    best_product = product_performance.idxmax()
    insights['best_product'] = f"Best-selling product: {best_product} with {product_performance.max()} units sold"

    # Profit vs Employee Count Insight
    correlation = df['employee_count'].corr(df['net_profit'])
    if correlation < 0:
        insights['employee_profit'] = "Higher employee count is negatively correlated with net profit. Consider optimizing team size."
    elif correlation > 0:
        insights['employee_profit'] = "Higher employee count positively contributes to net profit."
    else:
        insights['employee_profit'] = "No significant relationship found between employee count and profit."

    return insights
