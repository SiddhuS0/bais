import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st

def forecast_metric(df, column_name):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df[column_name]

    last_date = df['ds'].max()
    if last_date.month != 12:
        last_month = pd.to_datetime(f"{last_date.year}-12-01")
        last_value = df[df['ds'] == last_date]['y'].iloc[0]
        new_row = pd.DataFrame([{'ds': last_month, 'y': last_value}])
        df = pd.concat([df, new_row], ignore_index=True)

    weekly_df = df[['ds', 'y']].groupby(pd.Grouper(key='ds', freq='W')).sum().reset_index()

    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
    model.fit(weekly_df)

    future = model.make_future_dataframe(periods=26, freq='W')
    forecast = model.predict(future)

    return forecast, model

def plot_forecast(model, forecast, historical_df, column_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    model.plot(forecast, ax=ax)

    cutoff = model.history['ds'].max()
    ax.axvline(cutoff, color='green', linestyle='--', label='Forecast Start')

    latest_date = pd.to_datetime(historical_df['date']).max()
    future_forecast = forecast[forecast['ds'] > latest_date]
    dip_row = future_forecast.loc[future_forecast['yhat'].idxmin()]
    dip_date, dip_value = dip_row['ds'], dip_row['yhat']

    ax.plot(dip_date, dip_value, 'go')
    ax.annotate(
        f" Dip: ₹{dip_value:.2f}",
        xy=(dip_date, dip_value),
        xytext=(dip_date, dip_value + dip_value * 0.05),
        arrowprops=dict(arrowstyle="->", color='black'),
        fontsize=10,
        color='black'
    )

    ax.set_title(f"{column_label} Forecast (Next 6 Months)", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(column_label, fontsize=12)
    ax.legend()

    return fig
