#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time series analysis and forecasting models.

@author: lefteris
"""

import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_log_error,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


def seasonality_fit(df):
    """Calculate seasonal factors using classical decomposition."""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    wide_df = df.pivot_table(index='month', columns='year', values='total_amount', aggfunc='mean')

    # Add column with row (month) averages
    wide_df['row_average'] = wide_df.mean(axis=1)

    # Calculate seasonal factors
    wide_df['seasonal_factor'] = wide_df['row_average'] / wide_df['row_average'].mean()

    return wide_df.seasonal_factor.reset_index()


def calculate_ma_and_errors_season(order, timeseries, season_coef):
    """Calculate Moving Average with seasonality adjustment and error metrics."""
    timeseries = timeseries.copy()
    timeseries['month'] = pd.to_datetime(timeseries.index)

    merged_df = pd.merge(
        timeseries, season_coef,
        left_on=timeseries['month'].dt.month, right_on='month',
        how='left'
    ).drop(columns=['month', 'month_y']).rename({'month_x': 'month'}, axis=1)

    # Store original values before deseasonalizing
    merged_df['original_amount'] = merged_df['total_amount'].copy()

    # Deseasonalize for MA calculation
    merged_df['deseasonalized'] = merged_df['total_amount'] / merged_df['seasonal_factor']

    # Calculate MA on deseasonalized data
    ma = merged_df['deseasonalized'].rolling(window=order).mean()
    merged_df['prediction'] = ma

    # Re-seasonalize the prediction
    merged_df['MA_Seasonalized'] = merged_df['prediction'] * merged_df['seasonal_factor']

    merged_df = merged_df.set_index(merged_df['month'])

    # Use original values for error calculation
    actual_values = merged_df['original_amount']
    predicted_values = merged_df['MA_Seasonalized']

    # Calculate errors
    errors = actual_values - predicted_values

    # Calculate Mean Absolute Deviation (MAD)
    mad = errors.abs().mean()

    # Calculate Mean Squared Error (MSE)
    mse = (errors**2).mean()

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = ((errors / actual_values) * 100).abs().mean()

    return actual_values, predicted_values, errors, mad, mse, mape


def regression_results(actual_values, predicted_values):
    """Display regression metrics in Streamlit."""
    explained_variance = explained_variance_score(actual_values, predicted_values)
    mean_squared_log = mean_squared_log_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    mean_absolute = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)

    st.write(f'Explained Variance: {explained_variance:.4f}')
    st.write(f'Mean Squared Log Error: {mean_squared_log:.4f}')
    st.write(f'R-squared (R2): {r2:.4f}')
    st.write(f'Mean Absolute Error (MAE): {mean_absolute:.4f}')
    st.write(f'Mean Squared Error (MSE): {mse:.4f}')
    st.write(f'Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}')


def fit_linear_regression(train_data):
    """Fit a linear regression model to time series data."""
    X_train = np.array(train_data.index).reshape(-1, 1)
    y_train = train_data.values

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def plot_data_with_fitted_line(data, model, test_data, trend, base):
    """Plot time series data with fitted regression line."""
    fig = px.scatter(
        data, x=data.index, y=data.values,
        labels={'x': 'Date', 'y': 'Values'},
        title='Time Series with Fitted Line'
    )

    # Plot the fitted line
    fitted_line = model.predict(np.array(data.index).reshape(-1, 1))
    fig.add_trace(
        px.line(
            x=data.index, y=fitted_line,
            title=f'Fitted Line: {trend[0]:.2f}x + {base:.2f}'
        ).data[0]
    )

    # Highlight the test set
    fig.add_trace(
        px.scatter(test_data, x=test_data.index, y=test_data.values, title='Test Set').data[0]
    )

    st.plotly_chart(fig)


def calculate_ma_and_errors(order, timeseries):
    """Calculate Moving Average and error metrics."""
    # Calculate Moving Average
    ma = timeseries.rolling(window=order).mean().dropna()

    # Calculate errors
    errors = timeseries - ma

    # Calculate Mean Absolute Deviation (MAD)
    mad = errors.abs().mean()

    # Calculate Mean Squared Error (MSE)
    mse = (errors**2).mean()

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = ((errors / timeseries) * 100).abs().mean()

    return ma, errors, mad, mse, mape


def train_arima_model(time_series, order):
    """Train an ARIMA model and display summary."""
    model = ARIMA(time_series, order=order)
    fitted_model = model.fit()
    st.code(fitted_model.summary())
    return fitted_model


def make_predictions(model, steps):
    """Make forecast predictions using a fitted model."""
    predictions = model.forecast(steps=steps)
    return predictions


def plot_bar_plots(actual, predictions, title):
    """Plot actual vs predicted values as grouped bar chart."""
    fig = go.Figure()

    # Add bars for actual values
    fig.add_trace(go.Bar(
        x=actual.index, y=actual,
        name='Actual', marker_color='blue', opacity=0.7
    ))

    # Add bars for predicted values
    fig.add_trace(go.Bar(
        x=predictions.index, y=predictions,
        name='Predicted', marker_color='orange', opacity=0.7
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title='Value',
        barmode='group',
    )

    st.plotly_chart(fig, use_container_width=True)
