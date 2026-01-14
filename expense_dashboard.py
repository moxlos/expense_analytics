#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expense Analytics Dashboard

A Streamlit-based dashboard for analyzing organizational expenses with
descriptive, predictive, and prescriptive analytics capabilities.

@author: eleftherios ntovoris
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
from babel.numbers import format_currency
from dash_queries import DashboardQueries
from arima_models import (
    calculate_ma_and_errors,
    calculate_ma_and_errors_season,
    seasonality_fit,
    fit_linear_regression,
    plot_data_with_fitted_line,
    plot_bar_plots,
    regression_results
)
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dateutil.relativedelta import relativedelta
from scipy.stats import shapiro, norm


def format_cur(number):
    """Format numbers as Euro currency."""
    return format_currency(number, 'EUR', locale='de_DE')


@st.cache_data
def load_district_mapping():
    """Load district ID to name mapping from shapefile."""
    gdf = gpd.read_file('data/periphereies.shp')
    for col in gdf.select_dtypes(include=['object']).columns:
        if col != 'geometry':
            gdf[col] = gdf[col].apply(
                lambda x: x.encode('latin-1').decode('iso8859_7') if isinstance(x, str) else x
            )
    gdf = gdf.reset_index()
    # Map: database district ID (1-indexed) -> name
    id_to_name = {i + 1: name for i, name in enumerate(gdf['PER'])}
    name_to_id = {name: i + 1 for i, name in enumerate(gdf['PER'])}
    return id_to_name, name_to_id


st.set_page_config(layout="wide")


def main():
    st.title("Expense Analytics")

    # Connect to SQLite database
    conn = sqlite3.connect('data/expenses.sqlite')
    dash_q = DashboardQueries(conn)

    # Create a sidebar with navigation options
    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Descriptive", "Predictive", "Prescriptive"]
    )

    # Display content based on the selected page
    if page == "Home":
        home_page(dash_q)
    elif page == "Descriptive":
        page_descriptive(dash_q)
    elif page == "Predictive":
        page_predictive(dash_q)
    elif page == "Prescriptive":
        page_prescriptive(dash_q)

    conn.close()


def create_filters(dash_q):
    """Create sidebar filters for data selection."""
    st.sidebar.header('Filters')

    # Expense code filter
    expense_codes = dash_q.filters_exp_cd()
    selected_expense_code = st.sidebar.multiselect(
        'Select Expense Code',
        expense_codes + ["ALL"],
        default="ALL"
    )
    if "ALL" in selected_expense_code:
        selected_expense_code = expense_codes

    # Department filter
    departments = dash_q.filters_dept()
    selected_department = st.sidebar.multiselect(
        'Select Department',
        departments + ["ALL"],
        default="ALL"
    )
    if "ALL" in selected_department:
        selected_department = departments

    # Company filter
    company_filter = st.sidebar.text_input('Enter Company Filter (e.g., Keyword)', '')

    # Date filters
    result_ft_start_date, result_ft_end_date = dash_q.filters_dt()
    start_date = st.sidebar.date_input(
        'Select Start Date',
        value=pd.to_datetime(result_ft_start_date.iloc[0, 0], format="%Y-%m-%d")
    )
    end_date = st.sidebar.date_input(
        'Select End Date',
        value=pd.to_datetime(result_ft_end_date.iloc[0, 0], format="%Y-%m-%d")
    )

    # District filter
    district_ids = dash_q.filters_ditr()
    id_to_name, name_to_id = load_district_mapping()
    district_names = [id_to_name.get(d, f"District {d}") for d in district_ids]
    selected_district_names = st.sidebar.multiselect(
        'Select District',
        district_names + ["ALL"],
        default="ALL"
    )
    if "ALL" in selected_district_names:
        selected_district = district_ids
    else:
        selected_district = [name_to_id[name] for name in selected_district_names]

    return selected_expense_code, selected_department, company_filter, start_date, end_date, selected_district


def home_page(dash_q):
    """Display home page with expense data and geographic heatmap."""
    st.write("Expense data")

    row_number = st.sidebar.slider(
        'Select number of rows:',
        min_value=5, max_value=100, value=20, step=1
    )

    filters = create_filters(dash_q)
    selected_expense_code, selected_department, company_filter, start_date, end_date, selected_district = filters

    df = dash_q.table(
        start_date, end_date, selected_expense_code,
        selected_department, company_filter, selected_district, row_number
    )
    df['district'] = df['district'] - 1

    # Get district totals
    district_sum = dash_q.district(
        start_date, end_date, selected_expense_code,
        selected_department, company_filter, selected_district
    )
    district_sum['district'] = district_sum['district'] - 1

    # Load geographic data
    gdf = gpd.read_file('data/periphereies.shp')
    # Fix Greek encoding for string columns
    for col in gdf.select_dtypes(include=['object']).columns:
        if col != 'geometry':
            gdf[col] = gdf[col].apply(
                lambda x: x.encode('latin-1').decode('iso8859_7') if isinstance(x, str) else x
            )
    gdf = gdf.reset_index()

    merged_df = gdf.merge(district_sum, left_on='index', right_on='district', how='left')
    merged_df = merged_df.to_crs("WGS84")

    col1, col2 = st.columns([30, 70])

    with col1:
        fig, ax = plt.subplots(1, figsize=(10, 6))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        merged_df.plot(
            column='amount', cmap='Blues', linewidth=1,
            ax=ax, edgecolor='1', legend=True, cax=cax
        )
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        df = df.merge(merged_df[['index', 'PER']], left_on='district', right_on='index', how='left')
        df = df[['id', 'date', 'expense_code', 'department', 'company', 'amount', 'PER']]
        st.dataframe(df, use_container_width=True)


def page_descriptive(dash_q):
    """Display descriptive analytics with KPIs and charts."""
    filters = create_filters(dash_q)
    selected_expense_code, selected_department, company_filter, start_date, end_date, selected_district = filters

    # METRICS
    st.header("Metrics")
    col3, col4, col5, col6 = st.columns(4)

    with col3:
        total_amount = dash_q.query_total_amount(
            start_date, end_date, selected_expense_code,
            selected_department, company_filter, selected_district
        )
        st.metric("Total Amount", format_cur(total_amount))

    with col4:
        total_number = dash_q.query_total_number(
            start_date, end_date, selected_expense_code,
            selected_department, company_filter, selected_district
        )
        st.metric("Expense Number", total_number)

    with col5:
        st.metric("Date Start", str(start_date))

    with col6:
        st.metric("Date End", str(end_date))

    # Monthly data
    st.header("Monthly data")
    col7, col8 = st.columns(2)

    with col7:
        monthly_amount = dash_q.query_amount_by_month(
            start_date, end_date, selected_expense_code,
            selected_department, company_filter, selected_district
        )
        fig_bar_1 = px.bar(monthly_amount, x='month', y='total_amount', title='Total amount EUR')
        st.plotly_chart(fig_bar_1, use_container_width=True)

    with col8:
        monthly_number = dash_q.query_number_by_month(
            start_date, end_date, selected_expense_code,
            selected_department, company_filter, selected_district
        )
        fig_bar_2 = px.bar(monthly_number, x='month', y='total_number', title='Total number')
        st.plotly_chart(fig_bar_2, use_container_width=True)

    # Department/Expense code breakdown
    st.header("Department/Expense code")
    col9, col10 = st.columns(2)

    with col9:
        expense_breakdown = dash_q.query_exp_cd_prc(
            start_date, end_date, selected_expense_code,
            selected_department, company_filter, selected_district
        )
        fig_pie_1 = px.pie(expense_breakdown, names='expense_code', values='total_amount', title='Expense Code')
        st.plotly_chart(fig_pie_1, use_container_width=True)

    with col10:
        department_breakdown = dash_q.query_prg_prc(
            start_date, end_date, selected_expense_code,
            selected_department, company_filter, selected_district
        )
        fig_pie_2 = px.pie(department_breakdown, names='department', values='total_amount', title='Department')
        st.plotly_chart(fig_pie_2, use_container_width=True)


def page_predictive(dash_q):
    """Display predictive analytics options."""
    method = st.sidebar.radio("Select a method", ["Moving Average", "Yearly Trend", "Seasonality"])

    if method == "Moving Average":
        page_moving_average(dash_q)
    elif method == "Yearly Trend":
        page_yearly_trend(dash_q)
    elif method == "Seasonality":
        page_seasonality(dash_q)


def page_moving_average(dash_q):
    """Moving average forecasting page."""
    st.title("MOVING AVERAGE (STATIONARY)")

    filters = create_filters(dash_q)
    selected_expense_code, selected_department, company_filter, start_date, end_date, selected_district = filters

    time_series_data = dash_q.query_amount_by_month(
        start_date, end_date, selected_expense_code,
        selected_department, company_filter, selected_district
    )

    time_series_data['month'] = pd.to_datetime(time_series_data['month'])
    time_series_data.set_index('month', inplace=True)

    order = st.sidebar.number_input(
        'Enter the order of Moving Average:',
        min_value=2, max_value=len(time_series_data)
    )

    ma, errors, mad, mse, mape = calculate_ma_and_errors(order, time_series_data)

    col1, col2 = st.columns(2)
    with col1:
        plot_bar_plots(time_series_data['total_amount'], ma['total_amount'], title="Monthly Data and Prediction")
    with col2:
        st.dataframe(
            pd.concat([
                time_series_data,
                ma.rename(columns={'total_amount': 'prediction'}),
                errors.rename(columns={'total_amount': 'error'})
            ], axis=1),
            use_container_width=True
        )

    st.header('Error Metrics')
    st.write(f'Mean Absolute Deviation (MAD): {round(mad.iloc[0], 2)}')
    st.write(f'Mean Squared Error (MSE): {round(mse.iloc[0], 2)}')
    st.write(f'Mean Absolute Percentage Error (MAPE): {round(mape.iloc[0], 2)}%')


def page_yearly_trend(dash_q):
    """Yearly trend analysis with YoY change and monthly regression."""
    filters = create_filters(dash_q)
    selected_expense_code, selected_department, company_filter, start_date, end_date, selected_district = filters

    time_series_data = dash_q.query_amount_by_month(
        start_date, end_date, selected_expense_code,
        selected_department, company_filter, selected_district
    )

    time_series_data['month'] = pd.to_datetime(time_series_data['month'])
    time_series_data.set_index('month', inplace=True)

    # --- YEARLY OVERVIEW ---
    st.header('YEARLY OVERVIEW')

    time_series_data_y = time_series_data.reset_index()
    yearly_data = time_series_data_y.groupby(time_series_data_y['month'].dt.year)['total_amount'].sum().reset_index()
    yearly_data.columns = ['year', 'total_amount']

    # Calculate year-over-year change
    yearly_data['yoy_change'] = yearly_data['total_amount'].pct_change() * 100
    yearly_data['yoy_label'] = yearly_data['yoy_change'].apply(
        lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%" if pd.notna(x) else ""
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            yearly_data, x='year', y='total_amount',
            title='Yearly Totals',
            text='yoy_label'
        )
        fig.update_traces(textposition='outside')
        fig.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Year-over-Year Summary")
        summary_df = yearly_data[['year', 'total_amount', 'yoy_change']].copy()
        summary_df['total_amount'] = summary_df['total_amount'].apply(lambda x: format_cur(x))
        summary_df['yoy_change'] = summary_df['yoy_change'].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
        )
        summary_df.columns = ['Year', 'Total Amount', 'YoY Change']
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # --- MONTHLY TREND ANALYSIS ---
    st.header('MONTHLY TREND ANALYSIS')

    # Prepare monthly data for regression
    monthly_data = time_series_data.reset_index()
    monthly_data['month_num'] = range(1, len(monthly_data) + 1)

    model = fit_linear_regression(
        pd.Series(monthly_data['total_amount'].values, index=monthly_data['month_num'])
    )

    trend = model.coef_[0]
    base = model.intercept_
    monthly_data['fitted'] = model.predict(monthly_data[['month_num']])

    col3, col4 = st.columns(2)
    with col3:
        fig2 = px.scatter(
            monthly_data, x='month', y='total_amount',
            title='Monthly Trend with Regression Line'
        )
        fig2.add_scatter(
            x=monthly_data['month'], y=monthly_data['fitted'],
            mode='lines', name='Trend Line',
            line=dict(color='red', dash='dash')
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        st.subheader('Trend Statistics')
        st.write(f"**Monthly Trend:** {format_cur(trend)} per month")
        st.write(f"**Annualized Trend:** {format_cur(trend * 12)} per year")
        st.write(f"**Base Amount:** {format_cur(base)}")

        # Calculate R-squared
        ss_res = ((monthly_data['total_amount'] - monthly_data['fitted']) ** 2).sum()
        ss_tot = ((monthly_data['total_amount'] - monthly_data['total_amount'].mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot)
        st.write(f"**R² (Goodness of Fit):** {r_squared:.3f}")


def page_seasonality(dash_q):
    """Seasonality analysis page."""
    st.header('SEASONALITY')

    filters = create_filters(dash_q)
    selected_expense_code, selected_department, company_filter, start_date, end_date, selected_district = filters

    time_series_data = dash_q.query_amount_by_month(
        start_date, end_date, selected_expense_code,
        selected_department, company_filter, selected_district
    )

    time_series_data['month'] = pd.to_datetime(time_series_data['month'])
    time_series_data.set_index('month', inplace=True)

    test_start_date = st.sidebar.date_input(
        "Select test start date",
        value=end_date - relativedelta(months=12)
    )
    order = st.sidebar.number_input(
        'Enter the order of Moving Average:',
        min_value=3, max_value=len(time_series_data)
    )

    test_start_date = pd.to_datetime(test_start_date, format="%Y-%m")
    train_data = time_series_data[time_series_data.index < test_start_date]
    test_data = time_series_data[time_series_data.index >= test_start_date - relativedelta(months=order-1)]

    season_coef = seasonality_fit(train_data.reset_index().rename({'month': 'date'}, axis=1))

    actual_values, predicted_values, errors, mad, mse, mape = calculate_ma_and_errors_season(
        order, test_data, season_coef
    )

    col1, col2 = st.columns(2)
    with col1:
        plot_bar_plots(actual_values, predicted_values, title="Monthly Data and Prediction")
    with col2:
        st.dataframe(
            pd.concat([
                actual_values,
                pd.DataFrame({'predicted': predicted_values, 'errors': errors})
            ], axis=1),
            use_container_width=True
        )

    col3, col4 = st.columns(2)
    with col3:
        st.header('Error Metrics')
        st.write(f'Mean Absolute Deviation (MAD): {round(mad, 2)}')
        st.write(f'Mean Squared Error (MSE): {round(mse, 2)}')
        st.write(f'Mean Absolute Percentage Error (MAPE): {round(mape, 2)}%')
    with col4:
        st.dataframe(season_coef)


def page_prescriptive(dash_q):
    """Prescriptive analytics for budget allocation."""
    st.header("Prescriptive")
    st.title("Budget allocation based on demand distribution")

    page_demand_distribution(dash_q)


def page_demand_distribution(dash_q):
    """Demand distribution analysis for budget planning."""
    filters = create_filters(dash_q)
    selected_expense_code, selected_department, company_filter, start_date, end_date, selected_district = filters

    time_series_data = dash_q.query_amount_by_month_department(
        start_date, end_date, selected_expense_code,
        selected_department, company_filter, selected_district
    )

    # Plot distributions
    fig = px.histogram(
        time_series_data, x='total_amount', color='department', marginal='rug',
        title='Total Amount Distribution by Department'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.sidebar.subheader("Budget Parameters")

    alpha = st.sidebar.slider(
        "Confidence level",
        min_value=0.80, max_value=0.99, value=0.95, step=0.01,
        format="%.2f%%",
        help="Higher confidence = larger budget safety margin"
    )
    Z_alpha = norm.ppf(alpha)

    threshold = st.sidebar.number_input(
        "Budget threshold (EUR)",
        value=10000.0,
        step=1000.0,
        help="Budget limit before borrowing costs apply"
    )

    cost_rate = st.sidebar.number_input(
        "Borrowing cost rate (%)",
        value=1.25,
        min_value=0.0,
        step=0.25,
        help="Interest rate for exceeding budget threshold"
    )




    st.subheader("Normality Tests")
    for department in time_series_data['department'].unique():
        subset = time_series_data[time_series_data['department'] == department]['total_amount']
        stat, p_value = shapiro(subset)
        st.write(f"Department {department}: Shapiro-Wilk test p-value = {round(p_value, 4)}")


    st.subheader("Budget Recommendations")
    summary_stats = time_series_data.groupby('department')['total_amount'].agg(['mean', 'std'])
    departments = time_series_data['department'].unique()

    result_table = pd.DataFrame({
        'Department': departments,
        'Mean (EUR)': [summary_stats.loc[department, 'mean'] for department in departments],
        'Std Dev (EUR)': [summary_stats.loc[department, 'std'] for department in departments],
        f'Recommended Budget (EUR)\n{alpha*100:.0f}% Upper Bound': [
            summary_stats.loc[department, 'mean'] + Z_alpha * summary_stats.loc[department, 'std']
            for department in departments
        ],
    })

    # Convert cost_rate from percentage to decimal (e.g., 1.25% -> 0.0125)
    cost_decimal = cost_rate / 100
    result_table['Amount Over Threshold (EUR)'] = result_table.apply(
        lambda row: max(0, row.iloc[3] - threshold), axis=1
    )
    result_table['Borrowing Cost (EUR)'] = result_table.apply(
        lambda row: row['Amount Over Threshold (EUR)'] * cost_decimal, axis=1
    )

    # Format numeric columns
    for col in result_table.columns:
        if 'EUR' in col:
            result_table[col] = result_table[col].apply(lambda x: f'{x:,.2f}')

    st.dataframe(result_table)

    st.caption(
        f" The recommended budget is calculated as Mean + {Z_alpha:.2f} × Std Dev, "
        f"providing a {alpha*100:.0f}% confidence upper bound. "
        f"Borrowing costs apply at {cost_rate}% for amounts exceeding {threshold:,.0f} EUR."
    )


if __name__ == "__main__":
    main()
