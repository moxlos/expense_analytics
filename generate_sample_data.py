#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample Data Generator for Expense Analytics Dashboard

Generates synthetic expense data for demonstration purposes.
The generated data mimics organizational expense patterns with
seasonality, trends, and realistic distributions.

Usage:
    python generate_sample_data.py
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_data(
    start_date='2020-01-01',
    end_date='2023-12-31',
    n_records=5000,
    seed=42
):
    """
    Generate synthetic expense data.

    Args:
        start_date: Start date for the data range
        end_date: End date for the data range
        n_records: Number of expense records to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic expense data
    """
    np.random.seed(seed)

    # Define categories
    expense_codes = ['EC001', 'EC002', 'EC003', 'EC004', 'EC005']
    departments = ['Dpt A', 'Dpt B', 'Dpt C', 'Dpt D']
    companies = [
        'Acme Corp', 'Beta Industries', 'Gamma Services', 'Delta Solutions',
        'Epsilon Tech', 'Zeta Consulting', 'Eta Systems', 'Theta Partners',
        'Iota Holdings', 'Kappa Group'
    ]
    districts = list(range(1, 14))  # 13 districts (Greek regions)

    # Generate dates with some seasonality
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days

    # Generate random dates with higher frequency in certain months (budget cycles)
    dates = []
    for _ in range(n_records):
        random_days = np.random.randint(0, date_range)
        date = start + timedelta(days=random_days)
        # Add seasonality: more expenses in Q4 and Q1
        month = date.month
        if month in [10, 11, 12, 1, 2, 3]:
            if np.random.random() < 0.7:  # 70% chance to keep Q4/Q1 dates
                dates.append(date)
            else:
                # Shift to a Q4/Q1 month
                # Filter months based on day to avoid invalid dates
                day = date.day
                q4_q1_months = [10, 11, 12, 1, 2, 3]

                if day == 31:
                    # Only months with 31 days: Jan, Mar, Oct, Dec
                    valid_months = [m for m in q4_q1_months if m in [1, 3, 10, 12]]
                elif day >= 29:
                    # Exclude February (has max 29 days in leap years, but safer to exclude)
                    valid_months = [m for m in q4_q1_months if m != 2]
                else:
                    valid_months = q4_q1_months

                new_month = np.random.choice(valid_months)
                date = date.replace(month=new_month)
                dates.append(date)
        else:
            dates.append(date)

    # Generate amounts with log-normal distribution (realistic for expenses)
    # Different expense codes have different typical amounts
    expense_code_means = {
        'EC001': 10000,  # Large infrastructure expenses
        'EC002': 5000,   # Medium operational expenses
        'EC003': 2000,   # Small supplies
        'EC004': 15000,  # Equipment purchases
        'EC005': 3000,   # Services
    }

    records = []
    for i, date in enumerate(dates):
        expense_code = np.random.choice(expense_codes)
        base_amount = expense_code_means[expense_code]

        # Add some randomness and yearly trend (inflation)
        year_factor = 1 + 0.03 * (date.year - 2020)  # 3% yearly increase
        month_factor = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)  # Seasonality

        amount = np.random.lognormal(
            mean=np.log(base_amount * year_factor * month_factor),
            sigma=0.5
        )
        amount = round(amount, 2)

        records.append({
            'id': i + 1,
            'date': date.strftime('%Y-%m-%d'),
            'expense_code': expense_code,
            'department': np.random.choice(departments),
            'company': np.random.choice(companies),
            'amount': amount,
            'district': np.random.choice(districts)
        })

    return pd.DataFrame(records)


def create_database(df, db_path='data/expenses.sqlite'):
    """
    Create SQLite database from DataFrame.

    Args:
        df: DataFrame with expense data
        db_path: Path to the SQLite database file
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")

    # Create new database
    conn = sqlite3.connect(db_path)
    df.to_sql('expenses', conn, index=False, if_exists='replace')
    conn.close()

    print(f"Created database: {db_path}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Expense codes: {df['expense_code'].unique().tolist()}")
    print(f"Programs: {df['department'].unique().tolist()}")
    print(f"Districts: {sorted(df['district'].unique().tolist())}")


def main():
    print("Generating sample expense data...")
    df = generate_sample_data()

    print("\nSample of generated data:")
    print(df.head(10).to_string(index=False))

    print("\nSummary statistics:")
    print(f"Total amount: {df['amount'].sum():,.2f} EUR")
    print(f"Average amount: {df['amount'].mean():,.2f} EUR")
    print(f"Median amount: {df['amount'].median():,.2f} EUR")

    print("\nCreating database...")
    create_database(df)

    print("\nDone! You can now run the dashboard with: streamlit run expense_dashboard.py")


if __name__ == "__main__":
    main()
