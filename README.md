# Expense Analytics Dashboard

A Streamlit-based analytics dashboard for organizational expense data, featuring descriptive statistics, predictive forecasting, and prescriptive budget recommendations.

## Features

### Descriptive Analytics

- **KPI Dashboard**: Total expenses, transaction counts, date range summaries
- **Monthly Trends**: Bar charts showing expense amounts and transaction volumes over time
- **Category Breakdown**: Pie charts for expense codes and department distributions
- **Geographic Visualization**: Heatmap showing expenses by district/region

### Predictive Analytics

- **Moving Average Forecasting**: Simple moving average with configurable window size and error metrics (MAD, MSE, MAPE)
- **Yearly Trend Analysis**: Linear regression to identify expense trends over years
- **Seasonality Decomposition**: Classical multiplicative decomposition to identify and forecast seasonal patterns

### Prescriptive Analytics

- **Demand Distribution Analysis**: Histogram visualization of expense distributions by department
- **Normality Testing**: Shapiro-Wilk tests to validate statistical assumptions
- **Budget Recommendations**: Confidence interval-based budget allocation with borrowing cost calculations

## Analytics Methods

### Moving Average

Calculates a rolling average over a specified window to smooth time series data and make short-term predictions. Error metrics help evaluate forecast accuracy:

- **MAD** (Mean Absolute Deviation): Average absolute error
- **MSE** (Mean Squared Error): Average squared error (penalizes large errors)
- **MAPE** (Mean Absolute Percentage Error): Average percentage error

### Linear Trend Analysis

Fits a linear regression model (`y = ax + b`) to yearly aggregated data where:

- `a` (trend): Rate of change per year
- `b` (base): Baseline expense level

### Seasonality

Uses classical multiplicative decomposition:

1. Calculate monthly averages across years
2. Compute seasonal factors (month average / overall average)
3. Deseasonalize data for forecasting
4. Re-apply seasonal factors to predictions

### Budget Confidence Intervals

Uses normal distribution assumptions to calculate budget requirements at specified confidence levels:

- Budget = Mean + Z_alpha * Standard Deviation
- Where Z_alpha is the critical value for the desired confidence level (e.g., 1.645 for 95%)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/expense-analytics.git
cd expense-analytics

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using Sample Data

```bash
# Generate sample data
python generate_sample_data.py

# Run the dashboard
streamlit run expense_dashboard.py
```

### Using Your Own Data

The dashboard expects a SQLite database at `data/expenses.sqlite` with a table named `expenses` containing:

| Column       | Type    | Description               |
| ------------ | ------- | ------------------------- |
| id           | INTEGER | Unique identifier         |
| date         | TEXT    | Date in YYYY-MM-DD format |
| expense_code | TEXT    | Expense category code     |
| department   | TEXT    | Department/division name  |
| company      | TEXT    | Vendor/company name       |
| amount       | REAL    | Expense amount            |
| district     | INTEGER | Geographic district ID    |

For geographic visualization, place a shapefile at `data/periphereies.shp` with district boundaries.

## Project Structure

```
expense-analytics/
├── expense_dashboard.py           # Main Streamlit application
├── dash_queries.py        # Database query layer
├── arima_models.py        # Time series and forecasting functions
├── generate_sample_data.py # Sample data generator
├── requirements.txt       # Python dependencies
├── data/
│   ├── expenses.sqlite # SQLite database
│   └── periphereies.*     # Shapefile for geographic visualization
└── README.md
```

## Usage

1. **Home Page**: View raw expense data with geographic distribution
2. **Descriptive**: Explore KPIs, monthly trends, and category breakdowns
3. **Predictive**: Select forecasting method (Moving Average, Yearly Trend, or Seasonality)
4. **Prescriptive**: Analyze demand distributions and get budget recommendations

### Filters

All pages support filtering by:

- Expense Code
- Department
- Company (keyword search)
- Date Range
- District

## Screenshots
<img width="1908" height="783" alt="Screenshot from 2026-01-14 16-41-22" src="https://github.com/user-attachments/assets/66ca26b3-1e19-45c1-849f-5833a14d7e26" />

<img width="1908" height="783" alt="Screenshot from 2026-01-14 16-41-42" src="https://github.com/user-attachments/assets/a016778d-8af3-423b-85c1-80e35f32a6e2" />

<img width="1908" height="783" alt="Screenshot from 2026-01-14 16-42-00" src="https://github.com/user-attachments/assets/cc269b86-2f46-4c31-99e6-bc49b3649a45" />

## License

### GNU General Public License v3.0

## Author

Eleftherios Ntovoris
