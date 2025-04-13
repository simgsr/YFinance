import yfinance as yf
import pandas as pd
import psycopg2
from psycopg2 import Error
from datetime import datetime
import getpass

# Get CSV path and password from user input
csv_path = input("Please enter the path to your ticker CSV file: ")
password = getpass.getpass("Please enter your PostgreSQL password: ")
tickers_df = pd.read_csv(csv_path)
tickers = tickers_df.iloc[:, 0].tolist()  # Use first column automatically

# PostgreSQL connection details
db_params = {
    "host": "localhost",
    "database": "yfinance_db",
    "user": "postgres",
    "password": "X@dus3314",
    "port": "5432"
}

# Create table SQL
create_table_query = """
CREATE TABLE IF NOT EXISTS financial_data (
    ticker VARCHAR(20) PRIMARY KEY,
    roic FLOAT,
    free_cash_flow BIGINT,
    total_revenue BIGINT,
    price_to_owner_earnings FLOAT,
    pe_ratio FLOAT,
    pb_ratio FLOAT,
    roe FLOAT,
    insider_ownership FLOAT,
    eps FLOAT,
    reinvestment_rate FLOAT,
    debt_to_equity FLOAT,
    interest_coverage_ratio FLOAT,
    total_debt BIGINT,
    cash_reserves BIGINT,
    gross_margins FLOAT,
    fetch_date DATE
);
"""

# Function to calculate missing metrics
def calculate_metrics(ticker_data):
    info = ticker_data.info
    income_stmt = ticker_data.income_stmt
    balance_sheet = ticker_data.balance_sheet
    cashflow = ticker_data.cashflow

    # ROIC: Net Income / Total Invested Capital
    net_income = income_stmt.get('Net Income', {}).get('2023', 0) if 'Net Income' in income_stmt else 0
    total_invested_capital = balance_sheet.get('Total Assets', {}).get('2023', 0) - balance_sheet.get('Total Current Liabilities', {}).get('2023', 0) if 'Total Assets' in balance_sheet else 0
    roic = (net_income / total_invested_capital) * 100 if total_invested_capital != 0 else None

    # Free Cash Flow
    free_cash_flow = cashflow.get('Free Cash Flow', {}).get('2023', None)

    # Total Revenue
    total_revenue = income_stmt.get('Total Revenue', {}).get('2023', None)

    # Price/Owner Earnings: (Net Income + Depreciation - Capex) / Market Cap
    depreciation = cashflow.get('Depreciation', {}).get('2023', 0) if 'Depreciation' in cashflow else 0
    capex = cashflow.get('Capital Expenditures', {}).get('2023', 0) if 'Capital Expenditures' in cashflow else 0
    market_cap = info.get('marketCap', 0)
    owner_earnings = net_income + depreciation - capex
    price_to_owner_earnings = (market_cap / owner_earnings) if owner_earnings != 0 else None

    # P/E Ratio
    pe_ratio = info.get('trailingPE', None)

    # P/B Ratio
    pb_ratio = info.get('priceToBook', None)

    # ROE: Net Income / Total Equity
    total_equity = balance_sheet.get('Total Stockholder Equity', {}).get('2023', 0) if 'Total Stockholder Equity' in balance_sheet else 0
    roe = (net_income / total_equity) * 100 if total_equity != 0 else None

    # Insider Ownership
    insider_ownership = info.get('heldPercentInsiders', None)

    # EPS
    eps = info.get('trailingEps', None)

    # Reinvestment Rate: (Net Income - Dividends) / Net Income
    dividends = cashflow.get('Dividends Paid', {}).get('2023', 0) if 'Dividends Paid' in cashflow else 0
    reinvestment_rate = ((net_income - dividends) / net_income) * 100 if net_income != 0 else None

    # Debt/Equity
    debt_to_equity = info.get('debtToEquity', None)

    # Interest Coverage: Operating Income / Interest Expense
    operating_income = income_stmt.get('Operating Income', {}).get('2023', 0) if 'Operating Income' in income_stmt else 0
    interest_expense = income_stmt.get('Interest Expense', {}).get('2023', 0) if 'Interest Expense' in income_stmt else 0
    interest_coverage_ratio = operating_income / abs(interest_expense) if interest_expense != 0 else None

    # Total Debt
    total_debt = balance_sheet.get('Total Debt', {}).get('2023', None)

    # Cash Reserves
    cash_reserves = balance_sheet.get('Cash', {}).get('2023', None)

    # Gross Margins
    gross_margins = info.get('grossMargins', None)

    return (roic, free_cash_flow, total_revenue, price_to_owner_earnings, pe_ratio, pb_ratio, roe,
            insider_ownership, eps, reinvestment_rate, debt_to_equity, interest_coverage_ratio,
            total_debt, cash_reserves, gross_margins)

# Connect to PostgreSQL and fetch data
connection = None
cursor = None
try:
    # First connect to default 'postgres' database to create yfinance_db
    temp_params = db_params.copy()
    temp_params['database'] = 'postgres'
    temp_conn = psycopg2.connect(**temp_params)
    temp_conn.autocommit = True
    temp_cursor = temp_conn.cursor()

    # Create database if it doesn't exist
    try:
        temp_cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'yfinance_db'")
        if not temp_cursor.fetchone():
            temp_cursor.execute("CREATE DATABASE yfinance_db")
            print("Created database 'yfinance_db'")
    except Error as e:
        print(f"Database creation check failed: {e}")

    temp_cursor.close()
    temp_conn.close()

    # Now connect to yfinance_db
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    # Create table
    cursor.execute(create_table_query)
    connection.commit()

    # Fetch and insert data for each ticker
    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)
        metrics = calculate_metrics(ticker)

        insert_query = """
        INSERT INTO financial_data (ticker, roic, free_cash_flow, total_revenue, price_to_owner_earnings,
            pe_ratio, pb_ratio, roe, insider_ownership, eps, reinvestment_rate, debt_to_equity,
            interest_coverage_ratio, total_debt, cash_reserves, gross_margins, fetch_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET
            roic = EXCLUDED.roic,
            free_cash_flow = EXCLUDED.free_cash_flow,
            total_revenue = EXCLUDED.total_revenue,
            price_to_owner_earnings = EXCLUDED.price_to_owner_earnings,
            pe_ratio = EXCLUDED.pe_ratio,
            pb_ratio = EXCLUDED.pb_ratio,
            roe = EXCLUDED.roe,
            insider_ownership = EXCLUDED.insider_ownership,
            eps = EXCLUDED.eps,
            reinvestment_rate = EXCLUDED.reinvestment_rate,
            debt_to_equity = EXCLUDED.debt_to_equity,
            interest_coverage_ratio = EXCLUDED.interest_coverage_ratio,
            total_debt = EXCLUDED.total_debt,
            cash_reserves = EXCLUDED.cash_reserves,
            gross_margins = EXCLUDED.gross_margins,
            fetch_date = EXCLUDED.fetch_date;
        """
        cursor.execute(insert_query, (ticker_symbol, *metrics, datetime.today().date()))
        connection.commit()
        print(f"Data inserted for {ticker_symbol}")

except (Exception, Error) as error:
    print(f"Error: {error}")
finally:
    if cursor is not None:
        cursor.close()
    if connection is not None:
        connection.close()
        print("PostgreSQL connection closed.")
