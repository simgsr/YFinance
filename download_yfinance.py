import yfinance as yf
import pandas as pd

symbol = "AAPL"
stock = yf.Ticker(symbol)

# Fetch data
income = stock.income_stmt
balance = stock.balance_sheet
cashflow = stock.cashflow

# Save to CSV
income.to_csv(f"{symbol}_income_10y.csv")
balance.to_csv(f"{symbol}_balance_sheet_10y.csv")
cashflow.to_csv(f"{symbol}_cash_flow_10y.csv")

print(f"Data saved for {symbol}!")
