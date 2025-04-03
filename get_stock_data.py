def get_stock_data(symbol):
    """
    Fetch financial data for a given stock ticker symbol and save as CSV files.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
    """
    import yfinance as yf
    import pandas as pd

    try:
        # Create Ticker object
        stock = yf.Ticker(symbol)

        # Fetch financial data
        income = stock.income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow

        # Save data to CSV files
        income.to_csv(f"{symbol}_income_statement.csv")
        balance.to_csv(f"{symbol}_balance_sheet.csv")
        cashflow.to_csv(f"{symbol}_cash_flow.csv")

        print(f"CSV files generated for {symbol}:")
        print(f"- {symbol}_income_statement.csv")
        print(f"- {symbol}_balance_sheet.csv")
        print(f"- {symbol}_cash_flow.csv")

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")

# Example usage
if __name__ == "__main__":
    symbol = input("Enter stock ticker symbol (e.g., AAPL): ") or "AAPL"
    get_stock_data(symbol)
