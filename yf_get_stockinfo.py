def yFinance_multiperTicker(symbols):
    """
    Fetch financial data, fundamentals, and historical prices for multiple tickers,
    saving each ticker's data into a SINGLE Excel file with multiple sheets.
    """
    import yfinance as yf
    import pandas as pd

    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            filename = f"{symbol}_data.xlsx"

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # -----------------------------------------------------------------
                # Sheet 1: Financial Statements (Income, Balance Sheet, Cash Flow)
                # -----------------------------------------------------------------
                income = stock.income_stmt.reset_index().rename(columns={'index': 'Metric'})
                balance = stock.balance_sheet.reset_index().rename(columns={'index': 'Metric'})
                cashflow = stock.cashflow.reset_index().rename(columns={'index': 'Metric'})

                financials = pd.concat([
                    income.assign(Statement='Income Statement'),
                    balance.assign(Statement='Balance Sheet'),
                    cashflow.assign(Statement='Cash Flow')
                ], ignore_index=True)

                financials.to_excel(writer, sheet_name='Financials', index=False)

                # -----------------------------------------------------------------
                # Sheet 2: Fundamental Metrics (P/E ratio, market cap, etc.)
                # -----------------------------------------------------------------
                fundamentals = pd.DataFrame.from_dict(stock.info, orient='index', columns=['Value'])
                fundamentals.to_excel(writer, sheet_name='Fundamentals')

                # -----------------------------------------------------------------
                # Sheet 3: Historical Market Data (Daily OHLCV prices)
                # -----------------------------------------------------------------
                history = stock.history(period="max")
                if not history.empty:
                    # Remove timezone from datetime index
                    history.index = history.index.tz_localize(None)
                    history = history.reset_index()  # Convert index to "Date" column
                    history.to_excel(writer, sheet_name='Historical Prices', index=False)

            print(f"Saved {symbol} data to {filename}")

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

def get_tickers_from_input(user_input):
    import pandas as pd
    import os

    user_input = user_input.strip()
    # Check if input is a CSV file path
    if user_input.lower().endswith('.csv') and os.path.exists(user_input):
        try:
            df = pd.read_csv(user_input)
            # Assume tickers are in the first column, or column named 'ticker'/'symbol'
            if 'ticker' in df.columns:
                return df['ticker'].str.strip().str.upper().tolist()
            elif 'symbol' in df.columns:
                return df['symbol'].str.strip().str.upper().tolist()
            else:
                return df.iloc[:, 0].str.strip().str.upper().tolist()
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            return ["AAPL"]  # Default to AAPL if CSV reading fails
    else:
        # Treat as comma-separated tickers
        return [s.strip().upper() for s in user_input.split(",")] if user_input else ["AAPL"]

if __name__ == "__main__":
    user_input = input("Enter stock ticker(s) (comma-separated) or path to CSV file: ")
    symbols = get_tickers_from_input(user_input)
    yFinance_multiperTicker(symbols)
