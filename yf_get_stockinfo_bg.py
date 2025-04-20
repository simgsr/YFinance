# yf_get_stockinfo.py
import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pandas import ExcelWriter

def save_to_excel(ticker, data_dict):
    """Save all data to a single Excel file with multiple sheets"""
    filename = f"{ticker}_Financial_Reports_{datetime.today().strftime('%Y%m%d')}.xlsx"

    try:
        with ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                # Clean sheet names
                clean_sheet_name = sheet_name[:31].replace('/', ' ').replace('\\', ' ')

                # Make a copy of the DataFrame to avoid SettingWithCopyWarning
                df_to_save = df.copy()

                # Remove timezone from datetime columns
                for col in df_to_save.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_to_save[col]):
                        df_to_save[col] = df_to_save[col].dt.tz_localize(None)

                # Financial Statements need index (metrics) as rows
                if sheet_name == "Financial Statements":
                    df_to_save.to_excel(writer, sheet_name=clean_sheet_name, index=True)
                # Historical Prices needs date index
                elif sheet_name == "Historical Prices":
                    # Ensure the index is timezone-naive
                    if df_to_save.index.tz is not None:
                        df_to_save.index = df_to_save.index.tz_localize(None)
                    df_to_save.to_excel(writer, sheet_name=clean_sheet_name, index=True)
                # Other sheets don't need index
                else:
                    df_to_save.to_excel(writer, sheet_name=clean_sheet_name, index=False)

        print(f"💾 Saved {filename}")
        return filename

    except Exception as e:
        print(f"❌ Failed to save {filename}: {str(e)}")
        return None

def combine_financial_statements(stock):
    """Combine financial statements with years as columns and statement type identification"""
    try:
        # Get each statement (already has dates as columns)
        income = stock.financials
        balance = stock.balance_sheet
        cashflow = stock.cashflow

        # Add statement type columns
        income['Statement Type'] = 'Income Statement'
        balance['Statement Type'] = 'Balance Sheet'
        cashflow['Statement Type'] = 'Cash Flow'

        # Add category prefixes
        income = income.rename(index=lambda x: f"Income: {x}")
        balance = balance.rename(index=lambda x: f"Balance: {x}")
        cashflow = cashflow.rename(index=lambda x: f"CashFlow: {x}")

        # Combine all statements
        combined = pd.concat([income, balance, cashflow])

        # Convert dates to year format and remove timezone
        def format_date(col):
            if str(col) == 'Statement Type':
                return col
            try:
                dt = pd.to_datetime(col)
                if dt.tz is not None:
                    dt = dt.tz_localize(None)
                return dt.strftime('%Y-%m-%d')
            except:
                return col

        combined.columns = [format_date(col) for col in combined.columns]

        # Reorder columns to have Statement Type first
        cols = ['Statement Type'] + [col for col in combined.columns if col != 'Statement Type']
        combined = combined[cols]

        return combined

    except Exception as e:
        print(f"❌ Error combining financial statements: {str(e)}")
        return pd.DataFrame()

def get_historical_data(stock):
    """Get 10 years of historical data with retries"""
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365*10)

        # Try with different intervals if daily fails
        for interval in ['1d', '1wk', '1mo']:
            try:
                hist = stock.history(start=start_date, end=end_date, interval=interval)
                if not hist.empty:
                    # Clean up the data
                    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
                    hist = hist.dropna(how='all')  # Remove rows where all values are NA

                    # Ensure index is timezone-naive
                    if hist.index.tz is not None:
                        hist.index = hist.index.tz_localize(None)

                    return hist
            except Exception as e:
                print(f"⚠️ Warning: Failed to get history with interval {interval}: {str(e)}")
                continue

        print("❌ Failed to get historical data after multiple attempts")
        return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error getting historical data: {str(e)}")
        return pd.DataFrame()

def get_key_statistics(stock):
    """Get comprehensive key statistics"""
    try:
        info = stock.info
        stats = {
            'Valuation': [
                ('Market Cap', info.get('marketCap')),
                ('Enterprise Value', info.get('enterpriseValue')),
                ('Trailing P/E', info.get('trailingPE')),
                ('Forward P/E', info.get('forwardPE')),
                ('PEG Ratio', info.get('pegRatio')),
                ('Price/Sales', info.get('priceToSalesTrailing12Months')),
                ('Price/Book', info.get('priceToBook')),
            ],
            'Profitability': [
                ('Profit Margin', info.get('profitMargins')),
                ('Operating Margin', info.get('operatingMargins')),
                ('Return on Assets', info.get('returnOnAssets')),
                ('Return on Equity', info.get('returnOnEquity')),
            ],
            'Growth': [
                ('Revenue Growth', info.get('revenueGrowth')),
                ('EBITDA Growth', info.get('ebitdaGrowth')),
                ('Earnings Growth', info.get('earningsGrowth')),
            ],
            'Financial Health': [
                ('Current Ratio', info.get('currentRatio')),
                ('Quick Ratio', info.get('quickRatio')),
                ('Debt/Equity', info.get('debtToEquity')),
                ('Total Debt', info.get('totalDebt')),
                ('Total Cash', info.get('totalCash')),
            ],
            'Dividends': [
                ('Dividend Yield', info.get('dividendYield')),
                ('Payout Ratio', info.get('payoutRatio')),
            ]
        }

        # Create DataFrame with category headers
        data = []
        for category, metrics in stats.items():
            data.append((category, '', ''))  # Category header row
            data.extend(metrics)

        return pd.DataFrame(data, columns=['Metric', 'Value', 'Description'])

    except Exception as e:
        print(f"❌ Error getting key statistics: {str(e)}")
        return pd.DataFrame()

def get_fundamentals(stock):
    """Get company fundamentals with better formatting"""
    try:
        info = stock.info
        fundamentals = []

        # Organize fundamentals by category
        categories = {
            'Company Information': [
                'longName', 'sector', 'industry', 'fullTimeEmployees',
                'country', 'website', 'address1', 'city', 'state', 'zip'
            ],
            'Business Summary': ['longBusinessSummary'],
            'Financial Metrics': [
                'totalRevenue', 'ebitda', 'freeCashflow',
                'operatingCashflow', 'grossProfits'
            ],
            'Share Statistics': [
                'sharesOutstanding', 'floatShares', 'sharesShort',
                'sharesShortPriorMonth', 'shortRatio', 'heldPercentInsiders',
                'heldPercentInstitutions', 'shortPercentOfFloat'
            ],
            'Trading Information': [
                'exchange', 'quoteType', 'symbol', 'market',
                'currency', 'financialCurrency'
            ]
        }

        for category, keys in categories.items():
            fundamentals.append((category, '', ''))  # Add category header
            for key in keys:
                value = info.get(key, 'N/A')
                # Format large numbers
                if isinstance(value, (int, float)) and value >= 1e6:
                    value = f"{value/1e6:,.2f}M" if value < 1e9 else f"{value/1e9:,.2f}B"
                fundamentals.append((key, str(value), ''))

        return pd.DataFrame(fundamentals, columns=['Metric', 'Value', 'Description'])

    except Exception as e:
        print(f"❌ Error getting fundamentals: {str(e)}")
        return pd.DataFrame()

def process_ticker(ticker):
    print(f"\n📊 Processing {ticker}")
    try:
        stock = yf.Ticker(ticker)
        reports = {}

        # 1. Financial Statements (years as columns)
        reports["Financial Statements"] = combine_financial_statements(stock)

        # 2. Key Statistics
        reports["Key Statistics"] = get_key_statistics(stock)

        # 3. Historical Prices (10 years)
        reports["Historical Prices"] = get_historical_data(stock)

        # 4. Company Fundamentals
        reports["Fundamentals"] = get_fundamentals(stock)

        return save_to_excel(ticker, reports)

    except Exception as e:
        print(f"❌ Error processing {ticker}: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yf_get_stockinfo.py TICKER1 TICKER2...")
        print("Example: python yf_get_stockinfo.py AAPL MSFT TSLA")
        sys.exit(1)

    success_count = 0
    for ticker in sys.argv[1:]:
        ticker = ticker.upper().strip()
        if process_ticker(ticker):
            success_count += 1

    print(f"\n✅ Successfully processed {success_count}/{len(sys.argv[1:])} tickers")
