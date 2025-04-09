import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from textwrap import fill

def print_header(title, width=80):
    print(f"\n{'=' * width}")
    print(f"{title.upper():^{width}}")
    print(f"{'=' * width}")

def print_metric_group(metric_name, data):
    print(f"\n▪ {metric_name}:")
    print("-" * (len(metric_name) + 2))
    print(data.to_string(index=False))
    print()

def evaluate_stock(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)

    try:
        info = ticker.info
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None

    evaluation = []

    def add_eval(category, metric, value, threshold, condition):
        if value is not None:
            passed = condition(value, threshold) if threshold is not None else "N/A"
            evaluation.append({
                'Ticker': ticker_symbol,
                'Category': category,
                'Metric': metric,
                'Value': round(value, 4) if isinstance(value, (int, float)) else value,
                'Threshold': threshold,
                'Pass/Fail': "✅" if passed == True else "❌" if passed == False else "N/A"
            })

    criteria = [
        # Economic Moat Criteria
        ("Economic Moat", "ROIC (>12%)", [
            lambda: (income_stmt.loc['Net Income'].iloc[0] /
                    (balance_sheet.loc['Total Assets'].iloc[0] -
                     balance_sheet.loc['Current Liabilities'].iloc[0])),
            lambda: info.get('returnOnInvestedCapital')
        ], 0.12, lambda x, y: x > y),

        ("Economic Moat", "FCF/Sales (>15%)", [
            lambda: cashflow.loc['Free Cash Flow'].iloc[0] / income_stmt.loc['Total Revenue'].iloc[0],
            lambda: info.get('freeCashflow') / info.get('totalRevenue')
        ], 0.15, lambda x, y: x > y),

        # Margin of Safety Criteria
        ("Margin of Safety", "P/E (<15)", [
            lambda: info.get('trailingPE') or info.get('forwardPE'),
            lambda: info.get('marketCap') / income_stmt.loc['Net Income'].iloc[0]
        ], 15, lambda x, y: x < y),

        ("Margin of Safety", "P/B (<1.5)", [
            lambda: info.get('priceToBook'),
            lambda: info.get('marketCap') / balance_sheet.loc['Total Stockholder Equity'].iloc[0],
            lambda: info.get('marketCap') / (balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0] -
                                            balance_sheet.loc['Minority Interest'].iloc[0])
        ], 1.5, lambda x, y: x < y),

        # Management Quality Criteria
        ("Management Quality", "ROE (>15%)", [
            lambda: income_stmt.loc['Net Income'].iloc[0] / balance_sheet.loc['Total Stockholder Equity'].iloc[0],
            lambda: income_stmt.loc['Net Income'].iloc[0] / (balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0] -
                                                            balance_sheet.loc['Minority Interest'].iloc[0]),
            lambda: info.get('returnOnEquity')
        ], 0.15, lambda x, y: x > y),

        # Long-Term Focus Criteria
        ("Long-Term Focus", "Debt/Equity (<0.5)", [
            lambda: info.get('debtToEquity'),
            lambda: balance_sheet.loc['Total Debt'].iloc[0] / balance_sheet.loc['Total Stockholder Equity'].iloc[0],
            lambda: balance_sheet.loc['Total Debt'].iloc[0] / (balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0] -
                                                              balance_sheet.loc['Minority Interest'].iloc[0])
        ], 0.5, lambda x, y: x < y),

        ("Long-Term Focus", "Interest Coverage (>8x)", [
            lambda: income_stmt.loc['EBIT'].iloc[0] / income_stmt.loc['Interest Expense'].iloc[0],
            lambda: info.get('operatingIncome') / info.get('interestExpense')
        ], 8, lambda x, y: x > y),

        ("Long-Term Focus", "EPS CAGR (>6%)", [
            lambda: (income_stmt.loc['Diluted EPS'].iloc[0] / income_stmt.loc['Diluted EPS'].iloc[-1]) ** (1 / (len(income_stmt.columns) - 1)) - 1 if 'Diluted EPS' in income_stmt.index and len(income_stmt.columns) >= 2 else None,
            lambda: (income_stmt.loc['Basic EPS'].iloc[0] / income_stmt.loc['Basic EPS'].iloc[-1]) ** (1 / (len(income_stmt.columns) - 1)) - 1 if 'Basic EPS' in income_stmt.index and len(income_stmt.columns) >= 2 else None
        ], 0.06, lambda x, y: x > y),

        # Risk Management Criteria
        ("Risk Management", "FCF/Debt (>20%)", [
            lambda: cashflow.loc['Free Cash Flow'].iloc[0] / balance_sheet.loc['Total Debt'].iloc[0],
            lambda: info.get('freeCashflow') / info.get('totalDebt')
        ], 0.2, lambda x, y: x > y)
    ]

    for category, metric, value_funcs, threshold, condition in criteria:
        value = None
        errors = []
        for func in value_funcs:
            try:
                val = func()
                if val is not None:
                    value = val
                    break
            except Exception as e:
                errors.append(str(e))
                continue
        if value is not None:
            add_eval(category, metric, value, threshold, condition)
        else:
            print(f"⚠️ Could not calculate {metric} for {ticker_symbol}. Errors: {', '.join(errors[:2])}")

    return evaluation

def compare_tickers(ticker_list):
    all_results = []

    for ticker in ticker_list:
        print(f"\n🔍 Analyzing {ticker}...")
        result = evaluate_stock(ticker)
        if result:
            all_results.extend(result)

    return pd.DataFrame(all_results) if all_results else None

def export_results(df, format_type='csv'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stock_analysis_{timestamp}"

    try:
        if format_type.lower() == 'csv':
            filepath = f"{filename}.csv"
            df.to_csv(filepath, index=False)
            print(f"\n💾 Saved to {os.path.abspath(filepath)}")
        elif format_type.lower() == 'excel':
            filepath = f"{filename}.xlsx"
            df.to_excel(filepath, index=False)
            print(f"\n💾 Saved to {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")

def display_results(df):
    print_header("detailed analysis")

    for ticker, group in df.groupby('Ticker'):
        print(f"\n📊 {ticker} Summary:")
        print("-" * (len(ticker) + 10))
        for category, cat_group in group.groupby('Category'):
            print(f"\n{category}:")
            print(cat_group[['Metric', 'Value', 'Threshold', 'Pass/Fail']].to_string(index=False))

    print_header("comparative analysis")
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        print_metric_group(metric, metric_data[['Ticker', 'Value', 'Threshold', 'Pass/Fail']])

    print_header("performance summary")
    summary = df.groupby('Ticker')['Pass/Fail'].apply(
        lambda x: f"{sum(x == '✅')}/{sum(x.isin(['✅', '❌']))} passed"
    )
    print(summary.to_string())

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("📈 Stock Analysis Tool")
    print(fill("This tool evaluates stocks using multiple calculation methods with fallback data sources.", width=80))

    while True:
        tickers = input("\nEnter tickers (comma-separated) or 'q' to quit: ").strip()
        if tickers.lower() == 'q':
            break

        tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]

        if not tickers:
            print("❌ Please enter at least one valid ticker")
            continue

        results = compare_tickers(tickers)

        if results is not None:
            display_results(results)
            export_choice = input("\nExport results? (csv/excel/no): ").strip().lower()
            if export_choice in ('csv', 'excel'):
                export_results(results, export_choice)
            elif export_choice != 'no':
                print("⚠️ Invalid choice - skipping export")

        print("\n" + "=" * 80)
