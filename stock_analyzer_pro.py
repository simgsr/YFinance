import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import math
import subprocess
import sys
from textwrap import fill

# =====================
# EVALUATION CRITERIA
# =====================
CRITERIA = [
    ("Economic Moat", "ROIC (>12%)", [
        lambda income, balance, cashflow, info: (
            income.loc['Net Income'].iloc[0] /
            (balance.loc['Total Assets'].iloc[0] -
             balance.loc['Current Liabilities'].iloc[0])
        ),
        lambda income, balance, cashflow, info: info.get('returnOnInvestedCapital')
    ], 0.12, lambda x, y: x > y),

    ("Economic Moat", "FCF/Sales (>15%)", [
        lambda income, balance, cashflow, info: (
            cashflow.loc['Free Cash Flow'].iloc[0] /
            income.loc['Total Revenue'].iloc[0]
        ),
        lambda income, balance, cashflow, info: (
            info.get('freeCashflow') /
            info.get('totalRevenue')
        )
    ], 0.15, lambda x, y: x > y),

    ("Margin of Safety", "P/E (<15)", [
        lambda income, balance, cashflow, info: (
            info.get('trailingPE') or
            info.get('forwardPE')
        ),
        lambda income, balance, cashflow, info: (
            info.get('marketCap') /
            income.loc['Net Income'].iloc[0]
        )
    ], 15, lambda x, y: x < y),

    ("Margin of Safety", "P/B (<1.5)", [
        lambda income, balance, cashflow, info: info.get('priceToBook'),
        lambda income, balance, cashflow, info: (
            info.get('marketCap') /
            balance.loc['Total Stockholder Equity'].iloc[0]
        )
    ], 1.5, lambda x, y: x < y),

    ("Management Quality", "ROE (>15%)", [
        lambda income, balance, cashflow, info: (
            income.loc['Net Income'].iloc[0] /
            balance.loc['Total Stockholder Equity'].iloc[0]
        ),
        lambda income, balance, cashflow, info: info.get('returnOnEquity')
    ], 0.15, lambda x, y: x > y),

    ("Long-Term Focus", "Debt/Equity (<0.5)", [
        lambda income, balance, cashflow, info: info.get('debtToEquity'),
        lambda income, balance, cashflow, info: (
            balance.loc['Total Debt'].iloc[0] /
            balance.loc['Total Stockholder Equity'].iloc[0]
        )
    ], 0.5, lambda x, y: x < y),

    ("Long-Term Focus", "Interest Coverage (>8x)", [
        lambda income, balance, cashflow, info: (
            income.loc['EBIT'].iloc[0] /
            income.loc['Interest Expense'].iloc[0]
        ),
        lambda income, balance, cashflow, info: (
            info.get('operatingIncome') /
            info.get('interestExpense')
        )
    ], 8, lambda x, y: x > y),

    ("Long-Term Focus", "EPS CAGR (>6%)", [
        lambda income, balance, cashflow, info: (
            (income.loc['Diluted EPS'].iloc[0] /
             income.loc['Diluted EPS'].iloc[-1]) **
            (1 / (len(income.columns) - 1)) - 1
            if 'Diluted EPS' in income.index and len(income.columns) >= 2
            else None
        )
    ], 0.06, lambda x, y: x > y),

    ("Risk Management", "FCF/Debt (>20%)", [
        lambda income, balance, cashflow, info: (
            cashflow.loc['Free Cash Flow'].iloc[0] /
            balance.loc['Total Debt'].iloc[0]
        ),
        lambda income, balance, cashflow, info: (
            info.get('freeCashflow') /
            info.get('totalDebt')
        )
    ], 0.2, lambda x, y: x > y)
]

# =====================
# CORE FUNCTIONS
# =====================
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
            try:
                value = float(value.real) if isinstance(value, complex) else float(value)
                passed = condition(value, threshold) if threshold is not None else "N/A"
                evaluation.append({
                    'Ticker': ticker_symbol,
                    'Category': category,
                    'Metric': metric,
                    'Value': round(value, 4),
                    'Threshold': threshold,
                    'Pass/Fail': "✅" if passed == True else "❌" if passed == False else "N/A"
                })
            except (ValueError, TypeError):
                print(f"⚠️ Invalid value for {metric} in {ticker_symbol}: {value}")

    for category, metric, value_funcs, threshold, condition in CRITERIA:
        value = None
        for func in value_funcs:
            try:
                val = func(income_stmt, balance_sheet, cashflow, info)
                if val is not None:
                    value = val
                    break
            except:
                continue
        if value is not None:
            add_eval(category, metric, value, threshold, condition)

    return evaluation

def compare_tickers(ticker_list):
    all_results = []
    for ticker in ticker_list:
        print(f"\n🔍 Analyzing {ticker}...")
        result = evaluate_stock(ticker)
        if result:
            all_results.extend(result)
    return pd.DataFrame(all_results) if all_results else None

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

def save_high_performers(df, input_filename, output_file_prefix="high_performers"):
    total_criteria = len(CRITERIA)
    threshold = math.ceil(total_criteria * 0.85)

    scores = df.groupby('Ticker')['Pass/Fail'].apply(lambda x: sum(x == '✅'))
    high_performers = scores[scores >= threshold].index.tolist()

    if high_performers:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        input_base = os.path.splitext(os.path.basename(input_filename))[0]
        date_str = datetime.now().strftime("%Y%m%d")
        output_file = f"{script_name}_{input_base}_{date_str}.csv"

        output_data = []
        for ticker in high_performers:
            ticker_data = df[df['Ticker'] == ticker]
            passed = ticker_data[ticker_data['Pass/Fail'] == '✅']
            failed = ticker_data[ticker_data['Pass/Fail'] == '❌']
            failed_details = " | ".join(
                f"{row['Metric']} (Value: {row['Value']:.2f}, Threshold: {row['Threshold']})"
                for _, row in failed.iterrows()
            ) if not failed.empty else "All criteria met"
            output_data.append({
                'Ticker': ticker,
                'Score': scores[ticker],
                'Failed_Criteria': failed_details,
                'Total_Criteria': len(ticker_data)
            })

        high_performers_df = pd.DataFrame(output_data)
        high_performers_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved {len(high_performers)} high-performing tickers (≥85%) to {output_file}")
        return high_performers, output_file
    else:
        print(f"\nℹ️ No tickers met the 85% threshold ({threshold}/{total_criteria}) - no CSV generated")
        return [], None

def get_tickers_from_file(file_path):
    try:
        df = pd.read_csv(file_path)
        for col in ['Ticker', 'tickers', 'symbol']:
            if col in df.columns:
                tickers = df[col].dropna().str.strip().str.upper().tolist()
                seen = set()
                unique_tickers = [t for t in tickers if not (t in seen or seen.add(t))]
                return unique_tickers
        tickers = df.iloc[:, 0].dropna().str.strip().str.upper().tolist()
        seen = set()
        unique_tickers = [t for t in tickers if not (t in seen or seen.add(t))]
        return unique_tickers
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def run_external_stockinfo(tickers):
    """Execute external yf_get_stockinfo.py script with tickers"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), "yf_get_stockinfo.py")

        # Build command: python yf_get_stockinfo.py ticker1 ticker2...
        command = [sys.executable, script_path] + tickers

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print("\n" + result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Failed to run external script: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("📈 Stock Analysis Tool")
    print(fill("This tool evaluates stocks and optionally generates detailed financial data using external scripts.", width=80))

    while True:
        csv_path = input("\nPlease enter the path to your CSV file containing ticker symbols: ").strip()
        if not csv_path:
            print("❌ Please provide a file path")
            continue
        tickers = get_tickers_from_file(csv_path)
        if not tickers:
            print("❌ No valid tickers found in the file or file could not be read")
            continue
        print(f"\nFound {len(tickers)} tickers to analyze: {', '.join(tickers)}")
        break

    results = compare_tickers(tickers)
    if results is not None:
        display_results(results)
        high_performers, output_file = save_high_performers(results, csv_path)

        if high_performers:
            print(f"\nFound {len(high_performers)} high-performing tickers (≥85%): {', '.join(high_performers)}")
            while True:
                generate_info = input("\nGenerate detailed stock info using external script? (yes/no): ").lower().strip()
                if generate_info in ['yes', 'no']:
                    break
                print("Please enter 'yes' or 'no'")

            if generate_info == 'yes':
                print("\nExecuting external stock info script...")
                run_external_stockinfo(high_performers)

    print("\n" + "=" * 80)
