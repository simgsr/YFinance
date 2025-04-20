import numpy as np
from numpy_financial import npv  # Requires pip install numpy-financial

# Modified calculate_metrics function
def calculate_metrics(self, data):
    metrics = {
        'P/E Ratio': np.nan,
        'Debt/Equity': np.nan,
        'FCF Yield %': np.nan,
        'ROE': np.nan,
        'FCF Margin %': np.nan,
        'Price/Book': np.nan,
        'Current Ratio': np.nan,
        'Operating Margin %': np.nan,
        'Dividend Yield %': np.nan
    }

    try:
        # P/E Ratio
        if data.get('eps', 0) > 0 and data.get('price', 0) > 0:
            metrics['P/E Ratio'] = data['price'] / data['eps']

        # Debt/Equity
        if data.get('total_equity', 0) > 0:
            metrics['Debt/Equity'] = data['total_debt'] / data['total_equity']

        # ROE
        if data.get('total_equity', 0) > 0:
            metrics['ROE'] = data['net_income'] / data['total_equity']

        # FCF Yield
        if data.get('market_cap', 0) > 0:
            metrics['FCF Yield %'] = data['fcf'] / data['market_cap']

        # FCF Margin
        if data.get('revenue', 0) > 0:
            metrics['FCF Margin %'] = data['fcf'] / data['revenue']

        # Price/Book
        if data.get('book_value', 0) > 0:
            metrics['Price/Book'] = data['price'] / data['book_value']

        # Current Ratio
        if data.get('current_liabilities', 0) > 0:
            metrics['Current Ratio'] = data['current_assets'] / data['current_liabilities']

        # Operating Margin
        if data.get('revenue', 0) > 0:
            metrics['Operating Margin %'] = data['operating_income'] / data['revenue']

        # Dividend Yield
        if data.get('price', 0) > 0:
            metrics['Dividend Yield %'] = data['dividend_per_share'] / data['price']

    except Exception as e:
        print(f"⚠️ Metric calculation error: {str(e)}")

    return metrics

# Modified calculate_valuation function
def calculate_valuation(self, data):
    valuation = {'DCF': np.nan, 'P/E Relative': np.nan, 'P/FCF': np.nan}
    try:
        if all(k in data for k in ['fcf', 'terminal_growth', 'wacc', 'shares_outstanding']):
            if data['wacc'] <= data['terminal_growth']:
                raise ValueError("WACC must be greater than terminal growth rate")

            terminal_value = (data['fcf'] * (1 + data['terminal_growth'])) / (data['wacc'] - data['terminal_growth'])
            cash_flows = [data['fcf']] * 5 + [terminal_value]
            discount_factors = [(1 + data['wacc'])**(i+1) for i in range(len(cash_flows))]
            present_values = [cf/df for cf, df in zip(cash_flows, discount_factors)]
            dcf_value = sum(present_values)

            if data['shares_outstanding'] > 0:
                valuation['DCF'] = dcf_value / data['shares_outstanding']

        # Relative Valuation
        if data.get('eps', 0) > 0:
            valuation['P/E Relative'] = data['eps'] * 15

        # P/FCF Valuation
        if data.get('fcf', 0) > 0 and data.get('shares_outstanding', 0) > 0:
            valuation['P/FCF'] = (data['fcf'] / data['shares_outstanding']) * 18

    except Exception as e:
        print(f"⚠️ Valuation error: {str(e)}")

    return valuation

# Enhanced extract_financial_data
def extract_financial_data(text):
    patterns = {
        'revenue': r"Revenue[\s:]+([\d,\.]+)",
        'net_income': r"Net Income[\s:]+([\d,\.]+)",
        'total_equity': r"Total Equity[\s:]+([\d,\.]+)",
        'total_debt': r"Total Debt[\s:]+([\d,\.]+)",
        'fcf': r"Free Cash Flow[\s:]+([\d,\.]+)",
        'eps': r"EPS[\s:]+([\d,\.]+)",
        'shares_outstanding': r"Shares Outstanding[\s:]+([\d,\.]+)",
        'book_value': r"Book Value per Share[\s:]+([\d,\.]+)",
        'current_assets': r"Current Assets[\s:]+([\d,\.]+)",
        'current_liabilities': r"Current Liabilities[\s:]+([\d,\.]+)",
        'operating_income': r"Operating Income[\s:]+([\d,\.]+)",
        'dividend_per_share': r"Dividend per Share[\s:]+([\d,\.]+)"
    }

    data = {k: np.nan for k in patterns.keys()}

    for key, pattern in patterns.items():
        try:
            match = re.search(pattern, text.replace(",", ""))
            if match:
                data[key] = float(match.group(1))
        except:
            pass

    # Add market data with validation
    data['price'] = 5.00 if np.isnan(data.get('price')) else data['price']
    data['market_cap'] = data['price'] * data['shares_outstanding'] if not np.isnan(data['shares_outstanding']) else np.nan
    data['wacc'] = 0.08
    data['terminal_growth'] = 0.02

    return data

# Updated format_value function
def format_value(value, percentage=False):
    try:
        if np.isnan(value):
            return 'N/A'
        if percentage:
            return f"{value:.2%}".rstrip('%')
        return f"{value:,.2f}"
    except:
        return 'N/A'
