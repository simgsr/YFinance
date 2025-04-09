import yfinance as yf
import ollama
from datetime import datetime

# ======================
# CONFIGURATION
# ======================
DEEPSEEK_DEFAULT_CRITERIA = {
    "pe": {"threshold": 15, "compare": "<=", "desc": "P/E Ratio"},
    "debt_equity": {"threshold": 0.5, "compare": "<=", "desc": "Debt/Equity"},
    "fcf_yield": {"threshold": 0.05, "compare": ">=", "desc": "FCF Yield % (FCF/Price)"},
    "roe": {"threshold": 0.15, "compare": ">=", "desc": "Return on Equity"},
    "fcf_margin": {"threshold": 0.15, "compare": ">=", "desc": "FCF Margin %"},
    "pb": {"threshold": 1.5, "compare": "<=", "desc": "Price/Book"},
    "current_ratio": {"threshold": 1.5, "compare": ">=", "desc": "Current Ratio"},
    "op_margin": {"threshold": 0.15, "compare": ">=", "desc": "Operating Margin %"},
    "dividend_yield": {"threshold": 0.02, "compare": ">=", "desc": "Dividend Yield %"}
}

# ======================
# ANALYSIS MODULES
# ======================
def analyze_with_deepseek(prompt):
    try:
        response = ollama.generate(model="deepseek-llm", prompt=prompt)
        return response["response"]
    except Exception as e:
        return f"⚠️ DeepSeek analysis failed: {str(e)}"

def get_stock_data(ticker):
    """Fetch stock data and derive FCF Yield if not directly available."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Primary attempt: Use freeCashflow and marketCap
        market_cap = info.get("marketCap")
        fcf = info.get("freeCashflow")
        current_price = info.get("currentPrice")
        shares_outstanding = info.get("sharesOutstanding")

        # If FCF or Market Cap is missing, derive them
        if fcf is None or market_cap is None:
            # Derive FCF: Operating Cash Flow - Capital Expenditures
            operating_cf = info.get("operatingCashflow")
            cap_ex = info.get("capitalExpenditures")  # Note: cap_ex is negative in yfinance
            if operating_cf is not None and cap_ex is not None:
                fcf = operating_cf + cap_ex  # Add because cap_ex is negative
            elif operating_cf is not None:
                fcf = operating_cf  # Fallback to operating CF if cap_ex unavailable

            # Derive Market Cap: Price × Shares Outstanding
            if current_price is not None and shares_outstanding is not None:
                market_cap = current_price * shares_outstanding

        # Calculate FCF metrics
        fcf_yield = fcf / market_cap if (fcf is not None and market_cap is not None) else None
        revenue = info.get("totalRevenue")
        fcf_margin = fcf / revenue if (fcf is not None and revenue is not None) else None
        pfcf = market_cap / fcf if (fcf is not None and market_cap is not None) else None

        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "price": current_price,
            "industry": info.get("industry", "N/A"),
            "sector": info.get("sector", "N/A"),
            "metrics": {
                "pe": info.get("trailingPE"),
                "debt_equity": info.get("debtToEquity"),
                "fcf_yield": fcf_yield,
                "roe": info.get("returnOnEquity"),
                "fcf_margin": fcf_margin,
                "pb": info.get("priceToBook"),
                "current_ratio": info.get("currentRatio"),
                "op_margin": info.get("operatingMargins"),
                "dividend_yield": info.get("dividendYield", 0),
                "pfcf": pfcf
            }
        }
    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {str(e)}")
        return None

def evaluate_metric(value, threshold, compare):
    """Evaluate if a metric meets the criterion."""
    if value is None:
        return "N/A"
    if compare == "<=":
        return "Pass" if value <= threshold else "Fail"
    elif compare == ">=":
        return "Pass" if value >= threshold else "Fail"
    return "N/A"

def get_user_criteria(stock_data):
    """Evaluate metrics against criteria and allow customization."""
    print(f"\n📊 Current Metrics Evaluation for {stock_data['ticker']} ({stock_data['name']}):")
    for metric, config in DEEPSEEK_DEFAULT_CRITERIA.items():
        current_val = stock_data["metrics"].get(metric, None)
        current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
        eval_result = evaluate_metric(current_val, config["threshold"], config["compare"])
        print(f"- {config['desc']}: {current_val_str} (Threshold: {config['compare']} {config['threshold']}): {eval_result}")

    customize = input("\nWould you like to customize criteria? (y/n): ").strip().lower()
    if customize != 'y':
        return {metric: config["threshold"] for metric, config in DEEPSEEK_DEFAULT_CRITERIA.items()}

    print("\n📝 Customize Your Criteria (Press Enter for Defaults):")
    user_criteria = {}
    for metric, config in DEEPSEEK_DEFAULT_CRITERIA.items():
        default = config["threshold"]
        current_val = stock_data["metrics"].get(metric, None)
        current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
        prompt = f"{config['desc']} [Current: {current_val_str}] (Default: {default}): "
        user_input = input(prompt).strip()
        user_criteria[metric] = float(user_input or default)
    return user_criteria

def generate_report(stock_data, user_criteria, analysis):
    """Generate report with evaluation criteria."""
    report = f"""
📈 **Advanced Stock Analysis Report** 📉
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Ticker: {stock_data['ticker']} ({stock_data['name']})
Industry: {stock_data['industry']} | Sector: {stock_data['sector']}
Current Price: ${stock_data['price']:.2f}

🔍 **Key Metrics Evaluation:"""
    for metric, config in DEEPSEEK_DEFAULT_CRITERIA.items():
        current_val = stock_data["metrics"].get(metric, None)
        current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
        threshold = user_criteria[metric]
        eval_result = evaluate_metric(current_val, threshold, config["compare"])
        report += f"\n- {config['desc']}: {current_val_str} (Threshold: {config['compare']} {threshold}): {eval_result}"
    if stock_data["metrics"].get("pfcf"):
        report += f"\n- Price/FCF: {stock_data['metrics']['pfcf']:.2f}"

    report += "\n\n📌 **Your Custom Criteria:"
    for metric in user_criteria:
        report += f"\n- {DEEPSEEK_DEFAULT_CRITERIA[metric]['desc']}: {DEEPSEEK_DEFAULT_CRITERIA[metric]['compare']} {user_criteria[metric]:.2f}"

    report += f"\n\n📊 **DeepSeek-V3 Analysis:**\n{analysis}"
    return report

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("\n🛠️  Advanced Stock Value Investing Analyzer (w/ DeepSeek-V3)")

    while True:
        ticker = input("\n🔢 Enter Stock Ticker (e.g., AAPL) or 'quit': ").strip().upper()
        if ticker.lower() == 'quit':
            break

        stock_data = get_stock_data(ticker)
        if not stock_data:
            print("❌ Failed to fetch data. Please try another ticker.")
            continue

        user_criteria = get_user_criteria(stock_data)

        # Build DeepSeek prompt
        prompt = f"""
Perform a comprehensive value investing analysis for {stock_data['ticker']} ({stock_data['name']}):

Current Price: ${stock_data['price']:.2f}
Industry: {stock_data['industry']}

Key Metrics:"""
        for metric in user_criteria:
            current_val = stock_data["metrics"].get(metric, None)
            current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
            prompt += f"\n- {DEEPSEEK_DEFAULT_CRITERIA[metric]['desc']}: {current_val_str} (Your Threshold: {user_criteria[metric]:.2f})"
        if stock_data["metrics"].get("pfcf"):
            prompt += f"\n- Price/FCF: {stock_data['metrics']['pfcf']:.2f}"

        prompt += f"""

Required Analysis:
1. Calculate intrinsic value using 3 methods (DCF, P/E relative, P/FCF)
2. Suggest specific buy (<80% of fair value) and sell (>120% of fair value) prices
3. Evaluate against all {len(user_criteria)} custom criteria
4. Provide detailed Buy/Hold/Sell recommendation
5. Highlight key risks and opportunities
"""

        print("\n🤖 Running DeepSeek Analysis...")
        analysis = analyze_with_deepseek(prompt)
        report = generate_report(stock_data, user_criteria, analysis)
        print(report)

        if input("💾 Save report? (y/n): ").lower() == 'y':
            filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"✅ Report saved as {filename}")

    print("\n🎉 Analysis complete! Goodbye!")
