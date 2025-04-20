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
    """Use DeepSeek's value investing trained model for consistent analysis"""
    try:
        response = ollama.generate(
            model="deepseek-r1",
            prompt=prompt,
            options={
                'temperature': 0.1,  # Lower for more consistent, analytical responses
                'num_ctx': 16384,     # Larger context for detailed analysis
                'system': """You are a value investing analyst specializing in fundamental analysis.
                Always provide: 1) DCF, 2) P/E relative, and 3) P/FCF valuations with calculations.
                Give specific intrinsic value ranges and clear buy/sell recommendations."""
            }
        )
        return response["response"]
    except Exception as e:
        return f"⚠️ DeepSeek analysis failed: {str(e)}"

def get_stock_data(ticker):
    """Fetch stock data with enhanced validation and fallback calculations"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Validate and clean data
        if info.get('trailingPE') and info['trailingPE'] < 0:
            info['trailingPE'] = None  # Invalid negative P/E

        # Enhanced FCF calculation
        market_cap = info.get("marketCap")
        fcf = info.get("freeCashflow")
        current_price = info.get("currentPrice")
        shares_outstanding = info.get("sharesOutstanding")

        # Fallback FCF calculation
        if fcf is None:
            operating_cf = info.get("operatingCashflow")
            cap_ex = info.get("capitalExpenditures")
            if operating_cf is not None and cap_ex is not None:
                fcf = operating_cf + cap_ex
            elif operating_cf is not None:
                fcf = operating_cf

        # Fallback market cap calculation
        if market_cap is None and current_price and shares_outstanding:
            market_cap = current_price * shares_outstanding

        # Calculate derived metrics
        fcf_yield = fcf / market_cap if (fcf and market_cap) else None
        revenue = info.get("totalRevenue")
        fcf_margin = fcf / revenue if (fcf and revenue) else None
        pfcf = market_cap / fcf if (fcf and market_cap) else None

        # Get industry peers for relative valuation
        industry_peers = []
        try:
            industry_peers = stock.recommendations['Firm'].unique()[:5] if hasattr(stock, 'recommendations') else []
        except:
            pass

        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "price": current_price,
            "industry": info.get("industry", "N/A"),
            "sector": info.get("sector", "N/A"),
            "peers": industry_peers,
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
                "pfcf": pfcf,
                "eps": info.get("trailingEps"),
                "revenue_growth": info.get("revenueGrowth"),
                "beta": info.get("beta")
            }
        }
    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {str(e)}")
        return None

def evaluate_metric(value, threshold, compare):
    """Evaluate if a metric meets the criterion"""
    if value is None:
        return "N/A"
    if compare == "<=":
        return "Pass" if value <= threshold else "Fail"
    elif compare == ">=":
        return "Pass" if value >= threshold else "Fail"
    return "N/A"

def get_user_criteria(stock_data):
    """Get valuation criteria with improved defaults based on industry"""
    print(f"\n📊 Current Metrics Evaluation for {stock_data['ticker']} ({stock_data['name']}):")

    # Adjust defaults based on sector
    sector_adjustments = {
        "Technology": {"pe": 20, "fcf_yield": 0.03},
        "Financial Services": {"debt_equity": 1.0, "pb": 1.0},
        "Communication Services": {"pe": 18, "dividend_yield": 0.03}
    }

    adjusted_criteria = DEEPSEEK_DEFAULT_CRITERIA.copy()
    if stock_data['sector'] in sector_adjustments:
        for k, v in sector_adjustments[stock_data['sector']].items():
            adjusted_criteria[k]["threshold"] = v

    for metric, config in adjusted_criteria.items():
        current_val = stock_data["metrics"].get(metric, None)
        current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
        eval_result = evaluate_metric(current_val, config["threshold"], config["compare"])
        print(f"- {config['desc']}: {current_val_str} (Threshold: {config['compare']} {config['threshold']}): {eval_result}")

    customize = input("\nWould you like to customize criteria? (y/n): ").strip().lower()
    if customize != 'y':
        return {metric: config["threshold"] for metric, config in adjusted_criteria.items()}

    print("\n📝 Customize Your Criteria (Press Enter for Defaults):")
    user_criteria = {}
    for metric, config in adjusted_criteria.items():
        default = config["threshold"]
        current_val = stock_data["metrics"].get(metric, None)
        current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
        prompt = f"{config['desc']} [Current: {current_val_str}] (Default: {default}): "
        user_input = input(prompt).strip()
        user_criteria[metric] = float(user_input or default)
    return user_criteria

def generate_report(stock_data, user_criteria, analysis):
    """Generate enhanced report with consistent structure"""
    report = f"""
📈 **Advanced Stock Analysis Report** 📉
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Ticker: {stock_data['ticker']} ({stock_data['name']})
Industry: {stock_data['industry']} | Sector: {stock_data['sector']}
Current Price: ${stock_data['price']:.2f} {'(⚠️ Data limitations noted)' if any(v is None for v in stock_data['metrics'].values()) else ''}

🔍 **Key Metrics Evaluation:"""

    for metric, config in DEEPSEEK_DEFAULT_CRITERIA.items():
        current_val = stock_data["metrics"].get(metric, None)
        current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
        threshold = user_criteria[metric]
        eval_result = evaluate_metric(current_val, threshold, config["compare"])
        report += f"\n- {config['desc']}: {current_val_str} (Threshold: {config['compare']} {threshold}): {eval_result}"

    if stock_data["metrics"].get("pfcf"):
        report += f"\n- Price/FCF: {stock_data['metrics']['pfcf']:.2f}"
    if stock_data["metrics"].get("beta"):
        report += f"\n- Beta: {stock_data['metrics']['beta']:.2f}"
    if stock_data.get("peers"):
        report += f"\n- Industry Peers: {', '.join(stock_data['peers'][:3])}..."

    report += "\n\n📌 **Your Custom Criteria:"
    for metric in user_criteria:
        report += f"\n- {DEEPSEEK_DEFAULT_CRITERIA[metric]['desc']}: {DEEPSEEK_DEFAULT_CRITERIA[metric]['compare']} {user_criteria[metric]:.2f}"

    report += f"\n\n📊 **DeepSeek-V3 Fundamental Analysis:**\n{analysis}"
    return report

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("\n🛠️  Advanced Stock Value Investing Analyzer (w/ DeepSeek-V3)")
    print("🌟 Now with consistent 3-method intrinsic valuation\n")

    while True:
        ticker = input("\n🔢 Enter Stock Ticker (e.g., AAPL) or 'quit': ").strip().upper()
        if ticker.lower() == 'quit':
            break

        print(f"\n🔍 Fetching data for {ticker}...")
        stock_data = get_stock_data(ticker)
        if not stock_data:
            print("❌ Failed to fetch data. Please try another ticker.")
            continue

        user_criteria = get_user_criteria(stock_data)

        # Build comprehensive DeepSeek prompt
        prompt = f"""
Perform a comprehensive value investing analysis for {stock_data['ticker']} ({stock_data['name']}) using three valuation methods:

Company Overview:
- Current Price: ${stock_data['price']:.2f}
- Industry: {stock_data['industry']}
- Sector: {stock_data['sector']}
- Market Cap: {'${:,.2f}M'.format(stock_data['metrics'].get('market_cap')/1e6) if stock_data['metrics'].get('market_cap') else 'N/A'}

Key Metrics:
{'\n'.join(f"- {DEEPSEEK_DEFAULT_CRITERIA[metric]['desc']}: {stock_data['metrics'].get(metric, 'N/A')}"
          for metric in user_criteria)}
- Beta: {stock_data['metrics'].get('beta', 'N/A')}

Required Analysis:
1. Intrinsic Value Calculation (3 Methods):
   A. Discounted Cash Flow (DCF):
      - Use {stock_data['metrics'].get('revenue_growth', 'industry average')} revenue growth
      - Conservative terminal growth rate
      - Appropriate discount rate for sector
      - Show calculations and assumptions

   B. P/E Relative Valuation:
      - Compare to industry average P/E of {stock_data['peers'][0] if stock_data.get('peers') else 'sector'}
      - Adjust for company-specific risk factors
      - Show calculation

   C. P/FCF Valuation:
      - Use current FCF yield of {stock_data['metrics'].get('fcf_yield', 'N/A')}
      - Compare to industry norms
      - Show calculation

2. Valuation Range:
   - Provide specific intrinsic value range from all three methods
   - Highlight most reliable method based on data quality

3. Buy/Sell Recommendations:
   - Buy: <80% of lowest intrinsic value estimate
   - Strong Buy: <70% of lowest estimate
   - Sell: >120% of highest intrinsic value estimate
   - Strong Sell: >150% of highest estimate

4. Risk Assessment:
   - Key financial risks (debt, margins, etc.)
   - Industry/sector-specific risks
   - Competitive position

5. Final Recommendation:
   - Clear Buy/Hold/Sell with price targets
   - Time horizon for investment
   - Suggested portfolio allocation (%)
"""
        print("\n🤖 Running DeepSeek Fundamental Analysis...")
        analysis = analyze_with_deepseek(prompt)
        report = generate_report(stock_data, user_criteria, analysis)
        print(report)

        if input("\n💾 Save report? (y/n): ").lower() == 'y':
            filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"✅ Report saved as {filename}")

    print("\n🎉 Analysis complete! Goodbye!")
