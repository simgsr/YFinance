import yfinance as yf
import ollama
from datetime import datetime
import json
import os
import numpy as np
from collections import defaultdict

# ======================
# CONFIGURATION
# ======================
DEEPSEEK_DEFAULT_CRITERIA = {
    "pe": {"threshold": 15, "compare": "<=", "desc": "P/E Ratio", "weight": 0.15},
    "debt_equity": {"threshold": 0.5, "compare": "<=", "desc": "Debt/Equity", "weight": 0.12},
    "fcf_yield": {"threshold": 0.05, "compare": ">=", "desc": "FCF Yield % (FCF/Price)", "weight": 0.13},
    "roe": {"threshold": 0.15, "compare": ">=", "desc": "Return on Equity", "weight": 0.10},
    "fcf_margin": {"threshold": 0.15, "compare": ">=", "desc": "FCF Margin %", "weight": 0.08},
    "pb": {"threshold": 1.5, "compare": "<=", "desc": "Price/Book", "weight": 0.10},
    "current_ratio": {"threshold": 1.5, "compare": ">=", "desc": "Current Ratio", "weight": 0.07},
    "op_margin": {"threshold": 0.15, "compare": ">=", "desc": "Operating Margin %", "weight": 0.10},
    "dividend_yield": {"threshold": 0.02, "compare": ">=", "desc": "Dividend Yield %", "weight": 0.05},
    "beta": {"threshold": 1.2, "compare": "<=", "desc": "Beta", "weight": 0.05}
}

HISTORICAL_DATA_FILE = "historical_analysis.json"
LEARNING_RATE = 0.1  # How quickly the system adapts to new information

# ======================
# SELF-LEARNING MODULE
# ======================
class AnalysisLearner:
    def __init__(self):
        self.historical_data = self.load_historical_data()
        self.method_weights = {
            "dcf": 0.4,
            "pe_relative": 0.3,
            "pfcf": 0.3
        }
        self.sector_adjustments = defaultdict(dict)
        self.metric_reliability = {metric: 0.8 for metric in DEEPSEEK_DEFAULT_CRITERIA}

    def load_historical_data(self):
        if os.path.exists(HISTORICAL_DATA_FILE):
            try:
                with open(HISTORICAL_DATA_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {"tickers": {}}
        return {"tickers": {}}

    def save_historical_data(self):
        with open(HISTORICAL_DATA_FILE, 'w') as f:
            json.dump(self.historical_data, f, indent=2)

    def update_method_weights(self, ticker, actual_price_movement):
        """Adjust valuation method weights based on prediction accuracy"""
        if ticker not in self.historical_data["tickers"]:
            return

        analysis = self.historical_data["tickers"][ticker]
        predictions = analysis.get("predictions", {})

        # Calculate prediction errors for each method
        errors = {}
        for method in self.method_weights:
            if method in predictions and predictions[method] is not None:
                predicted_change = (predictions[method] - analysis["initial_price"]) / analysis["initial_price"]
                actual_change = actual_price_movement
                errors[method] = abs(predicted_change - actual_change)

        if errors:
            # Inverse weighting - better methods get more weight
            total_error = sum(errors.values())
            if total_error > 0:
                for method in errors:
                    current_weight = self.method_weights[method]
                    error_ratio = errors[method] / total_error
                    # Adjust weight based on performance (lower error = higher weight)
                    self.method_weights[method] = current_weight * (1 - LEARNING_RATE) + \
                                                (1 - error_ratio) * LEARNING_RATE

            # Normalize weights to sum to 1
            total = sum(self.method_weights.values())
            if total > 0:
                self.method_weights = {k: v/total for k, v in self.method_weights.items()}

    def update_sector_adjustments(self, sector, metric, adjustment_factor):
        """Update sector-specific adjustments for metrics"""
        if metric not in self.sector_adjustments[sector]:
            self.sector_adjustments[sector][metric] = adjustment_factor
        else:
            # Moving average update
            self.sector_adjustments[sector][metric] = \
                self.sector_adjustments[sector][metric] * (1 - LEARNING_RATE) + \
                adjustment_factor * LEARNING_RATE

    def update_metric_reliability(self, metric, prediction_error):
        """Update how reliable each metric has been in predictions"""
        reliability = 1 / (1 + prediction_error)  # Inverse relationship
        self.metric_reliability[metric] = \
            self.metric_reliability[metric] * (1 - LEARNING_RATE) + \
            reliability * LEARNING_RATE

    def get_sector_adjusted_threshold(self, sector, metric):
        """Get sector-adjusted threshold for a metric"""
        default = DEEPSEEK_DEFAULT_CRITERIA[metric]["threshold"]
        if sector in self.sector_adjustments and metric in self.sector_adjustments[sector]:
            return default * self.sector_adjustments[sector][metric]
        return default

    def record_analysis(self, ticker, data, predictions, final_valuation):
        """Record analysis results for future learning"""
        if "tickers" not in self.historical_data:
            self.historical_data["tickers"] = {}

        self.historical_data["tickers"][ticker] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sector": data["sector"],
            "industry": data["industry"],
            "initial_price": data["price"],
            "metrics": data["metrics"],
            "predictions": predictions,
            "final_valuation": final_valuation,
            "method_weights": self.method_weights.copy()
        }
        self.save_historical_data()

# Initialize the learning system
learner = AnalysisLearner()

# ======================
# ANALYSIS MODULES
# ======================
def analyze_with_deepseek(prompt):
    """Use DeepSeek's value investing trained model for consistent analysis"""
    try:
        response = ollama.generate(
            model="deepseek-llm",
            prompt=prompt,
            options={
                'temperature': 0.3,
                'num_ctx': 8192,
                'system': """You are a value investing analyst specializing in fundamental analysis.
                Always provide: 1) DCF, 2) P/E relative, and 3) P/FCF valuations with calculations.
                Give specific intrinsic value ranges and clear buy/sell recommendations.
                Include confidence scores (0-100%) for each valuation method."""
            }
        )
        return response["response"]
    except Exception as e:
        return f"⚠️ DeepSeek analysis failed: {str(e)}"

def calculate_dcf(stock_data):
    """Calculate DCF valuation with self-learning improvements"""
    try:
        # Get sector-adjusted parameters
        sector = stock_data["sector"]
        revenue_growth = stock_data["metrics"].get("revenue_growth", -0.05)  # Default to -5% if missing
        op_margin = stock_data["metrics"].get("op_margin", 0.15)  # Default to 15% if missing

        # Base assumptions
        terminal_growth = 0.02  # 2% long-term growth
        discount_rate = 0.08    # 8% discount rate

        # Adjust based on historical accuracy
        if sector in learner.sector_adjustments:
            if "revenue_growth" in learner.sector_adjustments[sector]:
                revenue_growth_adjustment = learner.sector_adjustments[sector]["revenue_growth"] or 1
                revenue_growth *= revenue_growth_adjustment
            if "op_margin" in learner.sector_adjustments[sector]:
                op_margin_adjustment = learner.sector_adjustments[sector]["op_margin"] or 1
                op_margin *= op_margin_adjustment

        # Simplified DCF calculation
        fcf = stock_data["metrics"].get("freeCashflow")
        if fcf is None:
            # Estimate FCF if not available
            revenue = stock_data["metrics"].get("totalRevenue", 1e8)  # Default to 100M if missing
            fcf = revenue * (op_margin or 0.15) * 0.7  # Assume 70% conversion to FCF

        # Ensure we have a valid FCF value
        if fcf is None:
            return None, 0.5

        # 5-year projection
        years = 5
        projected_fcf = [fcf * (1 + (revenue_growth or 0))**i for i in range(1, years+1)]

        # Terminal value
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

        # Discount all cash flows
        discounted_cf = [cf / (1 + discount_rate)**i for i, cf in enumerate(projected_fcf, 1)]
        discounted_cf.append(terminal_value / (1 + discount_rate)**years)

        # Calculate equity value
        total_dcf = sum(discounted_cf)
        debt = stock_data["metrics"].get("totalDebt", 0) or 0
        cash = stock_data["metrics"].get("cash", 0) or 0
        equity_value = total_dcf - debt + cash

        # Calculate per share value
        shares = stock_data["metrics"].get("sharesOutstanding", 1e7)  # Default to 10M if missing
        if shares is None or shares <= 0:
            shares = 1e7

        dcf_value = equity_value / shares

        # Confidence score based on data completeness
        confidence = 0.7  # Base confidence
        if stock_data["metrics"].get("freeCashflow") is not None:
            confidence += 0.15
        if stock_data["metrics"].get("totalRevenue") is not None:
            confidence += 0.10
        if stock_data["metrics"].get("sharesOutstanding") is not None:
            confidence += 0.05
        confidence = min(0.95, max(0.5, confidence))  # Keep within reasonable bounds

        return dcf_value, confidence

    except Exception as e:
        print(f"DCF calculation error: {str(e)}")
        return None, 0.5

def calculate_pe_relative(stock_data):
    """Calculate P/E relative valuation with self-learning improvements"""
    try:
        pe = stock_data["metrics"].get("pe")
        eps = stock_data["metrics"].get("eps")

        if pe is None or eps is None:
            return None, 0.5

        # Get sector average P/E from historical data
        sector = stock_data["sector"]
        sector_pe = None

        # Find average P/E for this sector from historical analyses
        sector_pes = []
        for ticker, data in learner.historical_data.get("tickers", {}).items():
            if data.get("sector") == sector and data["metrics"].get("pe") is not None:
                sector_pes.append(data["metrics"]["pe"])

        if sector_pes:
            sector_pe = np.median(sector_pes)
        else:
            # Default sector P/E adjustments
            sector_pe_defaults = {
                "Technology": 20,
                "Financial Services": 12,
                "Communication Services": 18,
                "Consumer Cyclical": 15,
                "Healthcare": 16
            }
            sector_pe = sector_pe_defaults.get(sector, 15)

        # Adjust based on company-specific factors
        adjustment_factors = []

        # Higher growth gets higher P/E
        growth = stock_data["metrics"].get("revenue_growth", 0) or 0
        adjustment_factors.append(1 + growth)

        # Higher margins get higher P/E
        margins = stock_data["metrics"].get("op_margin", 0.1) or 0.1
        adjustment_factors.append(1 + (margins - 0.1) * 2)  # 10% margin = no adjustment

        # Lower debt gets higher P/E
        debt_equity = stock_data["metrics"].get("debt_equity", 0.5) or 0.5
        adjustment_factors.append(1 - min(0.5, debt_equity * 0.2))  # 0.5 D/E = 10% reduction

        # Composite adjustment
        adjustment = np.prod(adjustment_factors)
        adjusted_pe = sector_pe * adjustment

        # Calculate intrinsic value
        pe_value = (eps or 0) * adjusted_pe

        # Confidence score
        confidence = 0.6  # Base
        if len(sector_pes) > 3:  # Have good sector data
            confidence += 0.2
        if stock_data["metrics"].get("revenue_growth") is not None:
            confidence += 0.1
        if stock_data["metrics"].get("op_margin") is not None:
            confidence += 0.1
        confidence = min(0.9, max(0.4, confidence))

        return pe_value, confidence

    except Exception as e:
        print(f"P/E relative calculation error: {str(e)}")
        return None, 0.5

def calculate_pfcf(stock_data):
    """Calculate P/FCF valuation with self-learning improvements"""
    try:
        fcf = stock_data["metrics"].get("freeCashflow")
        shares = stock_data["metrics"].get("sharesOutstanding")

        if fcf is None or shares is None:
            return None, 0.5

        # Get sector average P/FCF from historical data
        sector = stock_data["sector"]
        sector_pfcf = None

        # Find average P/FCF for this sector from historical analyses
        sector_pfcfs = []
        for ticker, data in learner.historical_data.get("tickers", {}).items():
            if data.get("sector") == sector and data["metrics"].get("pfcf") is not None:
                sector_pfcfs.append(data["metrics"]["pfcf"])

        if sector_pfcfs:
            sector_pfcf = np.median(sector_pfcfs)
        else:
            # Default sector P/FCF adjustments
            sector_pfcf_defaults = {
                "Technology": 18,
                "Financial Services": 8,
                "Communication Services": 15,
                "Consumer Cyclical": 12,
                "Healthcare": 14
            }
            sector_pfcf = sector_pfcf_defaults.get(sector, 12)

        # Adjust based on company quality
        adjustment_factors = []

        # Higher growth gets higher multiple
        growth = stock_data["metrics"].get("revenue_growth", 0) or 0
        adjustment_factors.append(1 + growth * 0.5)

        # More stable FCF gets higher multiple
        # (For simplicity, we'll use current ratio as proxy for stability)
        current_ratio = stock_data["metrics"].get("current_ratio", 1.5) or 1.5
        adjustment_factors.append(1 + (current_ratio - 1.5) * 0.1)

        # Composite adjustment
        adjustment = np.prod(adjustment_factors)
        adjusted_pfcf = sector_pfcf * adjustment

        # Calculate intrinsic value
        fcf_per_share = (fcf or 0) / (shares or 1e7)  # Default to 10M shares if missing
        pfcf_value = fcf_per_share * adjusted_pfcf

        # Confidence score
        confidence = 0.7  # Base
        if len(sector_pfcfs) > 3:  # Have good sector data
            confidence += 0.15
        if stock_data["metrics"].get("revenue_growth") is not None:
            confidence += 0.1
        if stock_data["metrics"].get("current_ratio") is not None:
            confidence += 0.05
        confidence = min(0.95, max(0.5, confidence))

        return pfcf_value, confidence

    except Exception as e:
        print(f"P/FCF calculation error: {str(e)}")
        return None, 0.5

def get_stock_data(ticker):
    """Enhanced stock data fetching with reliability tracking"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Validate and clean data
        if info.get('trailingPE') and info['trailingPE'] < 0:
            info['trailingPE'] = None

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
                "beta": info.get("beta"),
                "freeCashflow": fcf,
                "totalRevenue": revenue,
                "sharesOutstanding": shares_outstanding,
                "totalDebt": info.get("totalDebt"),
                "cash": info.get("cash"),
                "marketCap": market_cap
            }
        }
    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {str(e)}")
        return None

def evaluate_metric(value, threshold, compare):
    """Evaluate if a metric meets the criterion"""
    if value is None:
        return "N/A", 0
    if compare == "<=":
        passed = value <= threshold
    elif compare == ">=":
        passed = value >= threshold
    else:
        return "N/A", 0

    # Calculate how much it passed/failed by (for learning)
    if compare == "<=":
        deviation = (value - threshold) / threshold if threshold != 0 else 0
    else:  # >=
        deviation = (threshold - value) / threshold if threshold != 0 else 0

    return "Pass" if passed else "Fail", deviation

def get_user_criteria(stock_data):
    """Get valuation criteria with self-learning improvements"""
    print(f"\n📊 Current Metrics Evaluation for {stock_data['ticker']} ({stock_data['name']}):")

    # Start with default criteria
    adjusted_criteria = DEEPSEEK_DEFAULT_CRITERIA.copy()

    # Apply sector adjustments from learning system
    sector = stock_data['sector']
    for metric in adjusted_criteria:
        adjusted_criteria[metric]["threshold"] = learner.get_sector_adjusted_threshold(sector, metric)

    # Evaluate metrics and show user
    for metric, config in adjusted_criteria.items():
        current_val = stock_data["metrics"].get(metric, None)
        threshold = config["threshold"]
        current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
        eval_result, deviation = evaluate_metric(current_val, threshold, config["compare"])
        print(f"- {config['desc']}: {current_val_str} (Threshold: {config['compare']} {threshold:.2f}): {eval_result}")

    customize = input("\nWould you like to customize criteria? (y/n): ").strip().lower()
    if customize != 'y':
        return {metric: config["threshold"] for metric, config in adjusted_criteria.items()}

    print("\n📝 Customize Your Criteria (Press Enter for Defaults):")
    user_criteria = {}
    for metric, config in adjusted_criteria.items():
        default = config["threshold"]
        current_val = stock_data["metrics"].get(metric, None)
        current_val_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else "N/A"
        prompt = f"{config['desc']} [Current: {current_val_str}] (Default: {default:.2f}): "
        user_input = input(prompt).strip()
        user_criteria[metric] = float(user_input or default)
    return user_criteria

def generate_report(stock_data, user_criteria, analysis, valuations):
    """Generate enhanced report with self-learning insights"""
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
        eval_result, _ = evaluate_metric(current_val, threshold, config["compare"])
        report += f"\n- {config['desc']}: {current_val_str} (Threshold: {config['compare']} {threshold:.2f}): {eval_result}"

    if stock_data["metrics"].get("pfcf"):
        report += f"\n- Price/FCF: {stock_data['metrics']['pfcf']:.2f}"
    if stock_data["metrics"].get("beta"):
        report += f"\n- Beta: {stock_data['metrics']['beta']:.2f}"
    if stock_data.get("peers"):
        report += f"\n- Industry Peers: {', '.join(stock_data['peers'][:3])}..."

    report += "\n\n📌 **Your Custom Criteria:"
    for metric in user_criteria:
        report += f"\n- {DEEPSEEK_DEFAULT_CRITERIA[metric]['desc']}: {DEEPSEEK_DEFAULT_CRITERIA[metric]['compare']} {user_criteria[metric]:.2f}"

    report += "\n\n📊 **Quantitative Valuation Summary:"
    for method, (value, confidence) in valuations.items():
        if value is not None:
            report += f"\n- {method.upper()} Valuation: ${value:.2f} (Confidence: {confidence*100:.0f}%)"

    # Calculate weighted average valuation
    valid_valuations = [(v, c) for v, c in valuations.values() if v is not None]
    if valid_valuations:
        values, confidences = zip(*valid_valuations)
        weights = [c/sum(confidences) for c in confidences]  # Normalized confidence weights
        weighted_avg = sum(v*w for v, w in zip(values, weights))
        report += f"\n\n🎯 Weighted Average Valuation: ${weighted_avg:.2f}"

        # Buy/sell ranges based on weighted average
        buy_price = weighted_avg * 0.8
        strong_buy = weighted_avg * 0.7
        sell_price = weighted_avg * 1.2
        strong_sell = weighted_avg * 1.5

        report += f"""

💰 **Recommendation Ranges:**
- Strong Buy: < ${strong_buy:.2f}
- Buy: < ${buy_price:.2f}
- Hold: ${buy_price:.2f} - ${sell_price:.2f}
- Sell: > ${sell_price:.2f}
- Strong Sell: > ${strong_sell:.2f}"""

    report += f"\n\n📊 **DeepSeek-V3 Fundamental Analysis:**\n{analysis}"
    return report

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("\n🛠️  Advanced Stock Value Investing Analyzer (w/ DeepSeek-V3)")
    print("🌟 Now with self-learning capabilities for improved consistency\n")

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

        # Calculate valuations using all methods
        valuations = {
            "dcf": calculate_dcf(stock_data),
            "pe_relative": calculate_pe_relative(stock_data),
            "pfcf": calculate_pfcf(stock_data)
        }

        # Build comprehensive DeepSeek prompt
        prompt = f"""
Perform a comprehensive value investing analysis for {stock_data['ticker']} ({stock_data['name']}) using three valuation methods:

Company Overview:
- Current Price: ${stock_data['price']:.2f}
- Industry: {stock_data['industry']}
- Sector: {stock_data['sector']}
- Market Cap: {'${:,.2f}M'.format(stock_data['metrics'].get('marketCap')/1e6) if stock_data['metrics'].get('marketCap') else 'N/A'}

Key Metrics:
{'\n'.join(f"- {DEEPSEEK_DEFAULT_CRITERIA[metric]['desc']}: {stock_data['metrics'].get(metric, 'N/A')}"
          for metric in user_criteria)}
- Beta: {stock_data['metrics'].get('beta', 'N/A')}

Quantitative Valuations:
{'\n'.join(f"- {method.upper()}: ${value:.2f} (Confidence: {confidence*100:.0f}%)"
          for method, (value, confidence) in valuations.items() if value is not None)}

Required Analysis:
1. Intrinsic Value Calculation (3 Methods):
   A. Discounted Cash Flow (DCF)
   B. P/E Relative Valuation
   C. P/FCF Valuation

2. Valuation Range: Provide specific intrinsic value range from all three methods
3. Buy/Sell Recommendations with price targets
4. Risk Assessment including data quality concerns
5. Final Recommendation with confidence level
6. Suggested portfolio allocation (%)
"""
        print("\n🤖 Running DeepSeek Fundamental Analysis...")
        analysis = analyze_with_deepseek(prompt)

        # Generate final valuation (weighted average of methods)
        valid_valuations = [(v, c) for v, c in valuations.values() if v is not None]
        if valid_valuations:
            values, confidences = zip(*valid_valuations)
            weights = [c/sum(confidences) for c in confidences]
            final_valuation = sum(v*w for v, w in zip(values, weights))
        else:
            final_valuation = None

        report = generate_report(stock_data, user_criteria, analysis, valuations)
        print(report)

        # Record this analysis for future learning
        predictions = {method: value for method, (value, _) in valuations.items()}
        learner.record_analysis(ticker, stock_data, predictions, final_valuation)

        if input("\n💾 Save report? (y/n): ").lower() == 'y':
            filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"✅ Report saved as {filename}")

        # Check if user wants to provide feedback for learning
        feedback = input("\n📊 Provide feedback for system learning? (Enter price change % or skip): ").strip()
        if feedback and final_valuation:
            try:
                price_change = float(feedback) / 100
                learner.update_method_weights(ticker, price_change)
                print("✅ Updated model weights based on your feedback")
            except ValueError:
                pass

    print("\n🎉 Analysis complete! The system has learned from this session.")
