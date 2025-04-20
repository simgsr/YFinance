import os
import sys
import csv
import time
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from pandas_market_calendars import get_calendar

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('value_investing_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Analysis criteria configuration
ANALYSIS_CRITERIA = {
    "pe": {"threshold": 15, "compare": "<=", "desc": "P/E Ratio"},
    "debt_equity": {"threshold": 0.5, "compare": "<=", "desc": "Debt/Equity"},
    "fcf_yield": {"threshold": 0.05, "compare": ">=", "desc": "FCF Yield % (FCF/Price)"},
    "roe": {"threshold": 0.15, "compare": ">=", "desc": "Return on Equity"},
    "fcf_margin": {"threshold": 0.15, "compare": ">=", "desc": "FCF Margin %"},
    "pb": {"threshold": 1.5, "compare": "<=", "desc": "Price/Book"},
    "current_ratio": {"threshold": 1.5, "compare": ">=", "desc": "Current Ratio"},
    "op_margin": {"threshold": 0.15, "compare": ">=", "desc": "Operating Margin %"},
    "dividend_yield": {"threshold": 0.02, "compare": ">=", "desc": "Dividend Yield %"},
    "beta": {"threshold": 1.2, "compare": "<=", "desc": "Beta"}
}

# Global weights for valuation methods
GLOBAL_WEIGHTS = {
    'dcf': 0.35,
    'pe_relative': 0.25,
    'pfcf': 0.20,
    'ddm': 0.10,
    'graham': 0.10
}

# Constants
START_DATE = datetime(2010, 1, 1)
RISK_FREE_RATE = 0.03  # 3% as long-term risk-free rate
MARKET_RETURN = 0.08  # 8% as expected market return
LEARNING_RATE = 0.1
LEARNING_DECAY = 0.995  # Decay factor per week

class ValueInvestingAnalyzer:
    def __init__(self):
        self.tickers = []
        self.historical_data = {}
        self.fundamentals_cache = {}
        self.valuation_results = {}
        self.model_weights = defaultdict(lambda: GLOBAL_WEIGHTS.copy())
        self.sector_weights = defaultdict(lambda: GLOBAL_WEIGHTS.copy())
        self.performance_metrics = defaultdict(list)
        self.missing_data_stats = defaultdict(int)
        self.current_ticker = None
        self.resume_state = {}
        self.load_state()

    def load_state(self):
        """Load previous state from file if exists"""
        try:
            with open('analysis_state.pkl', 'rb') as f:
                self.resume_state = pickle.load(f)
            logger.info("Resume state loaded successfully")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            self.resume_state = {}
            logger.info("No resume state found, starting fresh")

    def save_state(self):
        """Save current state to file"""
        self.resume_state = {
            'current_ticker': self.current_ticker,
            'processed_dates': {ticker: list(data.keys()) for ticker, data in self.valuation_results.items()},
            'model_weights': dict(self.model_weights),
            'sector_weights': dict(self.sector_weights),
            'performance_metrics': dict(self.performance_metrics),
            'missing_data_stats': dict(self.missing_data_stats)
        }
        with open('analysis_state.pkl', 'wb') as f:
            pickle.dump(self.resume_state, f)
        logger.info("Analysis state saved successfully")

    def load_tickers(self, input_path: str):
        """Load tickers from CSV file or single ticker input"""
        if os.path.isfile(input_path):
            try:
                with open(input_path, 'r') as f:
                    reader = csv.reader(f)
                    self.tickers = [row[0].strip().upper() for row in reader if row]
                logger.info(f"Loaded {len(self.tickers)} tickers from file")
            except Exception as e:
                logger.error(f"Error loading tickers from file: {e}")
                raise
        else:
            # Assume it's a single ticker
            self.tickers = [input_path.strip().upper()]
            logger.info(f"Processing single ticker: {self.tickers[0]}")

        # Validate tickers
        self.validate_tickers()

    def validate_tickers(self):
        """Validate that tickers exist and can be fetched"""
        valid_tickers = []
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if info.get('symbol', '').upper() == ticker.upper():
                    valid_tickers.append(ticker)
                    logger.info(f"Validated ticker: {ticker}")
                else:
                    logger.warning(f"Invalid ticker: {ticker} - not found in yfinance")
            except Exception as e:
                logger.warning(f"Error validating ticker {ticker}: {e}")

        self.tickers = valid_tickers
        if not self.tickers:
            raise ValueError("No valid tickers found to analyze")

    def get_historical_data(self, ticker: str):
        """Fetch weekly historical data for a ticker"""
        if ticker in self.historical_data:
            return self.historical_data[ticker]

        logger.info(f"Fetching weekly historical data for {ticker}")

        try:
            # Get NYSE calendar for business days
            nyse = get_calendar('NYSE')
            holidays = nyse.holidays().holidays

            # Fetch daily data and resample to weekly (Wednesday close)
            stock = yf.Ticker(ticker)
            df = stock.history(start=START_DATE, interval='1d')

            if df.empty:
                logger.warning(f"No historical data found for {ticker}")
                return None

            # Create weekly data with Wednesday close (or Thursday if Wednesday is holiday)
            weekly_data = {}
            current_date = START_DATE
            while current_date < datetime.now():
                # Find next Wednesday
                wednesday = current_date + timedelta(days=(2 - current_date.weekday() + 7) % 7)

                # Adjust if Wednesday is a holiday
                if wednesday in holidays:
                    wednesday += BDay(1)  # Move to next business day (Thursday)

                # Get the close price for this adjusted Wednesday
                if wednesday in df.index:
                    weekly_data[wednesday.date()] = {
                        'close': df.loc[wednesday, 'Close'],
                        'volume': df.loc[wednesday, 'Volume'],
                        'dividends': df.loc[wednesday:wednesday + timedelta(days=7), 'Dividends'].sum()
                    }

                current_date = wednesday + timedelta(days=1)

            self.historical_data[ticker] = weekly_data
            return weekly_data
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None

    def get_fundamentals(self, ticker: str, date: datetime.date):
        """Get fundamental data for a ticker at a specific date"""
        cache_key = f"{ticker}_{date}"
        if cache_key in self.fundamentals_cache:
            return self.fundamentals_cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get financial statements
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cash_flow

            # Find the most recent financials before our analysis date
            bs_date = balance_sheet.columns[balance_sheet.columns <= pd.Timestamp(date)].max()
            is_date = income_stmt.columns[income_stmt.columns <= pd.Timestamp(date)].max()
            cf_date = cash_flow.columns[cash_flow.columns <= pd.Timestamp(date)].max()

            fundamentals = {
                'pe': info.get('trailingPE', None),
                'pb': info.get('priceToBook', None),
                'debt_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'beta': info.get('beta', None),
                'dividend_yield': info.get('dividendYield', None),
                'roe': None,
                'fcf_yield': None,
                'fcf_margin': None,
                'op_margin': None,
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', None)
            }

            # Calculate ROE (Net Income / Shareholder's Equity)
            if bs_date and is_date:
                net_income = income_stmt[is_date].get('Net Income', None)
                equity = balance_sheet[bs_date].get('Total Stockholder Equity', None)
                if net_income and equity and equity != 0:
                    fundamentals['roe'] = net_income / equity

            # Calculate FCF Yield (FCF / Market Cap)
            if cf_date and fundamentals['market_cap']:
                fcf = cash_flow[cf_date].get('Free Cash Flow', None)
                if fcf and fundamentals['market_cap'] > 0:
                    fundamentals['fcf_yield'] = fcf / fundamentals['market_cap']

            # Calculate FCF Margin (FCF / Revenue)
            if cf_date and is_date:
                fcf = cash_flow[cf_date].get('Free Cash Flow', None)
                revenue = income_stmt[is_date].get('Total Revenue', None)
                if fcf and revenue and revenue != 0:
                    fundamentals['fcf_margin'] = fcf / revenue

            # Calculate Operating Margin (Operating Income / Revenue)
            if is_date:
                op_income = income_stmt[is_date].get('Operating Income', None)
                revenue = income_stmt[is_date].get('Total Revenue', None)
                if op_income and revenue and revenue != 0:
                    fundamentals['op_margin'] = op_income / revenue

            # Track missing data points
            for metric, value in fundamentals.items():
                if value is None and metric in ANALYSIS_CRITERIA:
                    self.missing_data_stats[metric] += 1

            self.fundamentals_cache[cache_key] = fundamentals
            return fundamentals
        except Exception as e:
            logger.error(f"Error getting fundamentals for {ticker} on {date}: {e}")
            return None

    def analyze_ticker(self, ticker: str):
        """Perform complete analysis for a single ticker"""
        self.current_ticker = ticker
        logger.info(f"Starting analysis for {ticker}")

        # Get historical data
        historical_data = self.get_historical_data(ticker)
        if not historical_data:
            logger.warning(f"No historical data available for {ticker}, skipping")
            return

        # Get sector information
        sector = self.get_fundamentals(ticker, datetime.now().date()).get('sector', 'Unknown')

        # Process each week chronologically
        dates = sorted(historical_data.keys())
        total_weeks = len(dates)

        for i, analysis_date in enumerate(dates):
            # Check if we should resume from this date
            if ticker in self.resume_state.get('processed_dates', {}) and \
               analysis_date in self.resume_state['processed_dates'][ticker]:
                continue

            logger.info(f"Processing {ticker} - {analysis_date} ({i+1}/{total_weeks})")

            # Get fundamentals for this date
            fundamentals = self.get_fundamentals(ticker, analysis_date)
            if not fundamentals:
                logger.warning(f"Missing fundamentals for {ticker} on {analysis_date}, skipping")
                continue

            current_price = historical_data[analysis_date]['close']

            # Perform valuation
            valuation = self.perform_valuation(ticker, analysis_date, current_price, fundamentals)

            # Generate projections
            projections = self.generate_projections(ticker, analysis_date, current_price, fundamentals)

            # Evaluate predictions against actual future prices
            self.evaluate_predictions(ticker, analysis_date, valuation, projections)

            # Store results
            if ticker not in self.valuation_results:
                self.valuation_results[ticker] = {}

            self.valuation_results[ticker][analysis_date] = {
                'fundamentals': fundamentals,
                'valuation': valuation,
                'projections': projections,
                'actual_prices': self.get_actual_prices(ticker, analysis_date)
            }

            # Update model weights based on performance
            self.update_model_weights(ticker, sector, analysis_date)

            # Save progress periodically
            if i % 10 == 0:
                self.save_state()

        # Generate final report for this ticker
        self.generate_report(ticker)
        logger.info(f"Completed analysis for {ticker}")

    def perform_valuation(self, ticker: str, date: datetime.date, current_price: float,
                         fundamentals: Dict) -> Dict:
        """Perform all valuation methods"""
        valuations = {}

        # Discounted Cash Flow
        valuations['dcf'] = self.dcf_valuation(ticker, date, fundamentals)

        # P/E Relative Valuation
        valuations['pe_relative'] = self.pe_relative_valuation(ticker, date, fundamentals)

        # P/FCF Valuation
        valuations['pfcf'] = self.pfcf_valuation(ticker, date, fundamentals)

        # Dividend Discount Model
        valuations['ddm'] = self.ddm_valuation(ticker, date, fundamentals)

        # Graham Number
        valuations['graham'] = self.graham_valuation(ticker, date, fundamentals)

        # Calculate weighted average intrinsic value
        weights = self.get_weights(ticker, fundamentals.get('sector', 'Unknown'))
        weighted_sum = sum(valuations[method] * weights[method] for method in valuations)
        total_weight = sum(weights[method] for method in valuations if valuations[method] is not None)

        if total_weight > 0:
            valuations['weighted_avg'] = weighted_sum / total_weight
        else:
            valuations['weighted_avg'] = None

        return valuations

    def dcf_valuation(self, ticker: str, date: datetime.date, fundamentals: Dict) -> Optional[float]:
        """Discounted Cash Flow valuation"""
        try:
            # Get required fundamentals
            fcf = self.get_fcf(ticker, date)
            growth_rate = self.estimate_growth_rate(ticker, date)
            if fcf is None or growth_rate is None:
                return None

            # Conservative assumptions
            terminal_growth_rate = 0.02  # 2% terminal growth
            discount_rate = self.calculate_discount_rate(fundamentals.get('beta', 1.0))

            # 10-year DCF model
            years = 10
            future_fcfs = []

            for year in range(1, years + 1):
                if year <= 5:
                    # Higher growth for first 5 years
                    future_fcf = fcf * (1 + growth_rate) ** year
                else:
                    # Slowing growth for next 5 years
                    reduced_growth = growth_rate * (1 - (year - 5) * 0.1)
                    future_fcf = future_fcfs[-1] * (1 + max(reduced_growth, terminal_growth_rate))

                future_fcfs.append(future_fcf)

            # Terminal value
            terminal_value = future_fcfs[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)

            # Discount all future cash flows
            present_value = 0
            for year, future_fcf in enumerate(future_fcfs, 1):
                present_value += future_fcf / (1 + discount_rate) ** year

            present_value += terminal_value / (1 + discount_rate) ** years

            # Get shares outstanding
            shares_outstanding = self.get_shares_outstanding(ticker, date)
            if shares_outstanding is None:
                return None

            intrinsic_value = present_value / shares_outstanding
            return intrinsic_value
        except Exception as e:
            logger.error(f"Error in DCF valuation for {ticker} on {date}: {e}")
            return None

    def pe_relative_valuation(self, ticker: str, date: datetime.date,
                            fundamentals: Dict) -> Optional[float]:
        """P/E Relative valuation"""
        try:
            if not fundamentals.get('pe'):
                return None

            # Get sector average P/E
            sector = fundamentals.get('sector', 'Unknown')
            sector_pe = self.get_sector_average_pe(sector, date)

            if sector_pe is None:
                # Use historical average P/E for the stock
                historical_pe = self.get_historical_average_pe(ticker, date)
                if historical_pe is None:
                    return None
                fair_pe = historical_pe
            else:
                # Use sector P/E with adjustment for company quality
                quality_factor = 1.0
                if fundamentals.get('roe', 0) > 0.2:
                    quality_factor *= 1.1
                if fundamentals.get('debt_equity', 1) < 0.3:
                    quality_factor *= 1.05

                fair_pe = sector_pe * quality_factor

            # Get EPS
            eps = self.get_eps(ticker, date)
            if eps is None:
                return None

            intrinsic_value = fair_pe * eps
            return intrinsic_value
        except Exception as e:
            logger.error(f"Error in P/E relative valuation for {ticker} on {date}: {e}")
            return None

    def pfcf_valuation(self, ticker: str, date: datetime.date,
                      fundamentals: Dict) -> Optional[float]:
        """Price to Free Cash Flow valuation"""
        try:
            fcf = self.get_fcf(ticker, date)
            shares_outstanding = self.get_shares_outstanding(ticker, date)

            if fcf is None or shares_outstanding is None:
                return None

            # Get sector average P/FCF
            sector = fundamentals.get('sector', 'Unknown')
            sector_pfcf = self.get_sector_average_pfcf(sector, date)

            if sector_pfcf is None:
                # Use historical average P/FCF for the stock
                historical_pfcf = self.get_historical_average_pfcf(ticker, date)
                if historical_pfcf is None:
                    return None
                fair_pfcf = historical_pfcf
            else:
                # Use sector P/FCF with adjustment
                fair_pfcf = sector_pfcf * 0.9  # Be slightly more conservative

            fcf_per_share = fcf / shares_outstanding
            intrinsic_value = fair_pfcf * fcf_per_share
            return intrinsic_value
        except Exception as e:
            logger.error(f"Error in P/FCF valuation for {ticker} on {date}: {e}")
            return None

    def ddm_valuation(self, ticker: str, date: datetime.date,
                     fundamentals: Dict) -> Optional[float]:
        """Dividend Discount Model valuation"""
        try:
            dividend = fundamentals.get('dividend_yield')
            if dividend is None or dividend <= 0:
                return None

            # Get current dividend per share
            stock = yf.Ticker(ticker)
            dividend_history = stock.dividends

            if dividend_history.empty:
                return None

            # Get most recent dividend before our analysis date
            recent_dividends = dividend_history[dividend_history.index <= pd.Timestamp(date)]
            if recent_dividends.empty:
                return None

            last_dividend = recent_dividends.iloc[-1]

            # Estimate growth rate
            growth_rate = self.estimate_dividend_growth_rate(ticker, date)
            if growth_rate is None:
                growth_rate = 0.02  # Default to 2% if no history

            # Cap growth rate at reasonable level
            growth_rate = min(growth_rate, 0.08)

            # Calculate required return
            required_return = self.calculate_discount_rate(fundamentals.get('beta', 1.0))

            # Gordon Growth Model
            if growth_rate >= required_return:
                return None  # Model doesn't work if growth >= discount rate

            intrinsic_value = last_dividend * (1 + growth_rate) / (required_return - growth_rate)
            return intrinsic_value
        except Exception as e:
            logger.error(f"Error in DDM valuation for {ticker} on {date}: {e}")
            return None

    def graham_valuation(self, ticker: str, date: datetime.date,
                        fundamentals: Dict) -> Optional[float]:
        """Graham Number valuation"""
        try:
            eps = self.get_eps(ticker, date)
            book_value = self.get_book_value_per_share(ticker, date)

            if eps is None or book_value is None:
                return None

            # Original Graham formula
            intrinsic_value = np.sqrt(22.5 * eps * book_value)
            return intrinsic_value
        except Exception as e:
            logger.error(f"Error in Graham valuation for {ticker} on {date}: {e}")
            return None

    def generate_projections(self, ticker: str, date: datetime.date,
                           current_price: float, fundamentals: Dict) -> Dict:
        """Generate 1/2/3 year price projections"""
        projections = {}

        # Get intrinsic value estimates
        valuations = self.perform_valuation(ticker, date, current_price, fundamentals)
        weighted_avg = valuations.get('weighted_avg')

        if weighted_avg is None:
            return projections

        # Calculate expected return based on margin of safety
        margin_of_safety = (weighted_avg - current_price) / weighted_avg if weighted_avg != 0 else 0

        # Base projection assumes price moves toward intrinsic value
        for years in [1, 2, 3]:
            # Adjust based on margin of safety and historical volatility
            if margin_of_safety > 0.2:  # Undervalued by >20%
                projection = current_price * (1 + margin_of_safety * 0.5) ** years
            elif margin_of_safety < -0.2:  # Overvalued by >20%
                projection = current_price * (1 + margin_of_safety * 0.3) ** years
            else:  # Fairly valued
                # Use historical growth rate
                growth_rate = self.estimate_growth_rate(ticker, date) or 0.05
                projection = current_price * (1 + growth_rate) ** years

            projections[f'{years}_year'] = projection

        return projections

    def evaluate_predictions(self, ticker: str, date: datetime.date,
                           valuation: Dict, projections: Dict):
        """Evaluate predictions against actual future prices"""
        actual_prices = self.get_actual_prices(ticker, date)
        if not actual_prices:
            return

        # Calculate errors for each valuation method
        errors = {}
        current_price = self.historical_data[ticker][date]['close']

        for method, intrinsic_value in valuation.items():
            if intrinsic_value is None:
                continue

            # Calculate percentage error for each time horizon
            method_errors = {}
            for years, actual_price in actual_prices.items():
                if actual_price is None:
                    continue

                # Calculate expected price based on valuation
                if method == 'weighted_avg':
                    # For weighted average, use the projection directly
                    projected_price = projections.get(f'{years}_year')
                    if projected_price is None:
                        continue
                else:
                    # For individual methods, assume price moves toward intrinsic value
                    if intrinsic_value > current_price:
                        # If undervalued, assume price increases to intrinsic value over 3 years
                        projected_price = current_price + (intrinsic_value - current_price) * (years / 3)
                    else:
                        # If overvalued, assume price decreases to intrinsic value over 1 year
                        projected_price = current_price + (intrinsic_value - current_price) * min(years, 1)

                if projected_price is not None and actual_price != 0:
                    error_pct = abs(projected_price - actual_price) / actual_price
                    method_errors[years] = error_pct

            if method_errors:
                errors[method] = method_errors

        # Store performance metrics
        if errors:
            self.performance_metrics[ticker].append({
                'date': date,
                'errors': errors,
                'actual_prices': actual_prices,
                'valuation': valuation,
                'current_price': current_price
            })

    def update_model_weights(self, ticker: str, sector: str, date: datetime.date):
        """Update model weights based on recent performance"""
        if ticker not in self.performance_metrics or not self.performance_metrics[ticker]:
            return

        # Get recent performance data (last 12 months)
        recent_performance = [
            pm for pm in self.performance_metrics[ticker]
            if (date - pm['date']).days <= 365
        ]

        if not recent_performance:
            return

        # Calculate average errors for each method
        method_errors = defaultdict(list)
        for pm in recent_performance:
            for method, errors in pm['errors'].items():
                for year, error in errors.items():
                    method_errors[method].append(error)

        avg_errors = {
            method: np.mean(errors) if errors else 1.0  # Default to high error if no data
            for method, errors in method_errors.items()
        }

        # Calculate new weights based on inverse errors
        total_inverse_error = sum(1 / (e + 0.0001) for e in avg_errors.values())  # Add small constant to avoid division by zero
        if total_inverse_error == 0:
            return

        # Calculate new weights with learning rate decay
        decayed_learning_rate = LEARNING_RATE * (LEARNING_DECAY ** len(self.performance_metrics[ticker]))

        new_weights = {}
        for method in avg_errors:
            # Calculate target weight based on performance
            target_weight = (1 / (avg_errors[method] + 0.0001)) / total_inverse_error

            # Blend with current weight using learning rate
            current_weight = self.model_weights[ticker].get(method, GLOBAL_WEIGHTS[method])
            new_weight = (1 - decayed_learning_rate) * current_weight + decayed_learning_rate * target_weight
            new_weights[method] = new_weight

        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in new_weights.items()}

            # Update weights
            self.model_weights[ticker] = normalized_weights

            # Also update sector weights
            current_sector_weights = self.sector_weights[sector]
            for method in normalized_weights:
                current_weight = current_sector_weights.get(method, GLOBAL_WEIGHTS[method])
                new_sector_weight = (1 - decayed_learning_rate) * current_weight + decayed_learning_rate * normalized_weights[method]
                current_sector_weights[method] = new_sector_weight

            # Normalize sector weights
            total_sector_weight = sum(current_sector_weights.values())
            if total_sector_weight > 0:
                self.sector_weights[sector] = {k: v / total_sector_weight for k, v in current_sector_weights.items()}

    def get_weights(self, ticker: str, sector: str) -> Dict[str, float]:
        """Get weights for valuation methods for this ticker"""
        # Start with global weights
        weights = GLOBAL_WEIGHTS.copy()

        # Apply sector adjustments
        sector_weights = self.sector_weights.get(sector, {})
        for method in weights:
            if method in sector_weights:
                weights[method] = (weights[method] + sector_weights[method]) / 2

        # Apply ticker-specific adjustments
        ticker_weights = self.model_weights.get(ticker, {})
        for method in weights:
            if method in ticker_weights:
                weights[method] = (weights[method] + ticker_weights[method]) / 2

        return weights

    def get_actual_prices(self, ticker: str, date: datetime.date) -> Dict[int, float]:
        """Get actual prices 1/2/3 years after analysis date"""
        actual_prices = {}
        historical_data = self.historical_data.get(ticker, {})

        for years in [1, 2, 3]:
            target_date = date + timedelta(days=365 * years)

            # Find the nearest date with data after target date
            future_dates = [d for d in historical_data.keys() if d >= target_date]
            if future_dates:
                closest_date = min(future_dates)
                actual_prices[years] = historical_data[ticker][closest_date]['close']
            else:
                actual_prices[years] = None

        return actual_prices

    def generate_report(self, ticker: str):
        """Generate comprehensive report for a ticker"""
        if ticker not in self.valuation_results:
            logger.warning(f"No valuation results found for {ticker}")
            return

        # Prepare report data
        fundamentals_summary = self.summarize_fundamentals(ticker)
        valuation_summary = self.summarize_valuations(ticker)
        performance_summary = self.summarize_performance(ticker)
        recommendation = self.generate_recommendation(ticker)
        missing_data_summary = self.summarize_missing_data(ticker)
        weights_summary = self.summarize_weights(ticker)

        # Generate report text
        report = f"""
# Value Investing Analysis Report
## Ticker: {ticker}
### Analysis Period: {START_DATE.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}

## Fundamentals Summary
{fundamentals_summary}

## Valuation Summary
{valuation_summary}

## Performance Summary
{performance_summary}

## Missing Data Summary
{missing_data_summary}

## Model Weights Summary
{weights_summary}

## Recommendation
{recommendation}
"""
        # Save report to file
        report_filename = f"value_analysis_report_{ticker}_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)

        logger.info(f"Report generated and saved to {report_filename}")

    def summarize_fundamentals(self, ticker: str) -> str:
        """Generate fundamentals summary for report"""
        if ticker not in self.valuation_results:
            return "No fundamentals data available."

        # Get most recent analysis
        dates = sorted(self.valuation_results[ticker].keys())
        if not dates:
            return "No fundamentals data available."

        latest_date = dates[-1]
        fundamentals = self.valuation_results[ticker][latest_date]['fundamentals']

        # Create table of fundamentals vs criteria
        lines = ["| Metric | Value | Criteria | Pass/Fail |",
                "|--------|-------|----------|-----------|"]

        for metric, criteria in ANALYSIS_CRITERIA.items():
            value = fundamentals.get(metric, None)
            if value is None:
                lines.append(f"| {criteria['desc']} | Missing | {criteria['threshold']} {criteria['compare']} | N/A |")
                continue

            # Check if value meets criteria
            if criteria['compare'] == '<=':
                passed = value <= criteria['threshold']
            else:
                passed = value >= criteria['threshold']

            lines.append(f"| {criteria['desc']} | {value:.2f} | {criteria['threshold']} {criteria['compare']} | {'PASS' if passed else 'FAIL'} |")

        return "\n".join(lines)

    def summarize_valuations(self, ticker: str) -> str:
        """Generate valuation summary for report"""
        if ticker not in self.valuation_results:
            return "No valuation data available."

        # Get most recent analysis
        dates = sorted(self.valuation_results[ticker].keys())
        if not dates:
            return "No valuation data available."

        latest_date = dates[-1]
        valuation = self.valuation_results[ticker][latest_date]['valuation']
        current_price = self.historical_data[ticker][latest_date]['close']

        lines = ["| Valuation Method | Intrinsic Value | Current Price | Margin of Safety |",
                "|------------------|------------------|---------------|-------------------|"]

        for method, value in valuation.items():
            if value is None:
                lines.append(f"| {method.replace('_', ' ').title()} | N/A | {current_price:.2f} | N/A |")
                continue

            margin = (value - current_price) / value if value != 0 else 0
            lines.append(f"| {method.replace('_', ' ').title()} | {value:.2f} | {current_price:.2f} | {margin:.1%} |")

        return "\n".join(lines)

    def summarize_performance(self, ticker: str) -> str:
        """Generate performance summary for report"""
        if ticker not in self.performance_metrics or not self.performance_metrics[ticker]:
            return "No performance data available yet."

        # Calculate average errors for each method and time horizon
        method_errors = defaultdict(lambda: defaultdict(list))

        for pm in self.performance_metrics[ticker]:
            for method, errors in pm['errors'].items():
                for years, error in errors.items():
                    method_errors[method][years].append(error)

        # Calculate average errors
        avg_errors = {}
        for method, year_errors in method_errors.items():
            avg_errors[method] = {
                years: np.mean(errors) if errors else None
                for years, errors in year_errors.items()
            }

        # Generate table
        lines = ["| Valuation Method | 1-Year Avg Error | 2-Year Avg Error | 3-Year Avg Error |",
                "|------------------|------------------|------------------|------------------|"]

        for method, errors in avg_errors.items():
            line = f"| {method.replace('_', ' ').title()} |"
            for years in [1, 2, 3]:
                error = errors.get(years)
                line += f" {error:.1%} |" if error is not None else " N/A |"
            lines.append(line)

        return "\n".join(lines)

    def generate_recommendation(self, ticker: str) -> str:
        """Generate buy/hold/sell recommendation"""
        if ticker not in self.valuation_results:
            return "Insufficient data for recommendation."

        # Get most recent analysis
        dates = sorted(self.valuation_results[ticker].keys())
        latest_date = dates[-1]

        valuation = self.valuation_results[ticker][latest_date]['valuation']
        current_price = self.historical_data[ticker][latest_date]['close']

        if 'weighted_avg' not in valuation or valuation['weighted_avg'] is None:
            return "Insufficient data for recommendation."

        intrinsic_value = valuation['weighted_avg']
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value

        # Consider recent performance trends
        performance_trend = self.assess_performance_trend(ticker)

        # Generate recommendation based on margin of safety and performance
        if margin_of_safety > 0.3:
            recommendation = "STRONG BUY"
            rationale = f"The stock appears significantly undervalued with a margin of safety of {margin_of_safety:.1%}."
        elif margin_of_safety > 0.15:
            recommendation = "BUY"
            rationale = f"The stock appears undervalued with a margin of safety of {margin_of_safety:.1%}."
        elif margin_of_safety > -0.1:
            recommendation = "HOLD"
            rationale = "The stock appears fairly valued with limited margin of safety."
        else:
            recommendation = "SELL"
            rationale = f"The stock appears overvalued with a negative margin of safety of {margin_of_safety:.1%}."

        # Adjust based on performance trend
        if performance_trend == "improving" and margin_of_safety > 0:
            recommendation += " (with increasing confidence)"
            rationale += " The model's recent predictions for this stock have been improving in accuracy."
        elif performance_trend == "worsening":
            recommendation += " (with caution)"
            rationale += " Caution is advised as the model's recent predictions for this stock have been less accurate."

        return f"**Recommendation:** {recommendation}\n\n**Rationale:** {rationale}"

    def assess_performance_trend(self, ticker: str) -> str:
        """Assess whether model performance is improving or worsening"""
        if ticker not in self.performance_metrics or len(self.performance_metrics[ticker]) < 4:
            return "neutral"

        # Get last 4 quarters of performance
        recent_performance = self.performance_metrics[ticker][-4:]

        # Calculate average error for each period
        avg_errors = []
        for pm in recent_performance:
            errors = [e for method_errors in pm['errors'].values() for e in method_errors.values()]
            avg_errors.append(np.mean(errors) if errors else 0)

        # Calculate trend
        if len(avg_errors) >= 2:
            first_half = np.mean(avg_errors[:2])
            second_half = np.mean(avg_errors[2:])

            if second_half < first_half * 0.9:  # Error decreased by >10%
                return "improving"
            elif second_half > first_half * 1.1:  # Error increased by >10%
                return "worsening"

        return "neutral"

    def summarize_missing_data(self, ticker: str) -> str:
        """Summarize missing data points for the ticker"""
        if ticker not in self.valuation_results:
            return "No data available to assess missing data points."

        # Count missing data points across all analyses
        missing_counts = defaultdict(int)
        total_analyses = 0

        for date, analysis in self.valuation_results[ticker].items():
            total_analyses += 1
            fundamentals = analysis['fundamentals']
            for metric in ANALYSIS_CRITERIA:
                if fundamentals.get(metric) is None:
                    missing_counts[metric] += 1

        if not missing_counts:
            return "All required data points are available."

        # Generate summary
        lines = ["The following data points were missing in some analyses:",
                "| Metric | Missing Count | % of Analyses | Alternative Suggestion |",
                "|--------|--------------|--------------|-----------------------|"]

        for metric, count in missing_counts.items():
            pct_missing = count / total_analyses * 100
            desc = ANALYSIS_CRITERIA[metric]['desc']

            # Suggest alternatives
            if metric == 'pe':
                alternative = "Use sector average P/E or historical P/E"
            elif metric == 'debt_equity':
                alternative = "Estimate from similar companies in sector"
            elif metric == 'fcf_yield':
                alternative = "Calculate from cash flow statement if available"
            else:
                alternative = "No specific alternative suggested"

            lines.append(f"| {desc} | {count} | {pct_missing:.1f}% | {alternative} |")

        return "\n".join(lines)

    def summarize_weights(self, ticker: str) -> str:
        """Summarize model weights for the ticker"""
        if ticker not in self.valuation_results:
            return "No weight data available."

        # Get sector
        latest_date = max(self.valuation_results[ticker].keys())
        sector = self.valuation_results[ticker][latest_date]['fundamentals'].get('sector', 'Unknown')

        # Get weights
        global_weights = GLOBAL_WEIGHTS
        sector_weights = self.sector_weights.get(sector, GLOBAL_WEIGHTS)
        ticker_weights = self.model_weights.get(ticker, GLOBAL_WEIGHTS)

        # Generate table
        lines = ["| Valuation Method | Global Weight | Sector Weight | Ticker Weight |",
                "|------------------|--------------|--------------|--------------|"]

        methods = set(global_weights.keys()).union(sector_weights.keys()).union(ticker_weights.keys())
        for method in sorted(methods):
            gw = global_weights.get(method, 0)
            sw = sector_weights.get(method, 0)
            tw = ticker_weights.get(method, 0)

            lines.append(f"| {method.replace('_', ' ').title()} | {gw:.1%} | {sw:.1%} | {tw:.1%} |")

        explanation = """
**Weight Explanation:**
- **Global Weight:** Baseline weight for all stocks
- **Sector Weight:** Adjusted weight based on sector performance
- **Ticker Weight:** Customized weight based on this stock's historical performance
"""
        return "\n".join(lines) + explanation

    # Helper methods for valuation calculations
    def get_fcf(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Get free cash flow for a ticker at a specific date"""
        try:
            stock = yf.Ticker(ticker)
            cash_flow = stock.cash_flow
            cf_date = cash_flow.columns[cash_flow.columns <= pd.Timestamp(date)].max()
            return cash_flow[cf_date].get('Free Cash Flow', None)
        except Exception:
            return None

    def get_eps(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Get earnings per share for a ticker at a specific date"""
        try:
            stock = yf.Ticker(ticker)
            income_stmt = stock.income_stmt
            is_date = income_stmt.columns[income_stmt.columns <= pd.Timestamp(date)].max()
            net_income = income_stmt[is_date].get('Net Income', None)
            shares = self.get_shares_outstanding(ticker, date)
            if net_income and shares:
                return net_income / shares
            return None
        except Exception:
            return None

    def get_book_value_per_share(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Get book value per share for a ticker at a specific date"""
        try:
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            bs_date = balance_sheet.columns[balance_sheet.columns <= pd.Timestamp(date)].max()
            equity = balance_sheet[bs_date].get('Total Stockholder Equity', None)
            shares = self.get_shares_outstanding(ticker, date)
            if equity and shares:
                return equity / shares
            return None
        except Exception:
            return None

    def get_shares_outstanding(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Get shares outstanding for a ticker at a specific date"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('sharesOutstanding', None)
        except Exception:
            return None

    def estimate_growth_rate(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Estimate growth rate based on historical data"""
        try:
            stock = yf.Ticker(ticker)
            income_stmt = stock.income_stmt

            # Get up to 5 years of historical revenue data
            dates = [col for col in income_stmt.columns if col <= pd.Timestamp(date)]
            dates = sorted(dates, reverse=True)[:5]

            if len(dates) < 2:
                return None

            revenues = [income_stmt[date].get('Total Revenue', 0) for date in dates]
            revenues = [r for r in revenues if r > 0]

            if len(revenues) < 2:
                return None

            # Calculate CAGR
            start_revenue = revenues[-1]
            end_revenue = revenues[0]
            years = (dates[0] - dates[-1]).days / 365.25

            if start_revenue <= 0 or years <= 0:
                return None

            cagr = (end_revenue / start_revenue) ** (1 / years) - 1
            return min(max(cagr, 0), 0.2)  # Cap growth rate at 20%
        except Exception:
            return None

    def estimate_dividend_growth_rate(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Estimate dividend growth rate based on historical data"""
        try:
            stock = yf.Ticker(ticker)
            dividends = stock.dividends

            if dividends.empty:
                return None

            # Get dividends in the 3 years before our analysis date
            start_date = date - timedelta(days=3*365)
            relevant_dividends = dividends[(dividends.index >= pd.Timestamp(start_date)) &
                                         (dividends.index <= pd.Timestamp(date))]

            if len(relevant_dividends) < 2:
                return None

            # Calculate growth rate
            first_div = relevant_dividends.iloc[0]
            last_div = relevant_dividends.iloc[-1]
            years = (relevant_dividends.index[-1] - relevant_dividends.index[0]).days / 365.25

            if first_div <= 0 or years <= 0:
                return None

            growth_rate = (last_div / first_div) ** (1 / years) - 1
            return min(max(growth_rate, 0), 0.15)  # Cap growth rate at 15%
        except Exception:
            return None

    def calculate_discount_rate(self, beta: float) -> float:
        """Calculate discount rate using CAPM"""
        return RISK_FREE_RATE + beta * (MARKET_RETURN - RISK_FREE_RATE)

    def get_sector_average_pe(self, sector: str, date: datetime.date) -> Optional[float]:
        """Get sector average P/E ratio (simplified - in reality would use external data)"""
        # This is a placeholder - in a real implementation you would:
        # 1. Get all stocks in the sector
        # 2. Calculate their P/E ratios
        # 3. Return the average
        return None

    def get_historical_average_pe(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Get historical average P/E ratio for the stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('fiveYearAvgPE', None)
        except Exception:
            return None

    def get_sector_average_pfcf(self, sector: str, date: datetime.date) -> Optional[float]:
        """Get sector average P/FCF ratio (simplified - in reality would use external data)"""
        # This is a placeholder - similar to get_sector_average_pe
        return None

    def get_historical_average_pfcf(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Get historical average P/FCF ratio for the stock"""
        # This would require historical FCF data which isn't easily available from yfinance
        return None

def main():
    """Main execution function"""
    print("=== DeepSeek Value Investing Analysis System ===")
    print("This program performs comprehensive value investing analysis on stocks.")

    # Get user input
    input_path = input("Enter path to CSV file with tickers or a single ticker: ").strip()

    # Initialize analyzer
    analyzer = ValueInvestingAnalyzer()

    try:
        # Load and validate tickers
        analyzer.load_tickers(input_path)

        # Process each ticker
        for ticker in analyzer.tickers:
            try:
                analyzer.analyze_ticker(ticker)
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                continue

        # Save final state
        analyzer.save_state()

        print("\nAnalysis completed successfully!")
        print(f"Reports generated for: {', '.join(analyzer.tickers)}")
    except Exception as e:
        logger.error(f"Fatal error in analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
