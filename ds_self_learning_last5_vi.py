import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import os
import csv
import time
from typing import List, Dict, Union, Optional

# Constants
DEFAULT_DISCOUNT_RATE = 0.10
DEFAULT_TERMINAL_GROWTH = 0.02
DEFAULT_VALUATION_WEIGHTS = {'dcf': 0.5, 'pe': 0.25, 'pfcf': 0.25}
ANALYSIS_PERIODS = [1, 3, 5]  # Years for analysis
DB_NAME = 'stock_analysis.db'

class StockAnalysis:
    def __init__(self):
        """Initialize the StockAnalysis class with database connection"""
        self.conn = sqlite3.connect(DB_NAME)
        self._create_database_tables()
        self.current_date = datetime.now().strftime('%Y-%m-%d')

    def _create_database_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()

        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickers (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            industry TEXT,
            sector TEXT,
            last_updated TEXT
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS financials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            revenue REAL,
            net_income REAL,
            eps REAL,
            total_shares REAL,
            operating_income REAL,
            dividends_per_share REAL,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS balance_sheet (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            total_debt REAL,
            total_equity REAL,
            current_assets REAL,
            current_liabilities REAL,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cashflow (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            free_cash_flow REAL,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            close_price REAL,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            pe_ratio REAL,
            debt_equity REAL,
            fcf_yield REAL,
            fcf_margin REAL,
            roe REAL,
            current_ratio REAL,
            pb_ratio REAL,
            op_margin REAL,
            dividend_yield REAL,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS valuations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            dcf_low REAL,
            dcf_high REAL,
            pe_low REAL,
            pe_high REAL,
            pfcf_low REAL,
            pfcf_high REAL,
            composite_low REAL,
            composite_high REAL,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            parameter TEXT,
            value REAL,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            analysis_date TEXT,
            horizon_years INTEGER,
            actual_price REAL,
            predicted_low REAL,
            predicted_high REAL,
            error REAL,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        ''')

        self.conn.commit()

    def get_user_input(self):
        """Get user input for stock tickers"""
        print("\nStock Analysis and Valuation Program")
        print("-----------------------------------")

        while True:
            choice = input("Enter '1' for a single ticker or '2' for a file with multiple tickers: ").strip()

            if choice == '1':
                ticker = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()
                if self._validate_ticker(ticker):
                    return [ticker]
                else:
                    print(f"Invalid ticker: {ticker}. Please try again.")
            elif choice == '2':
                file_path = input("Enter the path to the file containing tickers: ").strip()
                try:
                    tickers = self._read_tickers_from_file(file_path)
                    valid_tickers = [t for t in tickers if self._validate_ticker(t)]
                    if valid_tickers:
                        print(f"Found {len(valid_tickers)} valid tickers.")
                        return valid_tickers
                    else:
                        print("No valid tickers found in the file. Please try again.")
                except Exception as e:
                    print(f"Error reading file: {e}. Please try again.")
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def _validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker exists using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            # Quick check to see if basic info is available
            if stock.info.get('symbol', '').upper() == ticker.upper():
                return True
            return False
        except:
            return False

    def _read_tickers_from_file(self, file_path: str) -> List[str]:
        """Read tickers from a file (CSV or TXT)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        tickers = []
        with open(file_path, 'r') as f:
            if file_path.endswith('.csv'):
                reader = csv.reader(f)
                for row in reader:
                    tickers.extend([t.strip().upper() for t in row if t.strip()])
            else:  # Assume plain text file
                tickers = [line.strip().upper() for line in f if line.strip()]

        return tickers

    def retrieve_and_store_data(self, tickers: List[str], years: int = 5):
        """Retrieve and store data for each ticker"""
        for ticker in tickers:
            try:
                print(f"\nRetrieving data for {ticker}...")

                # Get stock info
                stock = yf.Ticker(ticker)
                info = stock.info

                # Store basic ticker info
                self._store_ticker_info(
                    ticker,
                    info.get('longName', ''),
                    info.get('industry', ''),
                    info.get('sector', '')
                )

                # Get historical data
                end_date = datetime.now()
                start_date = end_date - pd.DateOffset(years=years)

                # Get financial statements
                self._retrieve_financials(ticker, stock, years)

                # Get historical prices
                self._retrieve_historical_prices(ticker, start_date, end_date)

                print(f"Successfully retrieved and stored data for {ticker}")

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

    def _store_ticker_info(self, ticker: str, name: str, industry: str, sector: str):
        """Store basic ticker information"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO tickers (ticker, name, industry, sector, last_updated)
        VALUES (?, ?, ?, ?, ?)
        ''', (ticker, name, industry, sector, self.current_date))
        self.conn.commit()

    def _retrieve_financials(self, ticker: str, stock: yf.Ticker, years: int):
        """Retrieve and store financial statements"""
        try:
            # Get financial statements
            financials = stock.financials.transpose()
            balance_sheet = stock.balance_sheet.transpose()
            cashflow = stock.cashflow.transpose()

            # Store financials
            for date, row in financials.iterrows():
                self._store_financial_statement(
                    ticker,
                    'financials',
                    date.strftime('%Y-%m-%d'),
                    row.get('Total Revenue'),
                    row.get('Net Income'),
                    row.get('Net Income') / stock.info.get('sharesOutstanding', 1),
                    stock.info.get('sharesOutstanding'),
                    row.get('Operating Income'),
                    row.get('Dividends Paid', 0) / stock.info.get('sharesOutstanding', 1) if 'Dividends Paid' in row else 0
                )

            # Store balance sheet
            for date, row in balance_sheet.iterrows():
                self._store_financial_statement(
                    ticker,
                    'balance_sheet',
                    date.strftime('%Y-%m-%d'),
                    row.get('Total Debt'),
                    row.get('Total Stockholder Equity'),
                    row.get('Total Current Assets'),
                    row.get('Total Current Liabilities')
                )

            # Store cashflow
            for date, row in cashflow.iterrows():
                self._store_financial_statement(
                    ticker,
                    'cashflow',
                    date.strftime('%Y-%m-%d'),
                    row.get('Free Cash Flow')
                )

        except Exception as e:
            print(f"Error retrieving financials for {ticker}: {e}")

    def _store_financial_statement(self, ticker: str, table: str, date: str, *values):
        """Store financial statement data in the database"""  # <-- Indent this line
        cursor = self.conn.cursor()  # <-- Also indented

        if table == 'financials':
            cursor.execute('''
            INSERT OR REPLACE INTO financials (ticker, date, revenue, net_income, eps, total_shares, operating_income, dividends_per_share)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (ticker, date, *values))
        elif table == 'balance_sheet':
            cursor.execute('''
            INSERT OR REPLACE INTO balance_sheet (ticker, date, total_debt, total_equity, current_assets, current_liabilities)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (ticker, date, *values))
        elif table == 'cashflow':
            cursor.execute('''
            INSERT OR REPLACE INTO cashflow (ticker, date, free_cash_flow)
            VALUES (?, ?, ?)
            ''', (ticker, date, *values))

            self.conn.commit()  # <-- Indentation matches the function body

    def _retrieve_historical_prices(self, ticker: str, start_date, end_date):
        """Retrieve and store historical prices"""
        try:
            hist = yf.download(ticker, start=start_date, end=end_date)

            cursor = self.conn.cursor()
            for date, row in hist.iterrows():
                cursor.execute('''
                INSERT OR REPLACE INTO prices (ticker, date, close_price)
                VALUES (?, ?, ?)
                ''', (ticker, date.strftime('%Y-%m-%d'), row['Close']))

            self.conn.commit()
        except Exception as e:
            print(f"Error retrieving historical prices for {ticker}: {e}")

    def calculate_financial_ratios(self, tickers: List[str]):
        """Calculate and store financial ratios for each ticker"""
        for ticker in tickers:
            try:
                print(f"\nCalculating financial ratios for {ticker}...")

                # Get the latest price
                cursor = self.conn.cursor()
                cursor.execute('''
                SELECT date, close_price FROM prices
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
                ''', (ticker,))
                latest_price_data = cursor.fetchone()

                if not latest_price_data:
                    print(f"No price data available for {ticker}")
                    continue

                latest_date, latest_price = latest_price_data

                # Get financial data for ratio calculations
                financial_data = self._get_financial_data_for_ratios(ticker, latest_date)

                if not financial_data:
                    print(f"Insufficient financial data for {ticker}")
                    continue

                # Calculate ratios
                ratios = self._calculate_ratios(ticker, financial_data, latest_price)

                # Store ratios
                self._store_ratios(ticker, latest_date, ratios)

                print(f"Successfully calculated ratios for {ticker}")

            except Exception as e:
                print(f"Error calculating ratios for {ticker}: {e}")
                continue

    def _get_financial_data_for_ratios(self, ticker: str, date: str) -> Optional[Dict]:
        """Retrieve financial data needed for ratio calculations"""
        cursor = self.conn.cursor()

        # Get most recent financial statements before the date
        cursor.execute('''
        SELECT * FROM financials
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT 1
        ''', (ticker, date))
        financials = cursor.fetchone()

        cursor.execute('''
        SELECT * FROM balance_sheet
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT 1
        ''', (ticker, date))
        balance_sheet = cursor.fetchone()

        cursor.execute('''
        SELECT * FROM cashflow
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT 1
        ''', (ticker, date))
        cashflow = cursor.fetchone()

        if not financials or not balance_sheet or not cashflow:
            return None

        # Map the data to a dictionary
        return {
            'revenue': financials[3],
            'net_income': financials[4],
            'eps': financials[5],
            'total_shares': financials[6],
            'operating_income': financials[7],
            'dividends_per_share': financials[8],
            'total_debt': balance_sheet[4],
            'total_equity': balance_sheet[5],
            'current_assets': balance_sheet[6],
            'current_liabilities': balance_sheet[7],
            'free_cash_flow': cashflow[4]
        }

    def _calculate_ratios(self, ticker: str, financial_data: Dict, latest_price: float) -> Dict:
        """Calculate financial ratios"""
        ratios = {}

        try:
            # P/E Ratio
            ratios['pe_ratio'] = latest_price / financial_data['eps'] if financial_data['eps'] and financial_data['eps'] != 0 else None

            # Debt/Equity Ratio
            ratios['debt_equity'] = financial_data['total_debt'] / financial_data['total_equity'] if financial_data['total_equity'] and financial_data['total_equity'] != 0 else None

            # FCF Yield and Margin
            market_cap = latest_price * financial_data['total_shares'] if financial_data['total_shares'] else None
            ratios['fcf_yield'] = financial_data['free_cash_flow'] / market_cap if market_cap and market_cap != 0 else None
            ratios['fcf_margin'] = financial_data['free_cash_flow'] / financial_data['revenue'] if financial_data['revenue'] and financial_data['revenue'] != 0 else None

            # ROE
            ratios['roe'] = financial_data['net_income'] / financial_data['total_equity'] if financial_data['total_equity'] and financial_data['total_equity'] != 0 else None

            # Current Ratio
            ratios['current_ratio'] = financial_data['current_assets'] / financial_data['current_liabilities'] if financial_data['current_liabilities'] and financial_data['current_liabilities'] != 0 else None

            # P/B Ratio
            book_value = financial_data['total_equity'] / financial_data['total_shares'] if financial_data['total_shares'] and financial_data['total_shares'] != 0 else None
            ratios['pb_ratio'] = latest_price / book_value if book_value and book_value != 0 else None

            # Operating Margin
            ratios['op_margin'] = financial_data['operating_income'] / financial_data['revenue'] if financial_data['revenue'] and financial_data['revenue'] != 0 else None

            # Dividend Yield
            ratios['dividend_yield'] = financial_data['dividends_per_share'] / latest_price if financial_data['dividends_per_share'] and latest_price != 0 else None

        except Exception as e:
            print(f"Error calculating ratios for {ticker}: {e}")

        return ratios

    def _store_ratios(self, ticker: str, date: str, ratios: Dict):
        """Store calculated ratios in the database"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO metrics (
            ticker, date, pe_ratio, debt_equity, fcf_yield, fcf_margin,
            roe, current_ratio, pb_ratio, op_margin, dividend_yield
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, date,
            ratios.get('pe_ratio'),
            ratios.get('debt_equity'),
            ratios.get('fcf_yield'),
            ratios.get('fcf_margin'),
            ratios.get('roe'),
            ratios.get('current_ratio'),
            ratios.get('pb_ratio'),
            ratios.get('op_margin'),
            ratios.get('dividend_yield')
        ))
        self.conn.commit()

    def perform_valuation(self, tickers: List[str]):
        """Perform valuation analysis for each ticker"""
        for ticker in tickers:
            try:
                print(f"\nPerforming valuation for {ticker}...")

                # Get the latest price and date
                cursor = self.conn.cursor()
                cursor.execute('''
                SELECT date, close_price FROM prices
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
                ''', (ticker,))
                latest_price_data = cursor.fetchone()

                if not latest_price_data:
                    print(f"No price data available for {ticker}")
                    continue

                latest_date, latest_price = latest_price_data

                # Get financial data for valuation
                financial_data = self._get_financial_data_for_valuation(ticker)

                if not financial_data:
                    print(f"Insufficient financial data for {ticker}")
                    continue

                # Perform valuations
                valuations = self._calculate_valuations(ticker, financial_data, latest_price)

                # Store valuations
                self._store_valuations(ticker, latest_date, valuations)

                # Generate report
                self.generate_valuation_report(ticker, latest_date, latest_price, financial_data, valuations)

                print(f"Successfully performed valuation for {ticker}")

            except Exception as e:
                print(f"Error performing valuation for {ticker}: {e}")
                continue

    def _get_financial_data_for_valuation(self, ticker: str) -> Optional[Dict]:
        """Retrieve financial data needed for valuation"""
        cursor = self.conn.cursor()

        # Get multiple years of financial data
        cursor.execute('''
        SELECT date, revenue, net_income, eps, total_shares, operating_income, dividends_per_share
        FROM financials
        WHERE ticker = ?
        ORDER BY date DESC
        ''', (ticker,))
        financials = cursor.fetchall()

        cursor.execute('''
        SELECT date, total_debt, total_equity, current_assets, current_liabilities
        FROM balance_sheet
        WHERE ticker = ?
        ORDER BY date DESC
        ''', (ticker,))
        balance_sheet = cursor.fetchall()

        cursor.execute('''
        SELECT date, free_cash_flow
        FROM cashflow
        WHERE ticker = ?
        ORDER BY date DESC
        ''', (ticker,))
        cashflow = cursor.fetchall()

        cursor.execute('''
        SELECT date, close_price
        FROM prices
        WHERE ticker = ?
        ORDER BY date DESC
        ''', (ticker,))
        prices = cursor.fetchall()

        cursor.execute('''
        SELECT industry, sector
        FROM tickers
        WHERE ticker = ?
        ''', (ticker,))
        ticker_info = cursor.fetchone()

        if not financials or not balance_sheet or not cashflow or not prices or not ticker_info:
            return None

        # Process the data into a structured format
        processed_data = {
            'financials': [],
            'balance_sheet': [],
            'cashflow': [],
            'prices': [],
            'industry': ticker_info[0],
            'sector': ticker_info[1]
        }

        for row in financials:
            processed_data['financials'].append({
                'date': row[0],
                'revenue': row[1],
                'net_income': row[2],
                'eps': row[3],
                'total_shares': row[4],
                'operating_income': row[5],
                'dividends_per_share': row[6]
            })

        for row in balance_sheet:
            processed_data['balance_sheet'].append({
                'date': row[0],
                'total_debt': row[1],
                'total_equity': row[2],
                'current_assets': row[3],
                'current_liabilities': row[4]
            })

        for row in cashflow:
            processed_data['cashflow'].append({
                'date': row[0],
                'free_cash_flow': row[1]
            })

        for row in prices:
            processed_data['prices'].append({
                'date': row[0],
                'close_price': row[1]
            })

        return processed_data

    def _calculate_valuations(self, ticker: str, financial_data: Dict, latest_price: float) -> Dict:
        """Calculate different valuation metrics"""
        valuations = {}

        try:
            # DCF Valuation
            dcf_valuation = self._dcf_valuation(financial_data)
            valuations['dcf_low'] = dcf_valuation['low']
            valuations['dcf_high'] = dcf_valuation['high']
            valuations['dcf_assumptions'] = {
                'wacc': dcf_valuation['wacc'],
                'growth': dcf_valuation['growth']
            }

            # P/E Relative Valuation
            pe_valuation = self._pe_relative_valuation(financial_data, latest_price)
            valuations['pe_low'] = pe_valuation['low']
            valuations['pe_high'] = pe_valuation['high']

            # P/FCF Valuation
            pfcf_valuation = self._pfcf_valuation(financial_data, latest_price)
            valuations['pfcf_low'] = pfcf_valuation['low']
            valuations['pfcf_high'] = pfcf_valuation['high']

            # Composite Valuation
            composite = self._composite_valuation(valuations)
            valuations['composite_low'] = composite['low']
            valuations['composite_high'] = composite['high']

        except Exception as e:
            print(f"Error calculating valuations for {ticker}: {e}")

        return valuations

    def _dcf_valuation(self, financial_data: Dict) -> Dict:
        """Perform Discounted Cash Flow valuation"""
        # Get FCF history (last 5 years)
        fcf_history = [entry['free_cash_flow'] for entry in financial_data['cashflow'][:5] if entry['free_cash_flow']]

        if not fcf_history or len(fcf_history) < 3:
            return {'low': None, 'high': None, 'wacc': DEFAULT_DISCOUNT_RATE, 'growth': DEFAULT_TERMINAL_GROWTH}

        # Calculate FCF growth rate (CAGR)
        cagr = (fcf_history[0] / fcf_history[-1]) ** (1/len(fcf_history)) - 1

        # Conservative growth assumption (50% of historical growth)
        growth = max(min(cagr * 0.5, 0.15), 0.02)  # Cap at 15%, floor at 2%

        # Use default discount rate (could be enhanced with WACC calculation)
        discount_rate = DEFAULT_DISCOUNT_RATE

        # Project FCF for 5 years
        projected_fcf = []
        current_fcf = fcf_history[0]

        for _ in range(5):
            current_fcf *= (1 + growth)
            projected_fcf.append(current_fcf)

        # Calculate terminal value (Gordon Growth Model)
        terminal_value = current_fcf * (1 + DEFAULT_TERMINAL_GROWTH) / (discount_rate - DEFAULT_TERMINAL_GROWTH)

        # Discount cash flows
        discounted_cashflows = []
        for i, fcf in enumerate(projected_fcf):
            discounted_cashflows.append(fcf / ((1 + discount_rate) ** (i + 1)))

        discounted_terminal = terminal_value / ((1 + discount_rate) ** 5)

        # Total enterprise value
        enterprise_value = sum(discounted_cashflows) + discounted_terminal

        # Get current shares outstanding
        shares_outstanding = financial_data['financials'][0]['total_shares']

        if not shares_outstanding or shares_outstanding <= 0:
            return {'low': None, 'high': None, 'wacc': discount_rate, 'growth': growth}

        # Calculate equity value per share
        intrinsic_value = enterprise_value / shares_outstanding

        # Create range (±10%)
        low_value = intrinsic_value * 0.9
        high_value = intrinsic_value * 1.1

        return {'low': low_value, 'high': high_value, 'wacc': discount_rate, 'growth': growth}

    def _pe_relative_valuation(self, financial_data: Dict, latest_price: float) -> Dict:
        """Perform P/E relative valuation"""
        try:
            # Get historical P/E ratios
            pe_ratios = []
            for entry in financial_data['financials'][:5]:
                if entry['eps'] and entry['eps'] > 0:
                    # Find closest price date to financial statement date
                    price_date = None
                    price = None
                    for price_entry in financial_data['prices']:
                        if price_entry['date'] <= entry['date']:
                            price_date = price_entry['date']
                            price = price_entry['close_price']
                            break

                    if price:
                        pe_ratios.append(price / entry['eps'])

            if not pe_ratios:
                return {'low': None, 'high': None}

            # Calculate median P/E
            median_pe = np.median(pe_ratios)

            # Get current EPS
            current_eps = financial_data['financials'][0]['eps']

            if not current_eps or current_eps <= 0:
                return {'low': None, 'high': None}

            # Calculate intrinsic value range
            intrinsic_value = median_pe * current_eps
            low_value = intrinsic_value * 0.8  # -20%
            high_value = intrinsic_value * 1.2  # +20%

            return {'low': low_value, 'high': high_value}

        except Exception as e:
            print(f"Error in P/E relative valuation: {e}")
            return {'low': None, 'high': None}

    def _pfcf_valuation(self, financial_data: Dict, latest_price: float) -> Dict:
        """Perform P/FCF valuation"""
        try:
            # Get historical FCF yields
            fcf_yields = []
            for entry in financial_data['cashflow'][:5]:
                if entry['free_cash_flow']:
                    # Find closest price date to cashflow date
                    price_date = None
                    price = None
                    for price_entry in financial_data['prices']:
                        if price_entry['date'] <= entry['date']:
                            price_date = price_entry['date']
                            price = price_entry['close_price']
                            break

                    if price and financial_data['financials'][0]['total_shares']:
                        market_cap = price * financial_data['financials'][0]['total_shares']
                        fcf_yields.append(entry['free_cash_flow'] / market_cap)

            if not fcf_yields:
                return {'low': None, 'high': None}

            # Calculate median FCF yield
            median_fcf_yield = np.median(fcf_yields)

            # Get current FCF
            current_fcf = financial_data['cashflow'][0]['free_cash_flow']

            if not current_fcf or financial_data['financials'][0]['total_shares'] <= 0:
                return {'low': None, 'high': None}

            # Calculate intrinsic value range
            if median_fcf_yield != 0:
                intrinsic_value = (current_fcf / financial_data['financials'][0]['total_shares']) / median_fcf_yield
                low_value = intrinsic_value * 0.8  # -20%
                high_value = intrinsic_value * 1.2  # +20%
            else:
                return {'low': None, 'high': None}

            return {'low': low_value, 'high': high_value}

        except Exception as e:
            print(f"Error in P/FCF valuation: {e}")
            return {'low': None, 'high': None}

    def _composite_valuation(self, valuations: Dict) -> Dict:
        """Combine different valuation methods into a composite valuation"""
        valid_methods = []
        weights = []
        values = []

        # Check which valuation methods produced valid results
        if valuations.get('dcf_low') is not None and valuations.get('dcf_high') is not None:
            valid_methods.append('dcf')
            weights.append(DEFAULT_VALUATION_WEIGHTS['dcf'])
            values.append((valuations['dcf_low'], valuations['dcf_high']))

        if valuations.get('pe_low') is not None and valuations.get('pe_high') is not None:
            valid_methods.append('pe')
            weights.append(DEFAULT_VALUATION_WEIGHTS['pe'])
            values.append((valuations['pe_low'], valuations['pe_high']))

        if valuations.get('pfcf_low') is not None and valuations.get('pfcf_high') is not None:
            valid_methods.append('pfcf')
            weights.append(DEFAULT_VALUATION_WEIGHTS['pfcf'])
            values.append((valuations['pfcf_low'], valuations['pfcf_high']))

        if not valid_methods:
            return {'low': None, 'high': None}

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Calculate weighted average
        composite_low = sum(val[0] * weight for val, weight in zip(values, normalized_weights))
        composite_high = sum(val[1] * weight for val, weight in zip(values, normalized_weights))

        return {'low': composite_low, 'high': composite_high}

    def _store_valuations(self, ticker: str, date: str, valuations: Dict):
        """Store valuation results in the database"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO valuations (
            ticker, date, dcf_low, dcf_high, pe_low, pe_high,
            pfcf_low, pfcf_high, composite_low, composite_high
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, date,
            valuations.get('dcf_low'),
            valuations.get('dcf_high'),
            valuations.get('pe_low'),
            valuations.get('pe_high'),
            valuations.get('pfcf_low'),
            valuations.get('pfcf_high'),
            valuations.get('composite_low'),
            valuations.get('composite_high')
        ))
        self.conn.commit()

    def generate_valuation_report(self, ticker: str, date: str, current_price: float,
                                financial_data: Dict, valuations: Dict):
        """Generate a valuation report for the ticker"""
        try:
            # Get the latest metrics
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT * FROM metrics
            WHERE ticker = ? AND date = ?
            ''', (ticker, date))
            metrics = cursor.fetchone()

            if not metrics:
                print(f"No metrics available for {ticker} on {date}")
                return

            # Create report directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'reports/{ticker}_{timestamp}_valuation_report.txt'

            with open(report_filename, 'w') as f:
                # Header
                f.write(f"Stock Valuation Report for {ticker}\n")
                f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

                # Valuation Summary
                f.write("[Valuation Summary]\n")

                if valuations.get('dcf_low') is not None:
                    f.write(f"- DCF Range: ${valuations['dcf_low']:.2f} - ${valuations['dcf_high']:.2f} ")
                    f.write(f"(Key assumptions: WACC={valuations['dcf_assumptions']['wacc']:.2%}, Growth={valuations['dcf_assumptions']['growth']:.2%})\n")
                else:
                    f.write("- DCF Range: Not available (insufficient data)\n")

                if valuations.get('pe_low') is not None:
                    f.write(f"- P/E Relative Range: ${valuations['pe_low']:.2f} - ${valuations['pe_high']:.2f}\n")
                else:
                    f.write("- P/E Relative Range: Not available (insufficient data)\n")

                if valuations.get('pfcf_low') is not None:
                    f.write(f"- P/FCF Range: ${valuations['pfcf_low']:.2f} - ${valuations['pfcf_high']:.2f}\n")
                else:
                    f.write("- P/FCF Range: Not available (insufficient data)\n")

                if valuations.get('composite_low') is not None:
                    f.write(f"- Composite Intrinsic Range: ${valuations['composite_low']:.2f} - ${valuations['composite_high']:.2f}\n\n")
                else:
                    f.write("- Composite Intrinsic Range: Not available (insufficient data)\n\n")

                # Recommendation
                f.write("[Recommendation]\n")
                f.write(f"- Current Price: ${current_price:.2f}\n")

                if valuations.get('composite_low') is not None and valuations.get('composite_high') is not None:
                    # Calculate recommendation
                    strong_buy_threshold = valuations['composite_low'] * 0.9
                    buy_threshold = valuations['composite_low'] * 1.1
                    sell_threshold = valuations['composite_high'] * 1.1

                    if current_price < strong_buy_threshold:
                        action = "Strong Buy"
                    elif current_price < buy_threshold:
                        action = "Buy"
                    elif current_price > sell_threshold:
                        action = "Sell"
                    else:
                        action = "Hold"

                    f.write(f"- Action: {action}\n")
                    f.write("- Price Targets:\n")
                    f.write(f"  - Strong Buy: <${strong_buy_threshold:.2f}\n")
                    f.write(f"  - Buy: <${buy_threshold:.2f}\n")
                    f.write(f"  - Hold: ${valuations['composite_low']:.2f}-${valuations['composite_high']:.2f}\n")
                    f.write(f"  - Sell: >${sell_threshold:.2f}\n")

                    # Suggested allocation (simplified)
                    if action == "Strong Buy":
                        allocation = 0.15  # 15% of portfolio
                    elif action == "Buy":
                        allocation = 0.10  # 10%
                    elif action == "Hold":
                        allocation = 0.05  # 5%
                    else:
                        allocation = 0.0  # 0%

                    f.write(f"- Suggested Allocation: {allocation:.0%}\n\n")
                else:
                    f.write("- Action: Not available (insufficient valuation data)\n\n")

                # Key Risks
                f.write("[Key Risks]\n")

                # Analyze metrics for risks
                risk_count = 1

                # High debt levels
                if metrics[4] and metrics[4] > 1.5:  # Debt/Equity > 1.5
                    f.write(f"- Risk {risk_count}: High debt levels may impact financial stability (Debt/Equity: {metrics[4]:.2f}).\n")
                    risk_count += 1

                # Low current ratio
                if metrics[7] and metrics[7] < 1.0:  # Current Ratio < 1.0
                    f.write(f"- Risk {risk_count}: Potential liquidity issues (Current Ratio: {metrics[7]:.2f}).\n")
                    risk_count += 1

                # Negative FCF
                if metrics[3] and metrics[3] < 0:  # FCF Margin < 0
                    f.write(f"- Risk {risk_count}: Negative free cash flow may indicate operational challenges (FCF Margin: {metrics[3]:.2%}).\n")
                    risk_count += 1

                # Low ROE
                if metrics[6] and metrics[6] < 0.1:  # ROE < 10%
                    f.write(f"- Risk {risk_count}: Low return on equity may indicate inefficient use of capital (ROE: {metrics[6]:.2%}).\n")
                    risk_count += 1

                # Industry cyclicality
                if financial_data['sector'] in ['Consumer Cyclical', 'Energy', 'Basic Materials']:
                    f.write(f"- Risk {risk_count}: Industry cyclicality may affect revenue growth.\n")
                    risk_count += 1

                # Add generic risks if none identified
                if risk_count == 1:
                    f.write("- Risk 1: General market risks may affect performance.\n")
                    f.write("- Risk 2: Competitive pressures could impact profitability.\n")
                    f.write("- Risk 3: Management execution risk exists in all companies.\n")

                f.write("\n")

                # Investment Horizon
                f.write("[Investment Horizon]\n")

                if valuations.get('dcf_assumptions', {}).get('growth', 0) > 0.1:
                    horizon = "Long term (high growth potential)"
                elif valuations.get('dcf_assumptions', {}).get('growth', 0) > 0.05:
                    horizon = "Medium term (moderate growth)"
                else:
                    horizon = "Short term (low growth)"

                f.write(f"- {horizon}\n\n")

                # Company Data
                f.write("[Company Data]\n")
                f.write(f"- Industry: {financial_data['industry']}\n")
                f.write(f"- Sector: {financial_data['sector']}\n")
                f.write("- Key Metrics:\n")
                f.write(f"  - P/E: {metrics[3]:.2f}\n")
                f.write(f"  - FCF Yield: {metrics[5]:.2%}\n")
                f.write(f"  - ROE: {metrics[6]:.2%}\n")
                f.write(f"  - Debt/Equity: {metrics[4]:.2f}\n")
                f.write(f"  - FCF Margin: {metrics[3]:.2%}\n")
                f.write(f"  - P/B: {metrics[9]:.2f}\n")
                f.write(f"  - Current Ratio: {metrics[7]:.2f}\n")
                f.write(f"  - Operating Margin: {metrics[10]:.2%}\n")
                f.write(f"  - Dividend Yield: {metrics[11]:.2%}\n")

            print(f"Valuation report saved to {report_filename}")

        except Exception as e:
            print(f"Error generating valuation report for {ticker}: {e}")

    def perform_multi_year_analysis(self, tickers: List[str], years: int = 5):
        """Perform analysis for each year over the specified period"""
        for ticker in tickers:
            try:
                print(f"\nPerforming multi-year analysis for {ticker}...")

                # Get the earliest date with financial data
                cursor = self.conn.cursor()
                cursor.execute('''
                SELECT MIN(date) FROM financials WHERE ticker = ?
                ''', (ticker,))
                earliest_date = cursor.fetchone()[0]

                if not earliest_date:
                    print(f"No financial data available for {ticker}")
                    continue

                # Calculate analysis dates (yearly)
                analysis_dates = pd.date_range(
                    start=earliest_date,
                    end=datetime.now().strftime('%Y-%m-%d'),
                    freq='Y'
                ).strftime('%Y-%m-%d').tolist()

                if len(analysis_dates) < 2:
                    print(f"Insufficient data history for multi-year analysis of {ticker}")
                    continue

                # Perform analysis for each date
                for analysis_date in analysis_dates[-years:]:  # Limit to specified years
                    try:
                        print(f"Analyzing {ticker} as of {analysis_date}...")

                        # Get price at analysis date
                        cursor.execute('''
                        SELECT close_price FROM prices
                        WHERE ticker = ? AND date <= ?
                        ORDER BY date DESC
                        LIMIT 1
                        ''', (ticker, analysis_date))
                        price_data = cursor.fetchone()

                        if not price_data:
                            print(f"No price data available for {ticker} on {analysis_date}")
                            continue

                        current_price = price_data[0]

                        # Get financial data as of analysis date
                        financial_data = self._get_financial_data_for_valuation_as_of_date(ticker, analysis_date)

                        if not financial_data:
                            print(f"Insufficient financial data for {ticker} on {analysis_date}")
                            continue

                        # Calculate ratios
                        ratios = self._calculate_ratios(ticker, financial_data, current_price)
                        self._store_ratios(ticker, analysis_date, ratios)

                        # Perform valuation
                        valuations = self._calculate_valuations(ticker, financial_data, current_price)
                        self._store_valuations(ticker, analysis_date, valuations)

                        # Generate report
                        self.generate_valuation_report(ticker, analysis_date, current_price, financial_data, valuations)

                        # Store performance metrics for self-learning
                        self._store_performance_metrics(ticker, analysis_date, current_price, valuations)

                    except Exception as e:
                        print(f"Error analyzing {ticker} on {analysis_date}: {e}")
                        continue

                print(f"Completed multi-year analysis for {ticker}")

            except Exception as e:
                print(f"Error performing multi-year analysis for {ticker}: {e}")
                continue

    def _get_financial_data_for_valuation_as_of_date(self, ticker: str, date: str) -> Optional[Dict]:
        """Retrieve financial data as of a specific date for valuation"""
        cursor = self.conn.cursor()

        # Get financial statements as of the date
        cursor.execute('''
        SELECT date, revenue, net_income, eps, total_shares, operating_income, dividends_per_share
        FROM financials
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        ''', (ticker, date))
        financials = cursor.fetchall()

        cursor.execute('''
        SELECT date, total_debt, total_equity, current_assets, current_liabilities
        FROM balance_sheet
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        ''', (ticker, date))
        balance_sheet = cursor.fetchall()

        cursor.execute('''
        SELECT date, free_cash_flow
        FROM cashflow
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        ''', (ticker, date))
        cashflow = cursor.fetchall()

        cursor.execute('''
        SELECT date, close_price
        FROM prices
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        ''', (ticker, date))
        prices = cursor.fetchall()

        cursor.execute('''
        SELECT industry, sector
        FROM tickers
        WHERE ticker = ?
        ''', (ticker,))
        ticker_info = cursor.fetchone()

        if not financials or not balance_sheet or not cashflow or not prices or not ticker_info:
            return None

        # Process the data into a structured format
        processed_data = {
            'financials': [],
            'balance_sheet': [],
            'cashflow': [],
            'prices': [],
            'industry': ticker_info[0],
            'sector': ticker_info[1]
        }

        for row in financials:
            processed_data['financials'].append({
                'date': row[0],
                'revenue': row[1],
                'net_income': row[2],
                'eps': row[3],
                'total_shares': row[4],
                'operating_income': row[5],
                'dividends_per_share': row[6]
            })

        for row in balance_sheet:
            processed_data['balance_sheet'].append({
                'date': row[0],
                'total_debt': row[1],
                'total_equity': row[2],
                'current_assets': row[3],
                'current_liabilities': row[4]
            })

        for row in cashflow:
            processed_data['cashflow'].append({
                'date': row[0],
                'free_cash_flow': row[1]
            })

        for row in prices:
            processed_data['prices'].append({
                'date': row[0],
                'close_price': row[1]
            })

        return processed_data

    def _store_performance_metrics(self, ticker: str, analysis_date: str,
                                 current_price: float, valuations: Dict):
        """Store performance metrics for self-learning"""
        cursor = self.conn.cursor()

        # For each analysis horizon (1, 3, 5 years)
        for horizon in ANALYSIS_PERIODS:
            try:
                # Calculate future date
                future_date = (datetime.strptime(analysis_date, '%Y-%m-%d') +
                             pd.DateOffset(years=horizon)).strftime('%Y-%m-%d')

                # Get actual price at future date
                cursor.execute('''
                SELECT close_price FROM prices
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
                ''', (ticker, future_date))
                future_price_data = cursor.fetchone()

                if not future_price_data:
                    continue  # No future price data yet

                future_price = future_price_data[0]

                # Calculate error if we have a composite valuation
                if valuations.get('composite_low') and valuations.get('composite_high'):
                    # Simple error calculation (distance from midpoint)
                    midpoint = (valuations['composite_low'] + valuations['composite_high']) / 2
                    error = abs((future_price - midpoint) / midpoint) if midpoint != 0 else None
                else:
                    error = None

                # Store performance metrics
                cursor.execute('''
                INSERT INTO performance_metrics (
                    ticker, analysis_date, horizon_years,
                    actual_price, predicted_low, predicted_high, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker, analysis_date, horizon,
                    future_price,
                    valuations.get('composite_low'),
                    valuations.get('composite_high'),
                    error
                ))

                self.conn.commit()

            except Exception as e:
                print(f"Error storing performance metrics for {ticker} horizon {horizon}: {e}")
                continue

    def self_learning_analysis(self):
        """Perform self-learning analysis to adjust valuation parameters"""
        try:
            print("\nPerforming self-learning analysis...")

            cursor = self.conn.cursor()

            # Get all tickers with performance metrics
            cursor.execute('''
            SELECT DISTINCT ticker FROM performance_metrics
            ''')
            tickers = [row[0] for row in cursor.fetchall()]

            if not tickers:
                print("No performance metrics available for self-learning")
                return

            # Analyze each ticker
            for ticker in tickers:
                try:
                    print(f"Analyzing {ticker} for self-learning...")

                    # Get all performance metrics for this ticker
                    cursor.execute('''
                    SELECT horizon_years, AVG(error)
                    FROM performance_metrics
                    WHERE ticker = ? AND error IS NOT NULL
                    GROUP BY horizon_years
                    ''', (ticker,))
                    error_metrics = cursor.fetchall()

                    if not error_metrics:
                        print(f"No error metrics available for {ticker}")
                        continue

                    # Calculate average error by horizon
                    avg_errors = {horizon: error for horizon, error in error_metrics}

                    # Simple adjustment: if error is high, adjust growth rate assumption
                    # (This is a simplified approach - could be enhanced with more sophisticated methods)
                    avg_error = np.mean([e for h, e in avg_errors.items()])

                    if avg_error > 0.2:  # If average error > 20%
                        # Get current parameters
                        cursor.execute('''
                        SELECT parameter, value FROM learning_params
                        WHERE ticker = ?
                        ORDER BY date DESC
                        LIMIT 1
                        ''', (ticker,))
                        current_params = {row[0]: row[1] for row in cursor.fetchall()}

                        # Adjust growth rate assumption
                        current_growth = current_params.get('growth_rate', 0.05)
                        new_growth = current_growth * (1 - 0.1)  # Reduce by 10%

                        # Store adjusted parameter
                        cursor.execute('''
                        INSERT INTO learning_params (ticker, date, parameter, value)
                        VALUES (?, ?, ?, ?)
                        ''', (ticker, self.current_date, 'growth_rate', new_growth))

                        self.conn.commit()
                        print(f"Adjusted growth rate for {ticker} from {current_growth:.2%} to {new_growth:.2%}")

                except Exception as e:
                    print(f"Error performing self-learning for {ticker}: {e}")
                    continue

            print("Completed self-learning analysis")

        except Exception as e:
            print(f"Error in self-learning analysis: {e}")

    def generate_summary_report(self):
        """Generate a final summary report"""
        try:
            print("\nGenerating summary report...")

            # Create report directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'reports/summary_report_{timestamp}.txt'

            with open(report_filename, 'w') as f:
                # Header
                f.write("Stock Analysis and Valuation Summary Report\n")
                f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

                cursor = self.conn.cursor()

                # Performance Metrics Summary
                f.write("[Performance Metrics Summary]\n")

                # Get all tickers with valuations
                cursor.execute('''
                SELECT DISTINCT ticker FROM valuations
                ''')
                tickers = [row[0] for row in cursor.fetchall()]

                if not tickers:
                    f.write("No valuation data available for performance analysis.\n\n")
                else:
                    f.write(f"Analyzed {len(tickers)} tickers.\n\n")

                    # Calculate average error by horizon
                    cursor.execute('''
                    SELECT horizon_years, AVG(error)
                    FROM performance_metrics
                    WHERE error IS NOT NULL
                    GROUP BY horizon_years
                    ''')
                    error_metrics = cursor.fetchall()

                    if error_metrics:
                        f.write("Average Prediction Errors:\n")
                        for horizon, error in error_metrics:
                            f.write(f"- {horizon}-year horizon: {error:.2%}\n")
                        f.write("\n")
                    else:
                        f.write("No performance metrics available for error analysis.\n\n")

                # Valuation Insights
                f.write("[Valuation Insights]\n")

                # Get latest valuations for all tickers
                cursor.execute('''
                SELECT v.ticker, t.name, v.composite_low, v.composite_high, p.close_price
                FROM valuations v
                JOIN tickers t ON v.ticker = t.ticker
                JOIN (
                    SELECT ticker, MAX(date) as latest_date
                    FROM valuations
                    GROUP BY ticker
                ) latest ON v.ticker = latest.ticker AND v.date = latest.latest_date
                JOIN prices p ON v.ticker = p.ticker
                WHERE p.date = (
                    SELECT MAX(date) FROM prices WHERE ticker = v.ticker
                )
                ''')

                valuations = cursor.fetchall()

                if valuations:
                    # Calculate undervalued/overvalued
                    undervalued = []
                    overvalued = []

                    for ticker, name, comp_low, comp_high, price in valuations:
                        if comp_low is None or comp_high is None:
                            continue

                        if price < comp_low * 0.9:
                            undervalued.append((ticker, name, price, comp_low, comp_high))
                        elif price > comp_high * 1.1:
                            overvalued.append((ticker, name, price, comp_low, comp_high))

                    if undervalued:
                        f.write("Potentially Undervalued Stocks:\n")
                        for ticker, name, price, low, high in undervalued:
                            discount = (low - price) / price
                            f.write(f"- {ticker} ({name}): Current ${price:.2f} vs. Intrinsic ${low:.2f}-${high:.2f} ({discount:.2%} below low)\n")
                        f.write("\n")

                    if overvalued:
                        f.write("Potentially Overvalued Stocks:\n")
                        for ticker, name, price, low, high in overvalued:
                            premium = (price - high) / high
                            f.write(f"- {ticker} ({name}): Current ${price:.2f} vs. Intrinsic ${low:.2f}-${high:.2f} ({premium:.2%} above high)\n")
                        f.write("\n")

                    if not undervalued and not overvalued:
                        f.write("No strongly undervalued or overvalued stocks identified based on current valuations.\n\n")
                else:
                    f.write("No valuation data available for insights.\n\n")

                # Self-Learning Outcomes
                f.write("[Self-Learning Outcomes]\n")

                cursor.execute('''
                SELECT parameter, AVG(value)
                FROM learning_params
                GROUP BY parameter
                ''')
                learned_params = cursor.fetchall()

                if learned_params:
                    f.write("Average Learned Parameters:\n")
                    for param, value in learned_params:
                        f.write(f"- {param}: {value:.4f}\n")
                    f.write("\n")

                    f.write("Recommendations for Model Improvement:\n")
                    f.write("- Consider adjusting growth rate assumptions based on historical errors\n")
                    f.write("- Review discount rate assumptions for different sectors\n")
                    f.write("- Incorporate macroeconomic factors that may affect valuation accuracy\n")
                    f.write("\n")
                else:
                    f.write("No self-learning parameter adjustments have been made yet.\n\n")

                # Final Recommendations
                f.write("[Final Recommendations]\n")

                if valuations:
                    # Top 3 undervalued stocks
                    cursor.execute('''
                    SELECT v.ticker, t.name, p.close_price, v.composite_low, v.composite_high,
                           (v.composite_low - p.close_price)/p.close_price as discount
                    FROM valuations v
                    JOIN tickers t ON v.ticker = t.ticker
                    JOIN (
                        SELECT ticker, MAX(date) as latest_date
                        FROM valuations
                        GROUP BY ticker
                    ) latest ON v.ticker = latest.ticker AND v.date = latest.latest_date
                    JOIN prices p ON v.ticker = p.ticker
                    WHERE p.date = (
                        SELECT MAX(date) FROM prices WHERE ticker = v.ticker
                    )
                    AND v.composite_low IS NOT NULL AND v.composite_high IS NOT NULL
                    ORDER BY discount DESC
                    LIMIT 3
                    ''')

                    top_undervalued = cursor.fetchall()

                    if top_undervalued:
                        f.write("Top Undervalued Stocks to Consider:\n")
                        for ticker, name, price, low, high, discount in top_undervalued:
                            f.write(f"- {ticker} ({name}): Current ${price:.2f}, Intrinsic ${low:.2f}-${high:.2f} ({discount:.2%} below low)\n")
                        f.write("\n")

                    # Top overvalued stocks
                    cursor.execute('''
                    SELECT v.ticker, t.name, p.close_price, v.composite_low, v.composite_high,
                           (p.close_price - v.composite_high)/v.composite_high as premium
                    FROM valuations v
                    JOIN tickers t ON v.ticker = t.ticker
                    JOIN (
                        SELECT ticker, MAX(date) as latest_date
                        FROM valuations
                        GROUP BY ticker
                    ) latest ON v.ticker = latest.ticker AND v.date = latest.latest_date
                    JOIN prices p ON v.ticker = p.ticker
                    WHERE p.date = (
                        SELECT MAX(date) FROM prices WHERE ticker = v.ticker
                    )
                    AND v.composite_low IS NOT NULL AND v.composite_high IS NOT NULL
                    ORDER BY premium DESC
                    LIMIT 3
                    ''')

                    top_overvalued = cursor.fetchall()

                    if top_overvalued:
                        f.write("Most Overvalued Stocks to Be Cautious About:\n")
                        for ticker, name, price, low, high, premium in top_overvalued:
                            f.write(f"- {ticker} ({name}): Current ${price:.2f}, Intrinsic ${low:.2f}-${high:.2f} ({premium:.2%} above high)\n")
                        f.write("\n")

                f.write("General Investment Advice:\n")
                f.write("- Diversify across sectors and industries\n")
                f.write("- Consider both valuation metrics and qualitative factors\n")
                f.write("- Regularly review and rebalance your portfolio\n")
                f.write("- Consult with a financial advisor for personalized advice\n")

            print(f"Summary report saved to {report_filename}")

        except Exception as e:
            print(f"Error generating summary report: {e}")

    def run(self):
        """Main method to run the stock analysis program"""
        try:
            # Get user input
            tickers = self.get_user_input()

            if not tickers:
                print("No valid tickers to analyze. Exiting.")
                return

            # Get analysis period from user
            while True:
                try:
                    years = int(input("Enter number of years for analysis (5 or 10): ").strip())
                    if years in [5, 10]:
                        break
                    else:
                        print("Please enter either 5 or 10.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Retrieve and store data
            self.retrieve_and_store_data(tickers, years)

            # Calculate financial ratios
            self.calculate_financial_ratios(tickers)

            # Perform valuation
            self.perform_valuation(tickers)

            # Perform multi-year analysis
            self.perform_multi_year_analysis(tickers, years)

            # Perform self-learning analysis
            self.self_learning_analysis()

            # Generate final summary report
            self.generate_summary_report()

            print("\nAnalysis completed successfully!")

        except Exception as e:
            print(f"Error running analysis: {e}")
        finally:
            self.conn.close()


if __name__ == "__main__":
    analyzer = StockAnalysis()
    analyzer.run()
