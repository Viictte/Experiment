"""Finance tool using yfinance with Alpha Vantage fallback and web extraction"""

from typing import Dict, Any, List, Optional
import pandas as pd
import os
import requests
import re
import json as json_lib
from datetime import datetime, timedelta
from rag_system.core.config import get_config

class FinanceTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.finance.enabled', True)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    def get_stock_price(self, ticker: str, period: str = '1mo', use_intraday: bool = False) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Finance tool is disabled'}
        
        if use_intraday and self.alpha_vantage_key:
            result = self._get_global_quote(ticker)
            if 'error' not in result:
                return result
            
            result = self._get_stock_price_intraday(ticker)
            if 'error' not in result:
                return result
        
        if self.alpha_vantage_key:
            result = self._get_global_quote(ticker)
            if 'error' not in result:
                return result
        
        try:
            import yfinance as yf
            import warnings
            warnings.filterwarnings("ignore")
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                if self.alpha_vantage_key:
                    return self._get_stock_price_alpha_vantage(ticker)
                return {'error': f'No data found for ticker: {ticker}'}
            
            return {
                'ticker': ticker,
                'period': period,
                'data': hist.to_dict('records'),
                'latest_price': float(hist['Close'].iloc[-1]),
                'change': float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]),
                'change_percent': float((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)
            }
        except Exception:
            if self.alpha_vantage_key:
                return self._get_stock_price_alpha_vantage(ticker)
            return {'error': f'Unable to fetch data for {ticker}'}
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Finance tool is disabled'}
        
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def compare_stocks(self, tickers: List[str], period: str = '1mo') -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Finance tool is disabled'}
        
        try:
            results = {}
            for ticker in tickers:
                results[ticker] = self.get_stock_price(ticker, period)
            
            return {
                'comparison': results,
                'period': period
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_global_quote(self, ticker: str) -> Dict[str, Any]:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                return {'error': f'Invalid ticker: {ticker}'}
            
            if 'Note' in data:
                return {'error': 'Alpha Vantage API rate limit reached'}
            
            quote = data.get('Global Quote', {})
            if not quote:
                return {'error': f'No quote data for ticker: {ticker}'}
            
            price = float(quote.get('05. price', 0))
            change = float(quote.get('09. change', 0))
            change_percent = float(quote.get('10. change percent', '0').replace('%', ''))
            volume = int(quote.get('06. volume', 0))
            latest_trading_day = quote.get('07. latest trading day', '')
            
            if price == 0:
                return {'error': f'No price data for ticker: {ticker}'}
            
            return {
                'ticker': ticker,
                'period': 'real-time',
                'source': 'Alpha Vantage Global Quote',
                'timestamp': latest_trading_day,
                'latest_price': price,
                'change': change,
                'change_percent': change_percent,
                'volume': volume,
                'data': [{
                    'date': latest_trading_day,
                    'close': price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': volume
                }]
            }
        except Exception as e:
            return {'error': f'Global quote failed: {str(e)}'}
    
    def _get_stock_price_intraday(self, ticker: str) -> Dict[str, Any]:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': ticker,
                'interval': '5min',
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                return {'error': f'Invalid ticker: {ticker}'}
            
            if 'Note' in data:
                return {'error': 'Alpha Vantage API rate limit reached'}
            
            time_series = data.get('Time Series (5min)', {})
            if not time_series:
                return self._get_stock_price_alpha_vantage(ticker)
            
            timestamps = sorted(time_series.keys(), reverse=True)
            if not timestamps:
                return self._get_stock_price_alpha_vantage(ticker)
            
            latest_timestamp = timestamps[0]
            latest_price = float(time_series[latest_timestamp]['4. close'])
            
            day_start_price = latest_price
            for ts in reversed(timestamps[-20:]):
                day_start_price = float(time_series[ts]['1. open'])
                break
            
            return {
                'ticker': ticker,
                'period': 'intraday',
                'source': 'Alpha Vantage Intraday',
                'timestamp': latest_timestamp,
                'data': [
                    {
                        'timestamp': ts,
                        'close': float(time_series[ts]['4. close']),
                        'volume': int(time_series[ts]['5. volume'])
                    }
                    for ts in timestamps[:20]
                ],
                'latest_price': latest_price,
                'change': latest_price - day_start_price,
                'change_percent': ((latest_price - day_start_price) / day_start_price * 100) if day_start_price > 0 else 0
            }
        except Exception as e:
            return self._get_stock_price_alpha_vantage(ticker)
    
    def _get_stock_price_alpha_vantage(self, ticker: str) -> Dict[str, Any]:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                return {'error': f'Invalid ticker: {ticker}'}
            
            if 'Note' in data:
                return {'error': 'Alpha Vantage API rate limit reached'}
            
            time_series = data.get('Time Series (Daily)', {})
            if not time_series:
                return {'error': f'No data found for ticker: {ticker}'}
            
            dates = sorted(time_series.keys(), reverse=True)[:30]
            latest_date = dates[0]
            oldest_date = dates[-1]
            
            latest_price = float(time_series[latest_date]['4. close'])
            oldest_price = float(time_series[oldest_date]['4. close'])
            
            return {
                'ticker': ticker,
                'period': '1mo',
                'source': 'Alpha Vantage',
                'data': [
                    {
                        'date': date,
                        'close': float(time_series[date]['4. close']),
                        'volume': int(time_series[date]['5. volume'])
                    }
                    for date in dates
                ],
                'latest_price': latest_price,
                'change': latest_price - oldest_price,
                'change_percent': ((latest_price - oldest_price) / oldest_price * 100)
            }
        except Exception as e:
            return {'error': f'Alpha Vantage fallback failed: {str(e)}'}
    
    def extract_price_from_web(self, ticker: str, html_content: str, url: str) -> Optional[Dict[str, Any]]:
        try:
            if 'yahoo' in url.lower() or 'finance.yahoo' in url.lower():
                return self._extract_from_yahoo(ticker, html_content)
            elif 'cnbc' in url.lower():
                return self._extract_from_cnbc(ticker, html_content)
            elif 'marketwatch' in url.lower():
                return self._extract_from_marketwatch(ticker, html_content)
            return None
        except Exception:
            return None
    
    def _extract_from_yahoo(self, ticker: str, html_content: str) -> Optional[Dict[str, Any]]:
        try:
            regular_price_match = re.search(r'"regularMarketPrice":\s*\{[^}]*"raw":\s*([0-9.]+)', html_content)
            regular_change_match = re.search(r'"regularMarketChange":\s*\{[^}]*"raw":\s*([0-9.-]+)', html_content)
            regular_change_pct_match = re.search(r'"regularMarketChangePercent":\s*\{[^}]*"raw":\s*([0-9.-]+)', html_content)
            regular_time_match = re.search(r'"regularMarketTime":\s*([0-9]+)', html_content)
            
            pre_price_match = re.search(r'"preMarketPrice":\s*\{[^}]*"raw":\s*([0-9.]+)', html_content)
            pre_change_match = re.search(r'"preMarketChange":\s*\{[^}]*"raw":\s*([0-9.-]+)', html_content)
            pre_change_pct_match = re.search(r'"preMarketChangePercent":\s*\{[^}]*"raw":\s*([0-9.-]+)', html_content)
            pre_time_match = re.search(r'"preMarketTime":\s*([0-9]+)', html_content)
            
            post_price_match = re.search(r'"postMarketPrice":\s*\{[^}]*"raw":\s*([0-9.]+)', html_content)
            post_change_match = re.search(r'"postMarketChange":\s*\{[^}]*"raw":\s*([0-9.-]+)', html_content)
            post_change_pct_match = re.search(r'"postMarketChangePercent":\s*\{[^}]*"raw":\s*([0-9.-]+)', html_content)
            post_time_match = re.search(r'"postMarketTime":\s*([0-9]+)', html_content)
            
            result = {
                'ticker': ticker,
                'source': 'Yahoo Finance (web extraction)',
                'data': []
            }
            
            if post_price_match and post_time_match:
                post_price = float(post_price_match.group(1))
                post_change = float(post_change_match.group(1)) if post_change_match else 0
                post_change_pct = float(post_change_pct_match.group(1)) if post_change_pct_match else 0
                post_timestamp = int(post_time_match.group(1))
                
                result['latest_price'] = post_price
                result['change'] = post_change
                result['change_percent'] = post_change_pct
                result['timestamp'] = datetime.fromtimestamp(post_timestamp).strftime('%Y-%m-%d %H:%M:%S ET')
                result['market_state'] = 'post-market'
                result['data'].append({
                    'price': post_price,
                    'change': post_change,
                    'change_percent': post_change_pct,
                    'market_state': 'post-market'
                })
                return result
            
            if pre_price_match and pre_time_match:
                pre_price = float(pre_price_match.group(1))
                pre_change = float(pre_change_match.group(1)) if pre_change_match else 0
                pre_change_pct = float(pre_change_pct_match.group(1)) if pre_change_pct_match else 0
                pre_timestamp = int(pre_time_match.group(1))
                
                result['latest_price'] = pre_price
                result['change'] = pre_change
                result['change_percent'] = pre_change_pct
                result['timestamp'] = datetime.fromtimestamp(pre_timestamp).strftime('%Y-%m-%d %H:%M:%S ET')
                result['market_state'] = 'pre-market'
                result['data'].append({
                    'price': pre_price,
                    'change': pre_change,
                    'change_percent': pre_change_pct,
                    'market_state': 'pre-market'
                })
                return result
            
            if regular_price_match:
                price = float(regular_price_match.group(1))
                change = float(regular_change_match.group(1)) if regular_change_match else 0
                change_pct = float(regular_change_pct_match.group(1)) if regular_change_pct_match else 0
                timestamp = int(regular_time_match.group(1)) if regular_time_match else int(datetime.now().timestamp())
                
                result['latest_price'] = price
                result['change'] = change
                result['change_percent'] = change_pct
                result['timestamp'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S ET')
                result['market_state'] = 'regular'
                result['data'].append({
                    'price': price,
                    'change': change,
                    'change_percent': change_pct,
                    'market_state': 'regular'
                })
                return result
            
            return None
        except Exception:
            return None
    
    def _extract_from_cnbc(self, ticker: str, html_content: str) -> Optional[Dict[str, Any]]:
        try:
            price_match = re.search(r'class="QuoteStrip-lastPrice">([0-9,.]+)', html_content)
            if not price_match:
                price_match = re.search(r'"last":"([0-9.]+)"', html_content)
            
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                price = float(price_str)
                
                return {
                    'ticker': ticker,
                    'source': 'CNBC (web extraction)',
                    'latest_price': price,
                    'timestamp': datetime.now().isoformat(),
                    'data': [{'price': price}]
                }
            return None
        except Exception:
            return None
    
    def _extract_from_marketwatch(self, ticker: str, html_content: str) -> Optional[Dict[str, Any]]:
        try:
            price_match = re.search(r'class="intraday__price[^>]*>.*?\$([0-9,.]+)', html_content)
            if not price_match:
                price_match = re.search(r'"price":\s*"?([0-9.]+)"?', html_content)
            
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                price = float(price_str)
                
                return {
                    'ticker': ticker,
                    'source': 'MarketWatch (web extraction)',
                    'latest_price': price,
                    'timestamp': datetime.now().isoformat(),
                    'data': [{'price': price}]
                }
            return None
        except Exception:
            return None

_finance_tool = None

def get_finance_tool() -> FinanceTool:
    global _finance_tool
    if _finance_tool is None:
        _finance_tool = FinanceTool()
    return _finance_tool
