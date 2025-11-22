"""Simple question detector for fast-path routing"""

import re
from typing import Optional

class SimpleQuestionDetector:
    """Detects simple questions that can be answered directly by LLM without retrieval"""
    
    def __init__(self):
        self.trivia_patterns = [
            r'\bwho wrote\b', r'\bwho invented\b', r'\bwho is\b', r'\bwho was\b',
            r'\bcapital of\b', r'\bformula for\b', r'\bchemical formula\b',
            r'\bknown as the\b', r'\bplanet\b.*\bred planet\b',
            r'\bhow many days in\b', r'\bhow many\b.*\bleap year\b',
            r'\bwhat are the.*colors\b', r'\bwhat is the.*formula\b',
            r'\bhow do you say\b.*\bin (cantonese|mandarin|chinese)\b',
            r'\bwhat does.*mean in\b', r'\bwhat is.*in (cantonese|mandarin)\b',
            r'\bemergency (phone )?number\b', r'\bphone number for\b',
            r'\bhow many (hours|days|weeks|months|years)\b',
            r'\bwhat year (was|did)\b', r'\bwhen was.*built\b', r'\bwhen was.*opened\b',
            
            r'誰寫', r'誰發明', r'誰是', r'什麼是',
            r'首都', r'公式', r'化學式',
            r'被稱為', r'行星', r'紅色行星',
            r'多少天', r'閏年', r'幾天', r'幾個小時',
            r'怎麼說', r'廣東話', r'粵語', r'什麼意思',
            r'電話號碼', r'緊急電話', r'報警電話',
            r'幾個', r'多少個', r'哪幾部', r'哪幾個',
            r'什麼時候.*建成', r'什麼時候.*開通',
        ]
        
        self.exclusion_patterns = [
            r'\b(stock|price|share|market|trading)\b',
            r'\b(now|today|tonight|tomorrow|yesterday|this (week|month|year))\b',
            r'\b(current|latest|recent|upcoming)\b',
            r'\b(weather|forecast|temperature|rain|snow|wind|humidity)\b',
            r'\b(UV|AQI|air quality|pollution)\b',
            r'\b(sunrise|sunset|typhoon|hurricane|storm)\b',
            r'\b(public holiday|holiday.*this year)\b',
            r'\b(route|driving|distance|directions|travel time)\b',
            r'\bwill it (rain|snow)\b', r'\bis it (raining|snowing)\b',
            r'\bcompare.*stock\b', r'\bperformance.*stock\b',
            
            r'股票|股價|市場|交易',
            r'現在|今天|今晚|明天|昨天|今年|本年',
            r'當前|最新|即時|實時|目前',
            r'天氣|氣溫|下雨|颳風|濕度|預報',
            r'紫外線|空氣質量|污染',
            r'日出|日落|颱風|熱帶氣旋|警告信號',
            r'公眾假期|假期.*今年',
            r'路線|駕駛|距離|行車時間',
            r'會.*下雨', r'會.*下雪',
            r'比較.*股票', r'表現.*股票',
        ]
        
        self.trivia_regex = [re.compile(p, re.IGNORECASE) for p in self.trivia_patterns]
        self.exclusion_regex = [re.compile(p, re.IGNORECASE) for p in self.exclusion_patterns]
    
    def is_simple(self, query: str) -> bool:
        """
        Check if a query is simple enough to answer directly without retrieval.
        
        Returns True for:
        - Arithmetic expressions
        - General trivia (who/what/when/where with no real-time data)
        - Simple translations
        
        Returns False for:
        - Real-time data queries (stock prices, weather, news)
        - Location-specific current information
        - Queries requiring external APIs or web search
        """
        query_lower = query.lower().strip()
        
        for pattern in self.exclusion_regex:
            if pattern.search(query):
                return False
        
        if self._is_arithmetic(query):
            return True
        
        for pattern in self.trivia_regex:
            if pattern.search(query):
                return True
        
        if re.search(r'\bwhat is\b', query_lower) or re.search(r'什麼是', query):
            if not any(term in query_lower for term in ['price', 'cost', 'weather', 'temperature', 'forecast']):
                return True
        
        return False
    
    def _is_arithmetic(self, query: str) -> bool:
        """Check if query is a simple arithmetic expression"""
        clean = query.strip()
        
        if any(op in clean for op in ['加', '減', '乘', '除', '等於']):
            if re.search(r'\d', clean):  # Has at least one digit
                return True
        
        if re.search(r'\d+\s*(multiplied by|times|divided by|plus|minus|subtract|add)\s*\d+', clean, re.IGNORECASE):
            return True
        
        if re.search(r'\d', clean) and re.search(r'[+\-*/^%×÷]', clean):
            cleaned = re.sub(r'[\d\s+\-*/^%×÷().,]', '', clean)
            if len(cleaned) < 5:
                return True
        
        return False
    
    def detect_language(self, query: str) -> str:
        """Detect if query is in English or Chinese"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
        total_chars = len(re.sub(r'\s', '', query))
        
        if total_chars == 0:
            return 'en'
        
        if chinese_chars / total_chars > 0.3:
            return 'zh'
        
        return 'en'

_simple_detector = None

def get_simple_detector() -> SimpleQuestionDetector:
    global _simple_detector
    if _simple_detector is None:
        _simple_detector = SimpleQuestionDetector()
    return _simple_detector
