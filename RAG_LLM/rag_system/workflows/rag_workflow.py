"""RAG Workflow orchestration using LangGraph"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import asyncio
import requests
from rag_system.core.config import get_config
from rag_system.workflows.llm_router import get_llm_router
from rag_system.workflows.simple_detector import get_simple_detector
from rag_system.workflows.attachment_handler import get_attachment_handler
from rag_system.services.hybrid_retrieval import get_hybrid_retrieval_service
from rag_system.tools.weather import get_weather_tool
from rag_system.tools.finance import get_finance_tool
from rag_system.tools.transport import get_transport_tool
from rag_system.tools.web_search import get_web_search_tool
from rag_system.parsers.document_parser import get_document_parser

class RAGWorkflow:
    def __init__(self):
        self.config = get_config()
        self.llm_router = get_llm_router()
        self.simple_detector = get_simple_detector()
        self.attachment_handler = get_attachment_handler()
        self.retrieval = get_hybrid_retrieval_service()
        self.weather_tool = get_weather_tool()
        self.finance_tool = get_finance_tool()
        self.transport_tool = get_transport_tool()
        self.web_search_tool = get_web_search_tool()
        self.document_parser = get_document_parser()
    
    def execute(self, query: str, strict_local: bool = False, fast_mode: bool = False, files: Optional[List[str]] = None, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        start_time = datetime.now()
        
        def report_progress(stage: str):
            if progress_callback:
                progress_callback(stage)
        
        if files:
            report_progress("Parsing attachments...")
            attachments = self.attachment_handler.parse_files(files, progress_callback=report_progress)
            attachment_context = self.attachment_handler.format_for_prompt(attachments)
            
            report_progress("Generating answer with attachments...")
            language = self.simple_detector.detect_language(query)
            answer = self.llm_router.answer_with_attachments(query, attachment_context, language=language)
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                'query': query,
                'answer': answer,
                'routing': {
                    'sources': ['attachments'],
                    'reasoning': f'Direct LLM processing with {len(files)} attached file(s)',
                    'query': query
                },
                'sources_used': ['attachments'],
                'tool_results': {
                    'attachments': [
                        {
                            'filename': att.filename,
                            'file_type': att.file_type,
                            'token_estimate': att.token_estimate,
                            'metadata': att.metadata
                        } for att in attachments
                    ]
                },
                'failed_tools': [],
                'context_count': len(attachments),
                'citations': [],
                'latency_ms': latency_ms,
                'timestamp': end_time.isoformat(),
                'attachments': True
            }
        
        report_progress("Analyzing query...")
        
        if not strict_local and self.simple_detector.is_simple(query):
            report_progress("Generating answer (fast path)...")
            language = self.simple_detector.detect_language(query)
            answer = self.llm_router.answer_direct(query, language=language)
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                'query': query,
                'answer': answer,
                'routing': {
                    'sources': [],
                    'reasoning': 'Simple question - answered directly using LLM knowledge',
                    'query': query
                },
                'sources_used': [],
                'tool_results': {},
                'failed_tools': [],
                'context_count': 0,
                'citations': [],
                'latency_ms': latency_ms,
                'timestamp': end_time.isoformat(),
                'fast_path': True
            }
        
        report_progress("Routing query...")
        
        if strict_local:
            routing = {
                'sources': ['local_knowledge_base'],
                'reasoning': 'Strict local mode enabled',
                'query': query
            }
        else:
            routing = self.llm_router.route_query(query)
        
        sources = routing['sources']
        
        all_context = []
        tool_results = {}
        failed_tools = []
        
        min_context_threshold = 3
        
        if 'local_knowledge_base' in sources:
            report_progress("Retrieving from knowledge base...")
            local_docs = self.retrieval.retrieve(query)
            all_context.extend(local_docs)
            tool_results['local_knowledge_base'] = {
                'count': len(local_docs),
                'docs': local_docs
            }
        
        if not strict_local:
            domain_tools_used = []
            
            if 'finance' in sources:
                report_progress("Fetching finance data...")
                finance_results = self._handle_finance(query)
                tool_results['finance'] = finance_results
                domain_tools_used.append('finance')
                if 'data' in finance_results and finance_results['data']:
                    all_context.append({
                        'text': str(finance_results),
                        'source': 'finance',
                        'credibility_score': 0.9,
                        'final_score': 0.85
                    })
                elif 'error' in finance_results:
                    failed_tools.append('finance')
                    
                    tickers = self._extract_tickers(query)
                    if tickers and len(tickers) == 1:
                        web_extraction_result = self._try_web_extraction_for_finance(tickers[0])
                        if web_extraction_result:
                            tool_results['finance_web_extraction'] = web_extraction_result
                            all_context.append({
                                'text': str(web_extraction_result),
                                'source': 'finance_web_extraction',
                                'credibility_score': 0.85,
                                'final_score': 0.8
                            })
            
            if 'weather' in sources:
                report_progress("Fetching weather data...")
                weather_results = self._handle_weather(query)
                tool_results['weather'] = weather_results
                domain_tools_used.append('weather')
                if 'data' in weather_results and weather_results['data']:
                    all_context.append({
                        'text': str(weather_results),
                        'source': 'weather',
                        'credibility_score': 0.85,
                        'final_score': 0.8
                    })
                elif 'error' in weather_results:
                    failed_tools.append('weather')
            
            if 'transport' in sources:
                report_progress("Fetching transport data...")
                transport_results = self._handle_transport(query)
                tool_results['transport'] = transport_results
                domain_tools_used.append('transport')
                if 'data' in transport_results and transport_results['data']:
                    all_context.append({
                        'text': str(transport_results),
                        'source': 'transport',
                        'credibility_score': 0.8,
                        'final_score': 0.75
                    })
                elif 'error' in transport_results:
                    failed_tools.append('transport')
            
            should_use_web_search = (
                'web_search' in sources or
                len(domain_tools_used) > 0 or
                len(failed_tools) > 0 or
                len(all_context) < min_context_threshold
            )
            
            if should_use_web_search and not fast_mode:
                report_progress("Searching the web...")
                web_query = self._enhance_query_for_web_search(query, domain_tools_used)
                web_results = self.web_search_tool.search(web_query, max_results=5)
                if 'results' in web_results:
                    for result in web_results['results']:
                        all_context.append({
                            'text': result.get('content', result.get('snippet', '')),
                            'source': 'web_search',
                            'url': result.get('url', ''),
                            'credibility_score': 0.6,
                            'final_score': 0.7
                        })
                tool_results['web_search'] = web_results
                if 'web_search' not in sources:
                    sources.append('web_search')
        
        citations = self._build_citations(all_context)
        
        report_progress("Generating answer...")
        answer = self.llm_router.synthesize_answer(query, all_context[:10], citations)
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            'query': query,
            'answer': answer,
            'routing': routing,
            'sources_used': sources,
            'tool_results': tool_results,
            'failed_tools': failed_tools,
            'context_count': len(all_context),
            'citations': citations,
            'latency_ms': latency_ms,
            'timestamp': end_time.isoformat()
        }
    
    def _handle_finance(self, query: str) -> Dict[str, Any]:
        tickers = self._extract_tickers(query)
        
        if not tickers:
            return {'error': 'No stock tickers found in query'}
        
        query_lower = query.lower()
        use_intraday = any(keyword in query_lower for keyword in ['current', 'now', 'today', 'latest', 'real-time', 'realtime'])
        
        if len(tickers) == 1:
            return self.finance_tool.get_stock_price(tickers[0], use_intraday=use_intraday)
        else:
            return self.finance_tool.compare_stocks(tickers)
    
    def _handle_weather(self, query: str) -> Dict[str, Any]:
        location = self._extract_location(query)
        date = self._extract_date(query)
        
        if not location:
            location = "New York"
        
        return self.weather_tool.get_weather(location, date)
    
    def _handle_transport(self, query: str) -> Dict[str, Any]:
        locations = self._extract_locations(query)
        
        if len(locations) < 2:
            return {'error': 'Need origin and destination for transport query'}
        
        return self.transport_tool.get_route(locations[0], locations[1])
    
    def _extract_tickers(self, query: str) -> List[str]:
        words = query.upper().split()
        common_tickers = ['NVDA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
        
        found_tickers = []
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word in common_tickers:
                found_tickers.append(clean_word)
        
        return found_tickers
    
    def _extract_location(self, query: str) -> Optional[str]:
        words = query.split()
        
        for i, word in enumerate(words):
            if word.lower() in ['in', 'at', 'for']:
                if i + 1 < len(words):
                    return ' '.join(words[i+1:i+3])
        
        return None
    
    def _extract_locations(self, query: str) -> List[str]:
        locations = []
        
        if ' to ' in query.lower():
            parts = query.lower().split(' to ')
            if len(parts) >= 2:
                locations.append(parts[0].split()[-1])
                locations.append(parts[1].split()[0])
        
        return locations
    
    def _extract_date(self, query: str) -> Optional[str]:
        return None
    
    def _enhance_query_for_web_search(self, query: str, domain_tools: List[str]) -> str:
        if 'finance' in domain_tools:
            tickers = self._extract_tickers(query)
            if tickers:
                if len(tickers) > 1:
                    return f"{' vs '.join(tickers)} stock price comparison today"
                else:
                    return f"{tickers[0]} stock price today latest news"
        
        if 'weather' in domain_tools:
            location = self._extract_location(query)
            if location:
                return f"{location} weather forecast today"
        
        if 'transport' in domain_tools:
            locations = self._extract_locations(query)
            if len(locations) >= 2:
                return f"driving time distance {locations[0]} to {locations[1]}"
        
        return query
    
    def _try_web_extraction_for_finance(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            urls_to_try = [
                f"https://finance.yahoo.com/quote/{ticker}",
                f"https://www.cnbc.com/quotes/{ticker}",
                f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
            ]
            
            for url in urls_to_try:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        result = self.finance_tool.extract_price_from_web(ticker, response.text, url)
                        if result:
                            result['url'] = url
                            return result
                except Exception:
                    continue
            
            return None
        except Exception:
            return None
    
    def _is_general_knowledge_query(self, query: str) -> bool:
        general_knowledge_keywords = [
            'what is', 'who is', 'who wrote', 'who invented', 'when was', 'where is',
            'how many', 'how much', 'what are', 'what does', 'define', 'explain',
            'capital of', 'formula for', 'planet', 'color', 'olympic', 'emergency',
            'phone number', 'multiplied', 'divided', 'plus', 'minus', 'subtract',
            '什麼是', '誰是', '誰寫', '什麼時候', '哪裡', '多少', '怎麼', '如何',
            '首都', '公式', '行星', '顏色', '電話', '乘', '除', '加', '減'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in general_knowledge_keywords)
    
    def _build_citations(self, context: List[Dict[str, Any]]) -> List[str]:
        citations = []
        for i, doc in enumerate(context[:10]):
            source = doc.get('source', 'Unknown')
            url = doc.get('url', '')
            
            if url:
                citations.append(f"[{i+1}] {source}: {url}")
            else:
                citations.append(f"[{i+1}] {source}")
        
        return citations

_rag_workflow = None

def get_rag_workflow() -> RAGWorkflow:
    global _rag_workflow
    if _rag_workflow is None:
        _rag_workflow = RAGWorkflow()
    return _rag_workflow
