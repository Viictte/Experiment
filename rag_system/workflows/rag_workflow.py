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
        
        # Perform query analysis for better domain detection and location extraction
        query_analysis = self.llm_router.analyze_query(query)
        
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
                'fast_path': True,
                'query_analysis': query_analysis
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
        
        min_kb_threshold = 3
        kb_docs_count = 0
        
        if 'local_knowledge_base' in sources:
            report_progress("Retrieving from knowledge base...")
            local_docs = self.retrieval.retrieve(query)
            all_context.extend(local_docs)
            kb_docs_count = len(local_docs)
            tool_results['local_knowledge_base'] = {
                'count': len(local_docs),
                'docs': local_docs
            }
        
        if not strict_local:
            domain_tools_used = []
            
            if 'finance' in sources:
                report_progress("Fetching finance data...")
                
                # Check cache first for consistency
                from rag_system.services.redis_service import get_redis_service
                redis = get_redis_service()
                cache_params = {'query': query, 'tool': 'finance'}
                cached_result = redis.get_tool_cache('finance', cache_params)
                
                if cached_result:
                    finance_results = cached_result
                else:
                    finance_results = self._handle_finance(query)
                    # Cache for 5 minutes (300 seconds)
                    redis.set_tool_cache('finance', cache_params, finance_results, ttl=300)
                
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
                
                # Use location from query analysis if available
                location = query_analysis.get('location', '')
                
                # Check cache first for consistency
                from rag_system.services.redis_service import get_redis_service
                redis = get_redis_service()
                cache_params = {'query': query, 'location': location, 'tool': 'weather'}
                cached_result = redis.get_tool_cache('weather', cache_params)
                
                if cached_result:
                    weather_results = cached_result
                else:
                    weather_results = self._handle_weather(query, location=location)
                    # Cache for 10 minutes (600 seconds) - weather changes slowly
                    redis.set_tool_cache('weather', cache_params, weather_results, ttl=600)
                
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
                
                # Check cache first for consistency
                from rag_system.services.redis_service import get_redis_service
                redis = get_redis_service()
                cache_params = {'query': query, 'tool': 'transport'}
                cached_result = redis.get_tool_cache('transport', cache_params)
                
                if cached_result:
                    transport_results = cached_result
                else:
                    transport_results = self._handle_transport(query)
                    # Cache for 15 minutes (900 seconds) - routes change less frequently
                    redis.set_tool_cache('transport', cache_params, transport_results, ttl=900)
                
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
            
            # Check if we have rich domain context from specialized tools
            has_domain_context = any(
                doc.get('source') in {'finance', 'weather', 'transport', 'finance_web_extraction'}
                for doc in all_context
            )
            
            # Only use web search when:
            # 1. Router explicitly requested it, OR
            # 2. Domain tools failed (need fallback), OR
            # 3. KB is insufficient AND we don't have domain context
            should_use_web_search = (
                'web_search' in sources or
                len(failed_tools) > 0 or
                (kb_docs_count < min_kb_threshold and not has_domain_context)
            )
            
            if should_use_web_search and not fast_mode:
                report_progress("Searching the web...")
                
                # Use query expansions from analysis for better recall
                query_expansions = query_analysis.get('query_expansions', [query])
                search_query = query_expansions[0] if query_expansions else query
                
                # Apply domain filtering based on query analysis
                filters = None
                domain = query_analysis.get('domain', 'general')
                location = query_analysis.get('location', '')
                
                # For HK-specific queries, prefer HK domains
                if location and 'hong kong' in location.lower():
                    filters = {
                        'preferred_domains': ['gov.hk', 'hkpl.gov.hk', 'hko.gov.hk', 'td.gov.hk', 'info.gov.hk'],
                        'blocked_domains': []
                    }
                
                # Block low-quality sources for factual queries
                if domain in ['weather', 'finance', 'hk_local']:
                    if filters is None:
                        filters = {'preferred_domains': [], 'blocked_domains': []}
                    filters['blocked_domains'].extend(['reddit.com', 'facebook.com', 'quora.com'])
                
                web_results = self.web_search_tool.search(search_query, max_results=5, filters=filters)
                if 'results' in web_results:
                    for result in web_results['results']:
                        # Handle both Google (uses 'url') and Tavily (uses 'url') formats
                        url = result.get('url', result.get('link', ''))
                        title = result.get('title', '')
                        domain_str = result.get('domain', '')
                        all_context.append({
                            'text': result.get('content', result.get('snippet', '')),
                            'source': 'web_search',
                            'url': url,
                            'title': title,
                            'domain': domain_str,
                            'credibility_score': 0.6,
                            'final_score': 0.7
                        })
                tool_results['web_search'] = web_results
                if 'web_search' not in sources:
                    sources.append('web_search')
        
        citations = self._build_citations(all_context)
        
        # Check if we have meaningful context before generating answer
        has_context = self._has_meaningful_context(all_context)
        
        if not has_context:
            # Unified fallback message instead of random LLM behavior
            answer = (
                "I don't know based on the available information. "
                "The current data sources do not provide a reliable answer to this question."
            )
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                'query': query,
                'answer': answer,
                'routing': routing,
                'sources_used': [],
                'tool_results': tool_results,
                'failed_tools': failed_tools,
                'context_count': 0,
                'citations': [],
                'latency_ms': latency_ms,
                'timestamp': end_time.isoformat(),
                'answerability': 'insufficient_context'
            }
        
        report_progress("Generating answer...")
        
        # Determine if we should use strict grounding based on sources
        # Use strict grounding for factual/time-sensitive queries (weather, finance, transport, web search)
        use_strict_grounding = any(
            source in sources for source in ['weather', 'finance', 'transport', 'web_search']
        )
        
        answer = self.llm_router.synthesize_answer(
            query, 
            all_context[:10], 
            citations,
            allow_direct_knowledge=False,
            strict_grounding=use_strict_grounding
        )
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Derive sources_used from actual context instead of routing intent
        sources_used = sorted(set(doc.get('source', 'Unknown') for doc in all_context))
        
        return {
            'query': query,
            'answer': answer,
            'routing': routing,
            'sources_used': sources_used,
            'tool_results': tool_results,
            'failed_tools': failed_tools,
            'context_count': len(all_context),
            'citations': citations,
            'latency_ms': latency_ms,
            'timestamp': end_time.isoformat(),
            'query_analysis': query_analysis
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
    
    def _handle_weather(self, query: str, location: str = '') -> Dict[str, Any]:
        # Use provided location from query analysis, or extract from query
        if not location:
            location = self._extract_location(query)
        
        date = self._extract_date(query)
        
        # If still no location, return error instead of defaulting to New York
        # This prevents wrong-city responses
        if not location:
            return {'error': 'No location specified in query'}
        
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
    
    def _has_meaningful_context(self, context: List[Dict[str, Any]]) -> bool:
        """Check if we have meaningful context to answer the query"""
        if not context:
            return False
        
        # Heuristic: any doc from trusted tools or with a decent score
        for doc in context:
            source = doc.get('source', '')
            
            # Trust structured tool outputs
            if source in {'weather', 'finance', 'transport', 'finance_web_extraction'}:
                return True
            
            # Check if we have a good score from retrieval/reranking
            score = doc.get('final_score') or doc.get('score') or doc.get('credibility_score')
            if score is not None and score >= 0.5:
                return True
        
        return False
    
    def _build_citations(self, context: List[Dict[str, Any]]) -> List[str]:
        citations = []
        for i, doc in enumerate(context[:10]):
            source = doc.get('source', 'Unknown')
            url = doc.get('url', '')
            title = doc.get('title', '')
            
            # Format citations based on source type
            if source == 'web_search':
                if url and title:
                    citations.append(f"[{i+1}] Web: {title} - {url}")
                elif url:
                    citations.append(f"[{i+1}] Web: {url}")
                else:
                    citations.append(f"[{i+1}] Web Search")
            elif source == 'local_knowledge_base':
                # For local KB, show the document source (file path or URL)
                if url:
                    citations.append(f"[{i+1}] Local KB: {url}")
                else:
                    doc_id = doc.get('doc_id', '')
                    if doc_id:
                        citations.append(f"[{i+1}] Local KB: {doc_id}")
                    else:
                        citations.append(f"[{i+1}] Local Knowledge Base")
            elif url:
                # Other sources with URLs (finance_web_extraction, etc.)
                citations.append(f"[{i+1}] {source}: {url}")
            else:
                # Sources without URLs (finance API, weather API, etc.)
                citations.append(f"[{i+1}] {source}")
        
        return citations

_rag_workflow = None

def get_rag_workflow() -> RAGWorkflow:
    global _rag_workflow
    if _rag_workflow is None:
        _rag_workflow = RAGWorkflow()
    return _rag_workflow
