"""RAG Workflow orchestration using LangGraph"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import asyncio
import requests
import pytz
from rag_system.core.config import get_config
from rag_system.workflows.llm_router import get_llm_router
from rag_system.workflows.simple_detector import get_simple_detector
from rag_system.workflows.attachment_handler import get_attachment_handler
from rag_system.workflows.planner_v2 import get_planner_v2
from rag_system.workflows.evaluator import get_evaluator
from rag_system.services.hybrid_retrieval import get_hybrid_retrieval_service
from rag_system.tools.weather import get_weather_tool
from rag_system.tools.finance import get_finance_tool
from rag_system.tools.transport import get_transport_tool
from rag_system.tools.web_search import get_web_search_tool
from rag_system.parsers.document_parser import get_document_parser

class RAGWorkflow:
    # Hong Kong location normalization mapping
    HK_LOCATIONS = {
        # Hong Kong Island
        '中環': 'Central, Hong Kong', '金鐘': 'Admiralty, Hong Kong', '灣仔': 'Wan Chai, Hong Kong',
        '銅鑼灣': 'Causeway Bay, Hong Kong', '天后': 'Tin Hau, Hong Kong', '炮台山': 'Fortress Hill, Hong Kong',
        '北角': 'North Point, Hong Kong', '鰂魚涌': 'Quarry Bay, Hong Kong', '太古': 'Taikoo, Hong Kong',
        '西灣河': 'Sai Wan Ho, Hong Kong', '筲箕灣': 'Shau Kei Wan, Hong Kong', '杏花邨': 'Heng Fa Chuen, Hong Kong',
        '柴灣': 'Chai Wan, Hong Kong', '上環': 'Sheung Wan, Hong Kong', '西營盤': 'Sai Ying Pun, Hong Kong',
        '石塘咀': 'Shek Tong Tsui, Hong Kong', '堅尼地城': 'Kennedy Town, Hong Kong',
        # Kowloon
        '尖沙咀': 'Tsim Sha Tsui, Hong Kong', '佐敦': 'Jordan, Hong Kong', '油麻地': 'Yau Ma Tei, Hong Kong',
        '旺角': 'Mong Kok, Hong Kong', '太子': 'Prince Edward, Hong Kong', '石硤尾': 'Shek Kip Mei, Hong Kong',
        '九龍塘': 'Kowloon Tong, Hong Kong', '樂富': 'Lok Fu, Hong Kong', '黃大仙': 'Wong Tai Sin, Hong Kong',
        '鑽石山': 'Diamond Hill, Hong Kong', '彩虹': 'Choi Hung, Hong Kong', '九龍灣': 'Kowloon Bay, Hong Kong',
        '牛頭角': 'Ngau Tau Kok, Hong Kong', '觀塘': 'Kwun Tong, Hong Kong', '藍田': 'Lam Tin, Hong Kong',
        '油塘': 'Yau Tong, Hong Kong', '紅磡': 'Hung Hom, Hong Kong', '何文田': 'Ho Man Tin, Hong Kong',
        '土瓜灣': 'To Kwa Wan, Hong Kong',
        # New Territories
        '荃灣': 'Tsuen Wan, Hong Kong', '葵涌': 'Kwai Chung, Hong Kong', '青衣': 'Tsing Yi, Hong Kong',
        '沙田': 'Sha Tin, Hong Kong', '大圍': 'Tai Wai, Hong Kong', '馬鞍山': 'Ma On Shan, Hong Kong',
        '大埔': 'Tai Po, Hong Kong', '粉嶺': 'Fanling, Hong Kong', '上水': 'Sheung Shui, Hong Kong',
        '元朗': 'Yuen Long, Hong Kong', '天水圍': 'Tin Shui Wai, Hong Kong', '屯門': 'Tuen Mun, Hong Kong',
        '將軍澳': 'Tseung Kwan O, Hong Kong', '西貢': 'Sai Kung, Hong Kong', '東涌': 'Tung Chung, Hong Kong',
    }
    
    def __init__(self):
        self.config = get_config()
        self.llm_router = get_llm_router()
        self.simple_detector = get_simple_detector()
        self.attachment_handler = get_attachment_handler()
        self.planner_v2 = get_planner_v2()
        self.evaluator = get_evaluator()
        self.retrieval = get_hybrid_retrieval_service()
        self.weather_tool = get_weather_tool()
        self.finance_tool = get_finance_tool()
        self.transport_tool = get_transport_tool()
        self.web_search_tool = get_web_search_tool()
        self.document_parser = get_document_parser()
        # Default timezone for time-aware queries
        self.default_timezone = pytz.timezone('Asia/Hong_Kong')
        # Initialize time tool
        from rag_system.tools.time_tool import get_time_tool
        self.time_tool = get_time_tool()
    
    def _get_current_time_context(self, query: str) -> str:
        """Get current time context for time-sensitive queries"""
        # Check if query contains time-sensitive keywords
        time_keywords = ['now', 'current', 'currently', 'right now', 'this moment', 'tonight', 'this afternoon', 
                        'this morning', 'this evening', 'today', 'open now', '現在', '目前', '正在', '今天', '今晚']
        
        query_lower = query.lower()
        is_time_sensitive = any(keyword in query_lower for keyword in time_keywords)
        
        if not is_time_sensitive:
            return ""
        
        # Get current time in UTC and local timezone
        now_utc = datetime.now(pytz.UTC)
        now_local = now_utc.astimezone(self.default_timezone)
        
        time_context = f"\n\nCurrent Time Information:\n"
        time_context += f"- UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        time_context += f"- Local ({self.default_timezone.zone}): {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        time_context += f"- Day of week: {now_local.strftime('%A')}\n"
        
        return time_context
    
    def execute(self, query: str, strict_local: bool = False, fast_mode: bool = False, allow_web_search: bool = True, files: Optional[List[str]] = None, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
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
        
        # Normalize Hong Kong location names (Chinese -> English + HK flag)
        is_hk_query = False
        if query_analysis.get('location'):
            location = query_analysis['location']
            # Check if location matches any HK district
            for hk_chinese, hk_english in self.HK_LOCATIONS.items():
                if hk_chinese in location:
                    query_analysis['location'] = hk_english
                    query_analysis['is_hk_query'] = True
                    is_hk_query = True
                    break
            # Also check if "hong kong" is already in the location
            if not is_hk_query and 'hong kong' in location.lower():
                query_analysis['is_hk_query'] = True
                is_hk_query = True
        
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
        
        # Use V2 planner for intelligent query planning
        report_progress("Planning query...")
        query_plan = self.planner_v2.plan_query(query)
        
        # Handle based on plan mode
        if query_plan['mode'] == 'direct_llm':
            # Direct LLM answer (simple queries, noise, etc.)
            report_progress("Generating answer (direct LLM)...")
            language = self.simple_detector.detect_language(query)
            answer = self.llm_router.answer_direct(query, language=language)
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                'query': query,
                'answer': answer,
                'routing': {
                    'sources': [],
                    'reasoning': query_plan.get('reason', 'Direct LLM mode'),
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
                'query_plan': query_plan
            }
        
        # Execute retrieval-based queries with V2 workflow (includes evaluation loop)
        if query_plan['mode'] in ['single_retrieval', 'multi_retrieval']:
            return self._execute_v2_workflow(query, query_plan, query_analysis, strict_local, allow_web_search, start_time, report_progress)
        
        # Fallback to old routing if plan mode is unexpected
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
        
        # Add keyword-based fallback routing for transport/finance queries
        # This ensures these APIs are called even if LLM router misses them
        query_lower = query.lower()
        
        # Transport keywords
        transport_keywords = [
            'how do i go', 'how to go', 'how do i get', 'how to get', 'travel from', 'travel to',
            'go from', 'go to', 'route from', 'route to', 'directions', 'fastest way',
            'bus', 'mtr', 'train', 'taxi', 'transport', 'commute',
            '怎麼去', '怎么去', '如何去', '怎樣去', '怎样去', '路線', '路线', '交通'
        ]
        if any(kw in query_lower for kw in transport_keywords) and 'transport' not in sources:
            # Check if query mentions two locations (origin and destination)
            locations = self._extract_locations(query)
            if len(locations) >= 2:
                sources.append('transport')
                routing['reasoning'] += ' (transport added via keyword detection)'
        
        # Finance keywords
        finance_keywords = [
            'stock', 'share', 'price of', 'stock price', 'market', 'ticker',
            'hkd', 'usd', 'exchange rate', 'currency', 'forex',
            '股票', '股价', '股價', '市場', '市场', '匯率', '汇率', '價格', '价格'
        ]
        if any(kw in query_lower for kw in finance_keywords) and 'finance' not in sources:
            sources.append('finance')
            routing['reasoning'] += ' (finance added via keyword detection)'
        
        # Time keywords (highest priority - should never use web search)
        time_keywords = [
            'what time', 'current time', 'time now', 'time right now', 'local time',
            'what\'s the time', 'tell me the time', 'time in', 'time at',
            '現在幾點', '现在几点', '現在時間', '现在时间', '當前時間', '当前时间'
        ]
        if any(kw in query_lower for kw in time_keywords):
            # For time queries, ONLY use time tool (no web search, no other tools)
            sources = ['time']
            routing['sources'] = ['time']
            routing['reasoning'] = 'Time query detected - using time API only (no web search needed)'
        
        # Weather keywords (including air quality, AQHI, etc.)
        weather_keywords = [
            'weather', 'forecast', 'temperature', 'rain', 'snow', 'uv', 'ultraviolet',
            'air quality', 'aqhi', 'aqi', 'humidity', 'wind',
            '天氣', '天气', '氣溫', '气温', '降雨', '紫外線', '紫外线', 
            '空氣質素', '空气质素', '空氣品質', '空气品质', '濕度', '湿度'
        ]
        # Only add weather if not a time query (to avoid confusion)
        if any(kw in query_lower for kw in weather_keywords) and 'weather' not in sources and not self._is_time_query(query):
            sources.append('weather')
            routing['reasoning'] += ' (weather added via keyword detection)'
        
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
            
            # Check for time queries first (highest priority, no web search needed)
            if self._is_time_query(query):
                report_progress("Getting current time...")
                location = query_analysis.get('location', '')
                time_results = self._handle_time(query, location=location)
                tool_results['time'] = time_results
                domain_tools_used.append('time')
                if 'error' not in time_results:
                    # Format time data for context
                    time_text = self._format_time_for_context(time_results)
                    all_context.append({
                        'text': time_text,
                        'source': 'time',
                        'credibility_score': 1.0,
                        'final_score': 1.0
                    })
                    # For time queries, skip web search and other tools
                    sources = ['time']
            
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
                
                # Check if this is an air quality query
                air_quality_keywords = ['air quality', 'aqhi', 'aqi', '空氣質素', '空气质素', '空氣品質', '空气品质']
                is_air_quality_query = any(kw in query.lower() for kw in air_quality_keywords)
                
                # Check cache first for consistency
                from rag_system.services.redis_service import get_redis_service
                redis = get_redis_service()
                cache_params = {'query': query, 'location': location, 'tool': 'weather', 'air_quality': is_air_quality_query}
                cached_result = redis.get_tool_cache('weather', cache_params)
                
                if cached_result:
                    weather_results = cached_result
                elif is_air_quality_query:
                    # Try Google Air Quality API first for air quality queries
                    weather_results = self.weather_tool.get_air_quality(location)
                    if 'error' in weather_results:
                        # Fallback to regular weather API
                        weather_results = self._handle_weather(query, location=location)
                    # Cache for 10 minutes (600 seconds)
                    redis.set_tool_cache('weather', cache_params, weather_results, ttl=600)
                else:
                    weather_results = self._handle_weather(query, location=location)
                    # Cache for 10 minutes (600 seconds) - weather changes slowly
                    redis.set_tool_cache('weather', cache_params, weather_results, ttl=600)
                
                tool_results['weather'] = weather_results
                domain_tools_used.append('weather')
                
                if is_air_quality_query and 'aqi' in weather_results:
                    # Format air quality data for context
                    aq_text = f"Air Quality for {location}:\n"
                    aq_text += f"AQI: {weather_results.get('aqi_display', 'N/A')}\n"
                    aq_text += f"Category: {weather_results.get('category', 'N/A')}\n"
                    aq_text += f"Dominant Pollutant: {weather_results.get('dominant_pollutant', 'N/A')}\n"
                    if weather_results.get('health_recommendations'):
                        aq_text += f"Health Recommendations: {weather_results['health_recommendations']}\n"
                    all_context.append({
                        'text': aq_text,
                        'source': 'weather',
                        'credibility_score': 0.95,
                        'final_score': 0.9
                    })
                elif 'data' in weather_results and weather_results['data']:
                    # Format weather data in a structured, readable way
                    weather_text = self._format_weather_for_context(weather_results)
                    all_context.append({
                        'text': weather_text,
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
                    # Format transport data for context
                    transport_text = self._format_transport_for_context(transport_results)
                    all_context.append({
                        'text': transport_text,
                        'source': 'transport',
                        'credibility_score': 0.9,
                        'final_score': 0.85
                    })
                elif 'error' in transport_results:
                    failed_tools.append('transport')
            
            # Check if we have rich domain context from specialized tools
            has_domain_context = any(
                doc.get('source') in {'finance', 'weather', 'transport', 'finance_web_extraction', 'time'}
                for doc in all_context
            )
            
            # Check if query needs extra info beyond domain tools (AQHI, hiking trails, etc.)
            needs_extra_info = any(kw in query.lower() for kw in [
                'aqhi', 'air quality', '空氣質素', '空气质素',
                'trail', 'hiking', '遠足', '远足', '行山', 'safety', 'safe'
            ])
            
            # Check if this is a time query (should never use web search)
            is_time_query = self._is_time_query(query)
            
            # Only use web search when:
            # 1. User allows web search (allow_web_search=True), AND
            # 2. NOT a time query (time queries should never use web search), AND
            # 3. One of these conditions:
            #    a. Router explicitly requested it, OR
            #    b. Domain tools failed (need fallback), OR
            #    c. KB is insufficient AND we don't have domain context, OR
            #    d. Query needs extra info beyond what domain tools provide
            should_use_web_search = (
                not is_time_query and (
                    'web_search' in sources or
                    len(failed_tools) > 0 or
                    (kb_docs_count < min_kb_threshold and not has_domain_context) or
                    needs_extra_info
                )
            )
            
            if allow_web_search and should_use_web_search and not fast_mode:
                report_progress("Searching the web...")
                
                # Use query expansions from analysis for better recall
                query_expansions = query_analysis.get('query_expansions', [query])
                search_query = query_expansions[0] if query_expansions else query
                
                # Apply domain filtering based on query analysis
                filters = None
                domain = query_analysis.get('domain', 'general')
                location = query_analysis.get('location', '')
                
                # For HK-specific queries, prefer HK domains
                is_hk = query_analysis.get('is_hk_query', False) or (location and 'hong kong' in location.lower())
                if is_hk:
                    filters = {
                        'preferred_domains': ['gov.hk', 'hkpl.gov.hk', 'hko.gov.hk', 'td.gov.hk', 'info.gov.hk'],
                        'blocked_domains': []
                    }
                    # For HK cuisine queries, prefer OpenRice and TripAdvisor
                    if domain == 'cuisine':
                        filters['preferred_domains'].extend(['openrice.com', 'openrice.com.hk', 'tripadvisor.com.hk'])
                
                # Block low-quality sources for factual queries
                if domain in ['weather', 'finance', 'hk_local', 'cuisine']:
                    if filters is None:
                        filters = {'preferred_domains': [], 'blocked_domains': []}
                    filters['blocked_domains'].extend(['reddit.com', 'facebook.com', 'quora.com', 'instagram.com', 'youtube.com'])
                
                web_results = self.web_search_tool.search(search_query, max_results=5, filters=filters)
                
                # If Tavily returned a synthesized answer, add it as highest-priority context
                if 'answer' in web_results and web_results['answer']:
                    all_context.append({
                        'text': web_results['answer'],
                        'source': 'web_search',
                        'url': '',
                        'title': 'Tavily AI Answer',
                        'domain': 'tavily.com',
                        'credibility_score': 1.0,  # Highest credibility for Tavily's answer
                        'final_score': 1.0
                    })
                
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
    
    def _execute_v2_workflow(self, query: str, query_plan: Dict[str, Any], query_analysis: Dict[str, Any], strict_local: bool, allow_web_search: bool, start_time: datetime, report_progress: Callable[[str], None]) -> Dict[str, Any]:
        """
        Execute V2 workflow with planner + retrieval + evaluator + optional refinement.
        
        Args:
            query: Original user query
            query_plan: Plan from planner_v2
            query_analysis: Query analysis results
            strict_local: Whether to use only local KB
            allow_web_search: Whether web search is allowed
            start_time: Query start time
            report_progress: Progress callback function
            
        Returns:
            Complete query result with evaluation metadata
        """
        max_iterations = 2
        iteration = 1
        
        all_subqueries = query_plan.get('subqueries', [])
        
        # First pass: execute all subqueries from planner
        all_context, tool_results, failed_tools, sources_used = self._execute_subqueries(
            all_subqueries, query_analysis, allow_web_search, report_progress, iteration
        )
        
        # Synthesize answer from first pass
        report_progress("Synthesizing answer...")
        language = self.simple_detector.detect_language(query)
        context_text = self._build_context_text(all_context)
        
        answer = self.llm_router.synthesize_answer(
            query=query,
            context=context_text,
            language=language,
            query_type='factual',
            strict_grounding=True
        )
        
        # Evaluate answer completeness
        report_progress("Evaluating answer completeness...")
        evidence_summary = self._build_evidence_summary(all_context, tool_results)
        evaluation = self.evaluator.evaluate(query, answer, evidence_summary)
        
        # If incomplete and we have follow-up queries, do second pass
        if not evaluation.get('complete', True) and evaluation.get('followup_subqueries') and iteration < max_iterations:
            iteration += 1
            report_progress(f"Answer incomplete, executing follow-up queries (pass {iteration})...")
            
            followup_subqueries = evaluation['followup_subqueries']
            
            # Execute follow-up subqueries
            followup_context, followup_tool_results, followup_failed, followup_sources = self._execute_subqueries(
                followup_subqueries, query_analysis, allow_web_search, report_progress, iteration
            )
            
            # Merge with existing context
            all_context.extend(followup_context)
            tool_results.update(followup_tool_results)
            failed_tools.extend(followup_failed)
            sources_used.update(followup_sources)
            
            # Re-synthesize with expanded context
            report_progress("Re-synthesizing answer with additional information...")
            context_text = self._build_context_text(all_context)
            
            answer = self.llm_router.synthesize_answer(
                query=query,
                context=context_text,
                language=language,
                query_type='factual',
                strict_grounding=True
            )
            
            # Re-evaluate (for metadata only, don't iterate further)
            evidence_summary = self._build_evidence_summary(all_context, tool_results)
            evaluation = self.evaluator.evaluate(query, answer, evidence_summary)
        
        # Build citations
        citations = self._build_citations(all_context)
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            'query': query,
            'answer': answer,
            'routing': {
                'sources': list(sources_used),
                'reasoning': query_plan.get('reason', 'V2 workflow with evaluation'),
                'query': query,
                'plan': query_plan
            },
            'sources_used': list(sources_used),
            'tool_results': tool_results,
            'failed_tools': failed_tools,
            'context_count': len(all_context),
            'citations': citations,
            'latency_ms': latency_ms,
            'timestamp': end_time.isoformat(),
            'query_analysis': query_analysis,
            'query_plan': query_plan,
            'evaluation': evaluation,
            'iterations': iteration,
            'v2_workflow': True
        }
    
    def _execute_subqueries(self, subqueries: List[Dict[str, Any]], query_analysis: Dict[str, Any], allow_web_search: bool, report_progress: Callable[[str], None], iteration: int) -> tuple:
        """Execute a list of subqueries and return combined results"""
        all_context = []
        tool_results = {}
        failed_tools = []
        sources_used = set()
        
        for i, subquery_info in enumerate(subqueries):
            subquery_id = subquery_info.get('id', f'sq{i+1}')
            subquery_text = subquery_info.get('query', '')
            tool = subquery_info.get('tool', 'web_search')
            
            report_progress(f"Pass {iteration}: Executing subquery {i+1}/{len(subqueries)} ({tool})...")
            
            try:
                if tool == 'web_search' and allow_web_search:
                    results = self.web_search_tool.search(subquery_text, max_results=5)
                    if results and 'results' in results:
                        for result in results['results']:
                            all_context.append({
                                'text': result.get('content', result.get('snippet', '')),
                                'source': 'web_search',
                                'url': result.get('url', ''),
                                'title': result.get('title', ''),
                                'domain': result.get('domain', ''),
                                'credibility_score': result.get('credibility_score', 0.5),
                                'final_score': result.get('final_score', 0.5),
                                'subquery_id': subquery_id
                            })
                        tool_results[f'web_search_{subquery_id}'] = results
                        sources_used.add('web_search')
                
                elif tool == 'weather':
                    location = self._extract_location(subquery_text) or query_analysis.get('location', 'Hong Kong')
                    weather_results = self._handle_weather(subquery_text, location=location)
                    if weather_results and 'error' not in weather_results:
                        weather_text = self._format_weather_for_context(weather_results)
                        all_context.append({
                            'text': weather_text,
                            'source': 'weather',
                            'credibility_score': 0.95,
                            'final_score': 0.9,
                            'subquery_id': subquery_id
                        })
                        tool_results[f'weather_{subquery_id}'] = weather_results
                        sources_used.add('weather')
                
                elif tool == 'finance':
                    finance_results = self._handle_finance(subquery_text)
                    if finance_results and 'error' not in finance_results:
                        tool_results[f'finance_{subquery_id}'] = finance_results
                        sources_used.add('finance')
                
                elif tool == 'transport':
                    locations = self._extract_locations(subquery_text)
                    if len(locations) >= 2:
                        transport_results = self._handle_transport(subquery_text, origin=locations[0], destination=locations[1])
                        if transport_results and 'error' not in transport_results:
                            transport_text = self._format_transport_for_context(transport_results)
                            all_context.append({
                                'text': transport_text,
                                'source': 'transport',
                                'credibility_score': 0.95,
                                'final_score': 0.9,
                                'subquery_id': subquery_id
                            })
                            tool_results[f'transport_{subquery_id}'] = transport_results
                            sources_used.add('transport')
                
                elif tool == 'time':
                    time_results = self._handle_time(subquery_text)
                    if time_results and 'error' not in time_results:
                        time_text = self._format_time_for_context(time_results)
                        all_context.append({
                            'text': time_text,
                            'source': 'time',
                            'credibility_score': 1.0,
                            'final_score': 1.0,
                            'subquery_id': subquery_id
                        })
                        tool_results[f'time_{subquery_id}'] = time_results
                        sources_used.add('time')
                
                elif tool == 'kb':
                    kb_results = self.retrieval.hybrid_search(subquery_text, top_k=5)
                    for doc in kb_results:
                        all_context.append({
                            'text': doc['text'],
                            'source': 'local_knowledge_base',
                            'metadata': doc.get('metadata', {}),
                            'score': doc.get('score', 0.5),
                            'final_score': doc.get('score', 0.5),
                            'subquery_id': subquery_id
                        })
                    sources_used.add('local_knowledge_base')
                    
            except Exception as e:
                print(f"Error executing subquery {subquery_id} ({tool}): {e}")
                failed_tools.append(f"{tool}_{subquery_id}")
        
        # Sort context by final_score
        all_context.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return all_context, tool_results, failed_tools, sources_used
    
    def _build_context_text(self, all_context: List[Dict[str, Any]]) -> str:
        """Build formatted context text from context list"""
        context_text = ""
        for i, ctx in enumerate(all_context[:20]):  # Top 20 contexts
            subquery_id = ctx.get('subquery_id', '')
            context_text += f"\n[{i+1}] "
            if subquery_id:
                context_text += f"[Subquery: {subquery_id}] "
            context_text += f"[Source: {ctx.get('source', 'unknown')}]\n"
            context_text += f"{ctx.get('text', '')}\n"
        return context_text
    
    def _build_evidence_summary(self, all_context: List[Dict[str, Any]], tool_results: Dict[str, Any]) -> str:
        """Build a brief evidence summary for evaluator"""
        sources = set()
        for ctx in all_context[:10]:
            sources.add(ctx.get('source', 'unknown'))
        
        summary = f"Retrieved information from {len(all_context)} sources: {', '.join(sources)}\n"
        summary += f"Tools used: {', '.join(tool_results.keys())}"
        return summary
    
    def _execute_complex_query(self, query: str, query_plan: Dict[str, Any], query_analysis: Dict[str, Any], strict_local: bool, allow_web_search: bool, start_time: datetime, report_progress: Callable[[str], None]) -> Dict[str, Any]:
        """
        Execute a complex query with multi-step retrieval based on the query plan.
        
        Args:
            query: Original user query
            query_plan: Plan with subqueries from the planner
            query_analysis: Query analysis results
            strict_local: Whether to use only local KB
            allow_web_search: Whether web search is allowed
            start_time: Query start time
            report_progress: Progress callback function
            
        Returns:
            Complete query result with combined contexts
        """
        all_context = []
        tool_results = {}
        failed_tools = []
        sources_used = set()
        
        # Execute each subquery
        for i, subquery_info in enumerate(query_plan['subqueries']):
            subquery_id = subquery_info.get('id', f'sq{i+1}')
            subquery_text = subquery_info.get('query', '')
            domain = subquery_info.get('domain', 'web_search')
            
            report_progress(f"Executing subquery {i+1}/{len(query_plan['subqueries'])}: {domain}...")
            
            try:
                # Execute based on domain
                if domain == 'web_search' and allow_web_search:
                    results = self.web_search_tool.search(subquery_text, max_results=5)
                    if results and 'results' in results:
                        for result in results['results']:
                            all_context.append({
                                'text': result.get('content', result.get('snippet', '')),
                                'source': 'web_search',
                                'url': result.get('url', ''),
                                'title': result.get('title', ''),
                                'domain': result.get('domain', ''),
                                'credibility_score': result.get('credibility_score', 0.5),
                                'final_score': result.get('final_score', 0.5),
                                'subquery_id': subquery_id
                            })
                        tool_results[f'web_search_{subquery_id}'] = results
                        sources_used.add('web_search')
                
                elif domain == 'weather':
                    location = self._extract_location(subquery_text) or query_analysis.get('location', 'Hong Kong')
                    weather_results = self._handle_weather(subquery_text, location=location)
                    if weather_results and 'error' not in weather_results:
                        weather_text = self._format_weather_for_context(weather_results)
                        all_context.append({
                            'text': weather_text,
                            'source': 'weather',
                            'credibility_score': 0.95,
                            'final_score': 0.9,
                            'subquery_id': subquery_id
                        })
                        tool_results[f'weather_{subquery_id}'] = weather_results
                        sources_used.add('weather')
                
                elif domain == 'finance':
                    finance_results = self._handle_finance(subquery_text)
                    if finance_results and 'error' not in finance_results:
                        tool_results[f'finance_{subquery_id}'] = finance_results
                        sources_used.add('finance')
                
                elif domain == 'transport':
                    locations = self._extract_locations(subquery_text)
                    if len(locations) >= 2:
                        transport_results = self._handle_transport(subquery_text, origin=locations[0], destination=locations[1])
                        if transport_results and 'error' not in transport_results:
                            transport_text = self._format_transport_for_context(transport_results)
                            all_context.append({
                                'text': transport_text,
                                'source': 'transport',
                                'credibility_score': 0.95,
                                'final_score': 0.9,
                                'subquery_id': subquery_id
                            })
                            tool_results[f'transport_{subquery_id}'] = transport_results
                            sources_used.add('transport')
                
                elif domain == 'time':
                    time_results = self._handle_time(subquery_text)
                    if time_results and 'error' not in time_results:
                        time_text = self._format_time_for_context(time_results)
                        all_context.append({
                            'text': time_text,
                            'source': 'time',
                            'credibility_score': 1.0,
                            'final_score': 1.0,
                            'subquery_id': subquery_id
                        })
                        tool_results[f'time_{subquery_id}'] = time_results
                        sources_used.add('time')
                
                elif domain == 'kb' and not strict_local:
                    # Retrieve from knowledge base for this subquery
                    kb_results = self.retrieval.hybrid_search(subquery_text, top_k=5)
                    for doc in kb_results:
                        all_context.append({
                            'text': doc['text'],
                            'source': 'local_knowledge_base',
                            'metadata': doc.get('metadata', {}),
                            'score': doc.get('score', 0.5),
                            'final_score': doc.get('score', 0.5),
                            'subquery_id': subquery_id
                        })
                    sources_used.add('local_knowledge_base')
                    
            except Exception as e:
                print(f"Error executing subquery {subquery_id} ({domain}): {e}")
                failed_tools.append(f"{domain}_{subquery_id}")
        
        # Sort context by final_score
        all_context.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Synthesize final answer
        report_progress("Synthesizing answer from multiple sources...")
        language = self.simple_detector.detect_language(query)
        
        # Build context text with subquery information
        context_text = ""
        for ctx in all_context[:20]:  # Top 20 contexts
            subquery_id = ctx.get('subquery_id', '')
            context_text += f"\n[Source: {ctx.get('source', 'unknown')}]"
            if subquery_id:
                context_text += f" [Subquery: {subquery_id}]"
            context_text += f"\n{ctx.get('text', '')}\n"
        
        answer = self.llm_router.synthesize_answer(
            query=query,
            context=context_text,
            language=language,
            query_type='factual'
        )
        
        # Build citations
        citations = self._build_citations(all_context)
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            'query': query,
            'answer': answer,
            'routing': {
                'sources': list(sources_used),
                'reasoning': f'Complex query with {len(query_plan["subqueries"])} subqueries',
                'query': query,
                'plan': query_plan
            },
            'sources_used': list(sources_used),
            'tool_results': tool_results,
            'failed_tools': failed_tools,
            'context_count': len(all_context),
            'citations': citations,
            'latency_ms': latency_ms,
            'timestamp': end_time.isoformat(),
            'query_analysis': query_analysis,
            'query_plan': query_plan,
            'complex_query': True
        }
    
    def _handle_finance(self, query: str) -> Dict[str, Any]:
        # Check if this is an FX query first
        fx_pair = self._extract_fx_pair(query)
        if fx_pair:
            from_currency, to_currency = fx_pair
            return self.finance_tool.get_fx_rate(from_currency, to_currency)
        
        # Otherwise, handle as stock query
        tickers = self._extract_tickers(query)
        
        if not tickers:
            return {'error': 'No stock tickers or currency pairs found in query'}
        
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
        
        # If still no location, return error instead of defaulting to New York
        # This prevents wrong-city responses
        if not location:
            return {'error': 'No location specified in query'}
        
        # Check if query is asking for afternoon forecast
        query_lower = query.lower()
        if 'afternoon' in query_lower or 'this afternoon' in query_lower:
            return self.weather_tool.get_afternoon_forecast(location)
        else:
            # Extract date if present
            date = self._extract_date(query)
            return self.weather_tool.get_weather(location, date)
    
    def _handle_transport(self, query: str) -> Dict[str, Any]:
        """Handle transport queries"""
        locations = self._extract_locations(query)
        
        if len(locations) < 2:
            return {'error': 'Need origin and destination for transport query'}
        
        result = self.transport_tool.get_route(locations[0], locations[1], mode='transit')
        
        # Wrap in 'data' key for consistency with other tools
        if 'error' not in result:
            return {'data': result}
        return result
    
    def _handle_time(self, query: str, location: str = '') -> Dict[str, Any]:
        """Handle time queries"""
        # Extract location if not provided
        if not location:
            location = self._extract_location(query)
        
        return self.time_tool.get_current_time(location)
    
    def _is_time_query(self, query: str) -> bool:
        """Check if query is asking for current time"""
        query_lower = query.lower()
        time_keywords = [
            'what time', 'what\'s the time', 'current time', 'time now',
            'what is the time', 'tell me the time', '幾點', '現在幾點',
            '時間', '現在時間', 'what\'s time', 'whats the time'
        ]
        return any(keyword in query_lower for keyword in time_keywords)
    
    def _extract_fx_pair(self, query: str) -> Optional[tuple]:
        """Extract currency pair from FX queries (e.g., HKD/JPY, USD to EUR)"""
        query_upper = query.upper()
        
        # Common currency codes
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'HKD', 'AUD', 'CAD', 'CHF', 'SGD', 'KRW', 'TWD', 'THB']
        
        # Pattern 1: "HKD/JPY", "HKD-JPY", "HKD JPY"
        import re
        for pattern in [r'([A-Z]{3})[/\-\s]([A-Z]{3})', r'([A-Z]{3})\s*(?:to|TO|與|和)\s*([A-Z]{3})']:
            match = re.search(pattern, query_upper)
            if match:
                from_curr, to_curr = match.groups()
                if from_curr in currencies and to_curr in currencies:
                    return (from_curr, to_curr)
        
        # Pattern 2: "港幣" (HKD), "日元" (JPY), "美元" (USD) - Chinese currency names
        chinese_currencies = {
            '港幣': 'HKD', '港元': 'HKD', '日元': 'JPY', '日圓': 'JPY', '美元': 'USD', '美金': 'USD',
            '人民幣': 'CNY', '歐元': 'EUR', '英鎊': 'GBP', '澳元': 'AUD', '加元': 'CAD',
            '新台幣': 'TWD', '台幣': 'TWD', '韓元': 'KRW', '泰銖': 'THB', '新加坡元': 'SGD'
        }
        
        found_currencies = []
        for chinese_name, code in chinese_currencies.items():
            if chinese_name in query:
                found_currencies.append(code)
        
        if len(found_currencies) >= 2:
            return (found_currencies[0], found_currencies[1])
        
        # Pattern 3: Check for "匯率" (exchange rate) keyword with currency codes
        if '匯率' in query or 'exchange rate' in query.lower() or 'forex' in query.lower():
            found_codes = [curr for curr in currencies if curr in query_upper]
            if len(found_codes) >= 2:
                return (found_codes[0], found_codes[1])
        
        return None
    
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
        """Extract origin and destination from transport queries"""
        locations = []
        query_lower = query.lower()
        
        # Pattern 1: "go to Y" or "get to Y" with origin in "in X" or "at X"
        if 'go to ' in query_lower or 'get to ' in query_lower:
            # Extract destination
            if 'go to ' in query_lower:
                dest_part = query_lower.split('go to ')[-1]
            else:
                dest_part = query_lower.split('get to ')[-1]
            
            dest_words = dest_part.strip().rstrip('.,!?;:').split()
            dest = ' '.join(dest_words[:3]) if len(dest_words) >= 3 else dest_part.strip().rstrip('.,!?;:')
            
            # Extract origin from "in X" or "at X" (before "go to" or "get to")
            origin = None
            before_go = query_lower.split('go to')[0] if 'go to' in query_lower else query_lower.split('get to')[0]
            
            if ' in the ' in before_go:
                origin_part = before_go.split(' in the ')[-1].strip()
                # Stop at comma or "what" or "how"
                for stop_word in [',', ' what', ' how', ' where']:
                    if stop_word in origin_part:
                        origin_part = origin_part.split(stop_word)[0].strip()
                        break
                origin = 'the ' + origin_part
            elif ' in ' in before_go:
                origin_part = before_go.split(' in ')[-1].strip()
                # Stop at comma or "what" or "how"
                for stop_word in [',', ' what', ' how', ' where']:
                    if stop_word in origin_part:
                        origin_part = origin_part.split(stop_word)[0].strip()
                        break
                origin = origin_part
            elif ' at the ' in before_go:
                origin_part = before_go.split(' at the ')[-1].strip()
                for stop_word in [',', ' what', ' how', ' where']:
                    if stop_word in origin_part:
                        origin_part = origin_part.split(stop_word)[0].strip()
                        break
                origin = 'the ' + origin_part
            elif ' at ' in before_go:
                origin_part = before_go.split(' at ')[-1].strip()
                for stop_word in [',', ' what', ' how', ' where']:
                    if stop_word in origin_part:
                        origin_part = origin_part.split(stop_word)[0].strip()
                        break
                origin = origin_part
            elif ' from ' in before_go:
                origin_part = before_go.split(' from ')[-1].strip()
                for stop_word in [',', ' what', ' how', ' where']:
                    if stop_word in origin_part:
                        origin_part = origin_part.split(stop_word)[0].strip()
                        break
                origin = origin_part
            
            if origin and dest:
                locations.append(origin)
                locations.append(dest)
        
        # Pattern 2: "from X to Y"
        elif ' from ' in query_lower and ' to ' in query_lower:
            # Split by "from" first
            after_from = query_lower.split(' from ')[-1]
            # Then split by "to"
            if ' to ' in after_from:
                parts = after_from.split(' to ')
                origin = parts[0].strip().rstrip(',')
                dest = parts[1].strip().rstrip('.,!?;:').split()[0:3]
                dest = ' '.join(dest)
                locations.append(origin)
                locations.append(dest)
        
        # Pattern 3: "X to Y" (simple pattern)
        elif ' to ' in query_lower:
            parts = query_lower.split(' to ')
            if len(parts) >= 2:
                # Take last 3 words before "to" as origin
                origin_words = parts[0].strip().split()
                origin = ' '.join(origin_words[-3:]) if len(origin_words) >= 3 else parts[0].strip()
                
                # Take first 3 words after "to" as destination
                dest_words = parts[1].strip().rstrip('.,!?;:').split()
                dest = ' '.join(dest_words[:3]) if len(dest_words) >= 3 else parts[1].strip().rstrip('.,!?;:')
                
                locations.append(origin)
                locations.append(dest)
        
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
            if source in {'weather', 'finance', 'transport', 'finance_web_extraction', 'time'}:
                return True
            
            # Check if we have a good score from retrieval/reranking
            score = doc.get('final_score') or doc.get('score') or doc.get('credibility_score')
            if score is not None and score >= 0.5:
                return True
        
        return False
    
    def _format_weather_for_context(self, weather_results: Dict[str, Any]) -> str:
        """Format weather data in a structured, readable way for LLM context"""
        lines = []
        
        # Basic info
        location = weather_results.get('location', 'Unknown')
        provider = weather_results.get('provider', 'unknown')
        lines.append(f"Weather data for {location} (provider: {provider})")
        
        # Current time and timezone
        if 'current_time' in weather_results:
            lines.append(f"Current time: {weather_results['current_time']}")
        if 'timezone' in weather_results:
            lines.append(f"Timezone: {weather_results['timezone']}")
        
        # Current weather (if available)
        if 'temperature' in weather_results:
            lines.append(f"\nCurrent conditions:")
            lines.append(f"  Temperature: {weather_results.get('temperature')}°C")
            if 'weather_description' in weather_results:
                lines.append(f"  Condition: {weather_results['weather_description']}")
            if 'precipitation' in weather_results:
                lines.append(f"  Precipitation: {weather_results.get('precipitation')} mm")
            if 'windspeed' in weather_results:
                lines.append(f"  Wind speed: {weather_results.get('windspeed')} km/h")
            if 'humidity' in weather_results:
                lines.append(f"  Humidity: {weather_results.get('humidity')}%")
        
        # Afternoon forecast (if available)
        if 'afternoon_data' in weather_results:
            afternoon_data = weather_results['afternoon_data']
            if afternoon_data:
                lines.append(f"\nAfternoon forecast ({weather_results.get('afternoon_window', '12:00-18:00')}):")
                lines.append(f"Status: {weather_results.get('afternoon_status', 'unknown')}")
                lines.append(f"\nHourly breakdown:")
                for hour in afternoon_data:
                    time = hour.get('time', 'unknown')
                    temp = hour.get('temperature', 'N/A')
                    desc = hour.get('weather_description', 'N/A')
                    precip = hour.get('precipitation', 0)
                    wind = hour.get('windspeed', 'N/A')
                    humidity = hour.get('humidity', 'N/A')
                    lines.append(f"  {time}: {temp}°C, {desc}, precipitation: {precip}mm, wind: {wind}km/h, humidity: {humidity}%")
        
        return '\n'.join(lines)
    
    def _format_time_for_context(self, time_results: Dict[str, Any]) -> str:
        """Format time data in a structured, readable way for LLM context"""
        lines = []
        
        location = time_results.get('location', 'Unknown')
        timezone = time_results.get('timezone', 'Unknown')
        
        lines.append(f"Current time for {location} ({timezone}):")
        lines.append(f"  Date and time: {time_results.get('formatted_time')}")
        lines.append(f"  Day of week: {time_results.get('day_of_week_name')}")
        lines.append(f"  Timezone: {timezone}")
        lines.append(f"  UTC offset: {time_results.get('utc_offset')}")
        
        return '\n'.join(lines)
    
    def _format_transport_for_context(self, transport_results: Dict[str, Any]) -> str:
        """Format transport data in a structured, readable way for LLM context"""
        data = transport_results.get('data', {})
        lines = []
        
        origin = data.get('origin', 'Unknown')
        destination = data.get('destination', 'Unknown')
        
        lines.append(f"Transport route from {origin} to {destination}:")
        
        route = data.get('route', {})
        if route:
            total_duration = route.get('total_duration_minutes', 0)
            lines.append(f"  Total duration: {total_duration} minutes")
            
            departure_time = route.get('departure_time', '')
            arrival_time = route.get('arrival_time', '')
            if departure_time:
                lines.append(f"  Departure: {departure_time}")
            if arrival_time:
                lines.append(f"  Arrival: {arrival_time}")
            
            steps = route.get('steps', [])
            if steps:
                lines.append(f"\n  Route steps ({len(steps)} segments):")
                for i, step in enumerate(steps, 1):
                    step_type = step.get('type', 'unknown')
                    mode = step.get('mode', 'unknown')
                    
                    if step_type == 'transit':
                        transit = step.get('transit', {})
                        line = transit.get('line', 'N/A')
                        headsign = transit.get('headsign', 'N/A')
                        from_stop = step.get('from', 'N/A')
                        to_stop = step.get('to', 'N/A')
                        lines.append(f"    {i}. Take {mode} {line} (towards {headsign})")
                        lines.append(f"       From: {from_stop}")
                        lines.append(f"       To: {to_stop}")
                    elif step_type == 'pedestrian':
                        from_loc = step.get('from', 'N/A')
                        to_loc = step.get('to', 'N/A')
                        lines.append(f"    {i}. Walk from {from_loc} to {to_loc}")
        
        return '\n'.join(lines)
    
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
