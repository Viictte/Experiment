"""Query planner V2 - LLM-based planning with no hardcoded rules"""

import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
import os
from rag_system.core.config import get_config


class QueryPlannerV2:
    """
    Advanced query planner using LLM to decompose queries intelligently.
    No hardcoded rules - all logic driven by LLM reasoning.
    """
    
    def __init__(self):
        self.config = get_config()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"  # Fast mode for planning
        
    def plan_query(self, query: str) -> Dict[str, Any]:
        """
        Create an intelligent execution plan for any user query.
        
        Args:
            query: User's query (any language, any complexity)
            
        Returns:
            Dict with mode, subqueries, and routing hints
        """
        try:
            prompt = self._build_planner_prompt(query)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intelligent query planning assistant for a RAG system. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            plan = json.loads(content)
            
            # Validate plan structure
            if not isinstance(plan, dict) or 'mode' not in plan:
                raise ValueError("Invalid plan structure")
            
            # Ensure subqueries is a list
            if 'subqueries' not in plan:
                plan['subqueries'] = []
            elif not isinstance(plan['subqueries'], list):
                plan['subqueries'] = []
            
            # Limit to 5 subqueries
            if len(plan['subqueries']) > 5:
                plan['subqueries'] = plan['subqueries'][:5]
            
            return plan
            
        except Exception as e:
            # Fallback: treat as single retrieval
            print(f"Planning failed: {e}")
            return {
                "mode": "single_retrieval",
                "reason": f"Planning error: {str(e)}",
                "subqueries": [
                    {
                        "id": "q1",
                        "description": "Answer the user's question",
                        "query": query,
                        "tool": "web_search",
                        "priority": 1
                    }
                ]
            }
    
    def _build_planner_prompt(self, query: str) -> str:
        """Build the planner prompt with examples and instructions"""
        return f"""You are a query planning module for an advanced RAG (Retrieval-Augmented Generation) system.

Your task is to analyze the user's query and create an optimal execution plan.

## Available Tools
- **web_search**: General web search for factual information, news, reviews, ratings
- **kb**: Local knowledge base (documents, PDFs)
- **weather**: Weather forecasts, temperature, conditions, air quality
- **finance**: Stock prices, exchange rates, market data
- **transport**: Directions, routes, travel times, public transit
- **time**: Current time in any location or timezone
- **vision**: Image recognition and description (if user uploads image)

## Output Format
Return ONLY valid JSON with this structure:
```json
{{
  "mode": "direct_llm" | "single_retrieval" | "multi_retrieval",
  "reason": "brief explanation of your decision",
  "subqueries": [
    {{
      "id": "q1",
      "description": "natural language description of what to find",
      "query": "optimized search query (keywords, entities, no full sentences)",
      "tool": "web_search|kb|weather|finance|transport|time|vision",
      "priority": 1
    }}
  ]
}}
```

## Mode Selection Guidelines

**direct_llm**: Use when:
- Simple arithmetic or logic (e.g., "1024 - 768 = ?")
- Creative writing requests (e.g., "Write an encouraging message")
- Greetings or casual conversation (e.g., "Hello", "How are you?")
- Noise or meaningless input (e.g., ".", "???", "asdfgh")
- Questions answerable from general knowledge without retrieval
- For direct_llm mode, set subqueries to empty array []

**single_retrieval**: Use when:
- Query needs ONE piece of external information
- Single domain (weather, finance, transport, web search, etc.)
- Example: "What's the weather in Hong Kong tomorrow?"
- Create 1 subquery with appropriate tool

**multi_retrieval**: Use when:
- Query has multiple distinct parts (A AND B)
- Needs information from multiple domains
- Example: "Who won best actor at Hong Kong Film Awards AND what's the Douban rating?"
- Create separate subqueries for each part (max 5)

## Query Optimization Rules

1. **Rephrase for search engines**: Convert natural questions to keyword queries
   - Bad: "Can you tell me what the weather will be like tomorrow?"
   - Good: "Hong Kong weather forecast tomorrow"

2. **Extract key entities**: Focus on names, places, dates, specific items
   - Query: "誰在最近一屆香港電影金像獎中獲得了最佳男主角？"
   - Optimized: "最新一届 香港电影金像奖 最佳男主角 获奖"

3. **Keep language consistent**: Use same language as user's query

4. **Make subqueries independent**: Each subquery should be self-contained, not reference other subqueries

5. **Choose correct tool**: 
   - Weather/temperature/air quality → weather
   - Stock prices/exchange rates → finance
   - Directions/routes/travel → transport
   - Current time → time
   - Factual info/news/reviews/ratings → web_search
   - User's documents → kb

## Examples

**Example 1: Noise Input**
User: "."
Output:
```json
{{
  "mode": "direct_llm",
  "reason": "Input is meaningless noise, no retrieval needed",
  "subqueries": []
}}
```

**Example 2: Simple Math**
User: "1024 減去 768 等於多少？"
Output:
```json
{{
  "mode": "direct_llm",
  "reason": "Simple arithmetic, no external information needed",
  "subqueries": []
}}
```

**Example 3: Single Domain (Weather)**
User: "請告訴我明天早上香港將軍澳的天氣"
Output:
```json
{{
  "mode": "single_retrieval",
  "reason": "Single weather query for specific location",
  "subqueries": [
    {{
      "id": "q1",
      "description": "Get weather forecast for Tseung Kwan O tomorrow morning",
      "query": "香港將軍澳 明天早上 天氣預報",
      "tool": "weather",
      "priority": 1
    }}
  ]
}}
```

**Example 4: Multi-part Query**
User: "誰在最近一屆香港電影金像獎中獲得了最佳男主角？他獲獎電影的豆瓣評分是多少？"
Output:
```json
{{
  "mode": "multi_retrieval",
  "reason": "Two distinct questions: (1) award winner, (2) Douban rating",
  "subqueries": [
    {{
      "id": "q1",
      "description": "Find latest Hong Kong Film Awards Best Actor winner and film",
      "query": "最新一届 香港电影金像奖 最佳男主角 获奖 电影",
      "tool": "web_search",
      "priority": 1
    }},
    {{
      "id": "q2",
      "description": "Find Douban rating of the winning film",
      "query": "香港电影金像奖 最佳男主角 获奖电影 豆瓣 评分",
      "tool": "web_search",
      "priority": 2
    }}
  ]
}}
```

**Example 5: Cross-domain Query**
User: "幫我規劃明天在香港中環的一日行程：包含天氣預報和從K11 MUSEA去香港科技大學的交通路線"
Output:
```json
{{
  "mode": "multi_retrieval",
  "reason": "Needs weather forecast AND transport directions",
  "subqueries": [
    {{
      "id": "q1",
      "description": "Get weather forecast for Central, Hong Kong tomorrow",
      "query": "香港中環 明天 天氣預報",
      "tool": "weather",
      "priority": 1
    }},
    {{
      "id": "q2",
      "description": "Get directions from K11 MUSEA to HKUST",
      "query": "K11 MUSEA 香港科技大學 交通路線",
      "tool": "transport",
      "priority": 2
    }}
  ]
}}
```

Now analyze this user query and create an optimal plan:

User Query: {query}

Output (JSON only, no explanation):"""


# Singleton instance
_planner_v2 = None

def get_planner_v2() -> QueryPlannerV2:
    """Get or create the query planner V2 singleton"""
    global _planner_v2
    if _planner_v2 is None:
        _planner_v2 = QueryPlannerV2()
    return _planner_v2
