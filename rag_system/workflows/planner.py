"""Query planner for complex multi-step retrieval"""

import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
import os
from rag_system.core.config import get_config


class QueryPlanner:
    """
    Plans complex queries by decomposing them into sub-queries.
    Uses DeepSeek fast mode for efficient planning.
    """
    
    def __init__(self):
        self.config = get_config()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"  # Fast mode for planning
        
    def analyze_complexity(self, query: str) -> str:
        """
        Quickly determine if a query is simple or complex.
        
        Args:
            query: User's query
            
        Returns:
            "simple" or "complex"
        """
        # Heuristic-based complexity detection
        query_lower = query.lower()
        
        # Check for multiple questions
        question_markers = ['?', '？']
        question_count = sum(query.count(marker) for marker in question_markers)
        
        # Check for conjunctions indicating multiple parts
        conjunctions = [
            'and', '並且', '并且', '同时', '同時', '以及', '还有', '還有',
            'also', 'additionally', 'furthermore', 'moreover'
        ]
        has_conjunctions = any(conj in query_lower for conj in conjunctions)
        
        # Check for enumeration
        has_enumeration = any(marker in query for marker in ['1.', '2.', '3.', '①', '②', '③'])
        
        # Check for multiple domains (complex queries often span domains)
        domain_keywords = {
            'weather': ['weather', 'temperature', 'rain', '天气', '气温', '下雨'],
            'finance': ['stock', 'price', 'market', '股票', '价格', '市场'],
            'transport': ['route', 'bus', 'train', '路线', '公交', '地铁'],
            'time': ['time', 'when', 'date', '时间', '什么时候', '日期']
        }
        
        domains_found = 0
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains_found += 1
        
        # Determine complexity
        if question_count > 1 or has_enumeration or domains_found > 1:
            return "complex"
        elif has_conjunctions and len(query) > 100:
            return "complex"
        else:
            return "simple"
    
    def plan_query(self, query: str) -> Dict[str, Any]:
        """
        Create a structured plan for answering a complex query.
        
        Args:
            query: User's query
            
        Returns:
            Dict with complexity and subqueries
        """
        complexity = self.analyze_complexity(query)
        
        if complexity == "simple":
            return {
                "complexity": "simple",
                "subqueries": []
            }
        
        # Use DeepSeek to plan complex queries
        try:
            prompt = f"""You are a planning module for a RAG system. Your task is to break a user's question into a small number of retrieval sub-queries.

Output strictly in JSON with keys: "complexity" and "subqueries".
- "complexity" is "simple" or "complex".
- Each subquery has: "id", "description", "domain", "query", "priority".
- "domain" is one of: "web_search", "weather", "finance", "transport", "kb", "time", "vision".
- "query" should be a standalone search query string; do not reference answers from other subqueries.
- Keep subqueries independent and self-contained.
- Maximum 5 subqueries.

User question: {query}

Output JSON only, no explanation:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query planning assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON if wrapped in markdown
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            plan = json.loads(content)
            
            # Validate plan structure
            if not isinstance(plan, dict) or 'complexity' not in plan or 'subqueries' not in plan:
                raise ValueError("Invalid plan structure")
            
            # Ensure subqueries is a list
            if not isinstance(plan['subqueries'], list):
                plan['subqueries'] = []
            
            # Limit to 5 subqueries
            if len(plan['subqueries']) > 5:
                plan['subqueries'] = plan['subqueries'][:5]
            
            return plan
            
        except Exception as e:
            # Fallback: treat as simple if planning fails
            print(f"Planning failed: {e}")
            return {
                "complexity": "simple",
                "subqueries": []
            }


# Singleton instance
_planner = None

def get_planner() -> QueryPlanner:
    """Get or create the query planner singleton"""
    global _planner
    if _planner is None:
        _planner = QueryPlanner()
    return _planner
