"""Answer evaluator - checks completeness and suggests follow-up queries"""

import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
import os
from rag_system.core.config import get_config


class AnswerEvaluator:
    """
    Evaluates answer completeness and suggests follow-up queries.
    No hardcoded rules - all logic driven by LLM reasoning.
    """
    
    def __init__(self):
        self.config = get_config()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"
        
    def evaluate(self, question: str, answer: str, evidence_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate if answer fully addresses the question.
        
        Args:
            question: Original user question
            answer: Generated answer to evaluate
            evidence_summary: Optional summary of evidence used
            
        Returns:
            Dict with completeness assessment and follow-up queries
        """
        try:
            prompt = self._build_evaluator_prompt(question, answer, evidence_summary)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict answer quality evaluator. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            evaluation = json.loads(content)
            
            # Validate structure
            if not isinstance(evaluation, dict) or 'complete' not in evaluation:
                raise ValueError("Invalid evaluation structure")
            
            # Ensure followup_subqueries is a list
            if 'followup_subqueries' not in evaluation:
                evaluation['followup_subqueries'] = []
            elif not isinstance(evaluation['followup_subqueries'], list):
                evaluation['followup_subqueries'] = []
            
            # Limit to 3 follow-up queries
            if len(evaluation['followup_subqueries']) > 3:
                evaluation['followup_subqueries'] = evaluation['followup_subqueries'][:3]
            
            return evaluation
            
        except Exception as e:
            # Fallback: assume complete
            print(f"Evaluation failed: {e}")
            return {
                "complete": True,
                "completeness_score": 0.5,
                "issues": [f"Evaluation error: {str(e)}"],
                "followup_subqueries": []
            }
    
    def _build_evaluator_prompt(self, question: str, answer: str, evidence_summary: Optional[str]) -> str:
        """Build the evaluator prompt"""
        evidence_section = ""
        if evidence_summary:
            evidence_section = f"\n\n## Evidence Used\n{evidence_summary}\n"
        
        return f"""You are a strict answer quality evaluator for a RAG system.

Your task is to evaluate whether the answer FULLY addresses the user's question.

## Evaluation Criteria

1. **Completeness**: Does the answer provide ALL information requested in the question?
2. **Specificity**: Are specific facts, numbers, names, dates provided when asked?
3. **Accuracy**: Is the answer consistent with the evidence (if provided)?

## Output Format

Return ONLY valid JSON with this structure:
```json
{{
  "complete": true | false,
  "completeness_score": 0.0-1.0,
  "issues": ["list of specific missing items or problems"],
  "followup_subqueries": [
    {{
      "id": "f1",
      "description": "what specific information to find",
      "query": "optimized search query",
      "tool": "web_search|weather|finance|transport|time|kb",
      "priority": 1
    }}
  ]
}}
```

## Guidelines

**complete = true** when:
- All parts of the question are answered
- Specific requested information is provided (names, numbers, dates, etc.)
- Answer is confident and definitive (not vague or hedging)

**complete = false** when:
- Any part of the question is unanswered
- Answer says "information not available" but question clearly expects specific data
- Answer is vague or incomplete for multi-part questions
- Missing specific facts that were explicitly requested

**issues**: List SPECIFIC missing items
- Bad: "Answer is incomplete"
- Good: "Missing Douban rating for the film mentioned"
- Good: "Does not specify which year's award ceremony"

**followup_subqueries**: Generate TARGETED search queries to fill gaps
- Focus on the SPECIFIC missing information
- Use keywords and entities from the question and answer
- Choose appropriate tool (web_search for most factual queries)
- Make queries self-contained and searchable

## Examples

**Example 1: Complete Answer**
Question: "1024 減去 768 等於多少？"
Answer: "256"
Output:
```json
{{
  "complete": true,
  "completeness_score": 1.0,
  "issues": [],
  "followup_subqueries": []
}}
```

**Example 2: Partially Complete (Missing Rating)**
Question: "誰在最近一屆香港電影金像獎中獲得了最佳男主角？他獲獎電影的豆瓣評分是多少？"
Answer: "最近一屆香港電影金像獎（第43屆）的最佳男主角是劉青雲，他憑藉電影《爸爸》獲獎。然而，關於這部獲獎電影《爸爸》的豆瓣評分，在現有的資料中並未提供。"
Output:
```json
{{
  "complete": false,
  "completeness_score": 0.5,
  "issues": [
    "Missing Douban rating for the film '爸爸' (Dad)",
    "Answer acknowledges missing information but question expects specific rating"
  ],
  "followup_subqueries": [
    {{
      "id": "f1",
      "description": "Find Douban rating for the film '爸爸' starring 劉青雲",
      "query": "爸爸 劉青雲 豆瓣 评分",
      "tool": "web_search",
      "priority": 1
    }}
  ]
}}
```

**Example 3: Overly Cautious Answer**
Question: "What's the weather in Hong Kong tomorrow?"
Answer: "Based on the available data, it appears the temperature in Hong Kong tomorrow may be around 25°C with partly cloudy conditions, though I cannot confirm this with complete certainty."
Evidence: "Hong Kong weather forecast: Tomorrow 25°C, partly cloudy, humidity 70%"
Output:
```json
{{
  "complete": true,
  "completeness_score": 0.9,
  "issues": [
    "Answer is overly cautious despite having clear evidence"
  ],
  "followup_subqueries": []
}}
```

**Example 4: Multi-part Question, Fully Answered**
Question: "請告訴我明天香港中環的天氣，並說明是否適合帶小朋友去海邊玩。"
Answer: "明天香港中環的天氣預報為晴天，氣溫約28°C，濕度65%，風力較小。這樣的天氣非常適合帶小朋友去海邊玩，陽光充足但不會太熱，建議做好防曬措施並攜帶足夠的飲用水。"
Output:
```json
{{
  "complete": true,
  "completeness_score": 1.0,
  "issues": [],
  "followup_subqueries": []
}}
```

Now evaluate this answer:

## Question
{question}

## Answer
{answer}
{evidence_section}

Output (JSON only, no explanation):"""


# Singleton instance
_evaluator = None

def get_evaluator() -> AnswerEvaluator:
    """Get or create the answer evaluator singleton"""
    global _evaluator
    if _evaluator is None:
        _evaluator = AnswerEvaluator()
    return _evaluator
