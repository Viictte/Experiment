# RAG Workflow V2 Design

## Overview
Redesigned workflow with LLM-based planner and evaluator (no hardcoded rules).

## Architecture

```
Query
  ↓
[Fast Path Check] → Simple queries bypass to direct LLM
  ↓
Planner (LLM) → {mode, subqueries, routing_hint}
  ↓
First Execution (retrieval + synthesis OR direct LLM)
  ↓
Evaluator (LLM) → {complete?, missing_items, extra_queries}
  ├─ complete → return answer
  └─ incomplete + extra_queries
       ↓
     Targeted retrieval + re-synthesis
       ↓
     return answer (max 2 passes total)
```

## Components

### 1. Planner (LLM-based)
**Input**: User query
**Output**: JSON plan
```json
{
  "mode": "direct_llm" | "single_retrieval" | "multi_retrieval",
  "reason": "short explanation",
  "subqueries": [
    {
      "id": "q1",
      "description": "natural language description",
      "query": "search-optimized query string",
      "tool": "web_search" | "kb" | "weather" | "finance" | "transport" | "time" | "vision",
      "priority": 1
    }
  ]
}
```

**Behaviors**:
- Noise inputs (".", "???") → `mode: "direct_llm"`, empty subqueries
- Simple queries (math, greetings) → `mode: "direct_llm"`
- Single-domain queries → `mode: "single_retrieval"`, 1 subquery
- Multi-part queries → `mode: "multi_retrieval"`, multiple subqueries
- Rephrases queries for optimal keyword matching
- Decides tool routing per subquery

### 2. Evaluator (LLM-based)
**Input**: Original question, synthesized answer, evidence summary
**Output**: JSON evaluation
```json
{
  "complete": true | false,
  "completeness_score": 0.0-1.0,
  "issues": ["list of missing items"],
  "followup_subqueries": [
    {
      "id": "f1",
      "description": "what to find",
      "query": "search query",
      "tool": "web_search",
      "priority": 1
    }
  ]
}
```

**Behaviors**:
- Compares question vs answer to find gaps
- Identifies specific missing information
- Generates targeted search queries to fill gaps
- Does NOT invent data or hallucinate

### 3. Improved Synthesis
**Changes**:
- More confident when evidence is strong
- Less apologetic disclaimers
- Only mentions limitations when truly insufficient
- Concise, decisive sentences

## Iteration Logic
- Max 2 passes per query
- Pass 1: plan → retrieve → synthesize → evaluate
- Pass 2 (if incomplete): targeted retrieval → re-synthesize
- Stop conditions:
  - `complete == true`
  - No followup queries
  - Already did refinement pass

## Test Cases (New, Never Seen Before)

### Multi-part Cross-domain
1. "請告訴我最近一屆香港電影金像獎的最佳男主角是誰？並給出他所主演得獎電影在豆瓣上的評分與評價摘要。"
   - Expected: Winner name + film + Douban rating + review summary

2. "幫我規劃明天在香港中環的一日行程：包含天氣預報、適合的穿著建議，以及從K11 MUSEA去香港科技大學往返的交通路線與時間預估。"
   - Expected: Weather + clothing advice + transport routes with times

### Simple / Direct LLM
3. "1024 減去 768 等於多少？"
   - Expected: Direct answer, no retrieval

4. "寫一句鼓勵正在準備考試的學生的話。"
   - Expected: Direct LLM generation, no retrieval

### Noise Inputs
5. "."
   - Expected: Direct LLM handles gracefully

6. "？？？？"
   - Expected: Direct LLM handles gracefully

### Single-domain Retrieval
7. "請告訴我明天早上香港將軍澳的天氣，並說明是否適合帶小朋友去海邊玩。"
   - Expected: Weather data + recommendation

8. "請查一下騰訊控股（0700.HK）今天的股價大約是多少，以及和上週同一時間相比漲跌幅如何？"
   - Expected: Current price + comparison

9. "我現在在深圳灣口岸，想在三小時內到達香港科技大學，請給我兩條不同的路線選擇（包括時間和交通工具）。"
   - Expected: 2 route options with times and modes

10. "今年誰獲得了諾貝爾物理學獎？簡要說明得獎原因。"
    - Expected: Winner + reason

### Evaluator Stress (Partial Answer Detection)
11. "請告訴我最近三屆香港電影金像獎最佳男主角得獎者及其得獎電影，並比較三部電影在豆瓣上的評分高低。"
    - Expected: 3 winners + 3 films + 3 ratings + comparison
    - Tests: Evaluator should detect if any ratings missing

## Implementation Plan
1. Create `planner_v2.py` with improved LLM-based planner
2. Create `evaluator.py` with LLM-based completeness checker
3. Update `llm_router.py` synthesis prompt for more confidence
4. Modify `rag_workflow.py` to use new planner + evaluator
5. Test with all 11 test cases
6. Compare before/after results
