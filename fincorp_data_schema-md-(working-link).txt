# FinCorp Data Structure & Schema Contracts

**Document Owner:** NexusFlow AI Architecture Team  
**Client:** FinCorp  
**Project:** Enterprise RAG Pipeline Orchestration  
**Date:** October 24, 2023  
**Status:** Approved for Implementation

---

## Overview

This document outlines the strict data schemas required for the FinCorp orchestration pipeline. Because FinCorp’s downstream automated reporting tools are highly sensitive to schema drift and LLM hallucination, all unstructured and semi-structured text extracted by the LangChain workflow must be strictly cast into the following JSON contracts using Pydantic. 

Failure to adhere to these exact field names and types will result in fatal errors in the FinCorp reporting dashboard.

---

## 1. Financial Summary Schema (`financial_summary`)

This schema is used when the pipeline routes a query to the structured financial database or processes quarterly earnings reports. 

### Field Specifications

*   **`company_name`** (`string`): The official ticker or registered name of the entity (e.g., "FinCorp Inc.", "AAPL").
*   **`quarter`** (`string`): The fiscal quarter and year referenced in the query. Must follow the format "QX YYYY" (e.g., "Q3 2023").
*   **`total_revenue`** (`float`): The total consolidated revenue reported for the specified quarter, represented as a numeric float in millions (e.g., `12450.50`).
*   **`net_income`** (`float`): The bottom-line net income reported for the specified quarter, represented as a numeric float in millions (e.g., `3120.75`).

### Expected JSON Output

```json
{
  "company_name": "TechGlobal Corp",
  "quarter": "Q2 2023",
  "total_revenue": 14500.00,
  "net_income": 2100.50
}
```

---

## 2. Market News Schema (`market_news`)

This schema is used when the pipeline routes a query to the unstructured market news aggregator. It is designed to synthesize broad market noise into a clean, actionable sentiment overview.

### Field Specifications

*   **`company_name`** (`string`): The target company being analyzed in the news aggregator.
*   **`sentiment_score`** (`float`): A calculated sentiment score derived from the aggregated news. Must be a float between `-1.0` (Highly Negative) and `1.0` (Highly Positive).
*   **`top_headlines`** (`array of strings`): A list containing exactly the top 3 most relevant news headlines impacting the company's market perception.

### Expected JSON Output

```json
{
  "company_name": "TechGlobal Corp",
  "sentiment_score": 0.65,
  "top_headlines": [
    "TechGlobal announces breakthrough in enterprise AI",
    "Analyst upgrades TechGlobal stock to 'Strong Buy'",
    "TechGlobal faces minor supply chain disruptions in Q2"
  ]
}
```

---

## Implementation Notes for NexusFlow Engineering

1. **Pydantic Enforcement:** You must create two distinct Pydantic classes (`FinancialSummary` and `MarketNews`) mapping perfectly to the above specifications. Use Pydantic's `Field` descriptions to instruct the `GPT-4o-mini` model on how to correctly extract and format these values.
2. **Routing:** Your LCEL router chain (`RunnableLambda`) must dynamically determine whether a user's prompt is asking for structured financials or unstructured news sentiment, and pipe the LLM output into the correct Pydantic output parser to guarantee the JSON contracts shown above.