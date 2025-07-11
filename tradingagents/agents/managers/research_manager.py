import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作为投资组合经理和辩论主持人，你需要对本轮辩论进行批判性评估，并做出明确决策：支持空头、支持多头，或仅在有充分理由时选择观望。

请简明总结双方最有说服力的要点，推荐买入、卖出或观望，必须明确可执行，不能因为双方都合理就默认观望，要基于最有力的论据做出选择。

此外，请为交易员制定详细的投资计划，包括：
- 你的建议：基于最有说服力的论据做出明确立场。
- 理由：解释为何这些论据支持你的结论。
- 策略行动：具体执行建议。
请结合你在类似情形中的过往教训，优化决策，确保持续改进。分析请用自然对话方式表达，无需特殊格式。

以下是你过往的反思：
\"{past_memory_str}\"

本轮辩论内容：
辩论历史：
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
