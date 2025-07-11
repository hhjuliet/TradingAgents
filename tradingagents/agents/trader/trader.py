import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        context = {
            "role": "user",
            "content": f"基于分析师团队的综合分析，以下是为{company_name}量身定制的投资计划。该计划融合了当前技术面、宏观经济和社交媒体情绪的洞察。请以此为基础，评估你的下一个交易决策。\n\n建议投资计划: {investment_plan}\n\n请利用这些洞察做出明智且有策略的决策。",
        }

        messages = [
            {
                "role": "system",
                "content": f"你是一名交易代理，分析市场数据做出投资决策。请基于你的分析，给出明确的买入、卖出或持有建议。最后务必以'最终建议：**买入/持有/卖出**'结尾，确认你的建议。不要忘记结合过往决策的经验教训。以下是你在类似情形下的反思与经验：{past_memory_str}",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
        }

    return trader_node
